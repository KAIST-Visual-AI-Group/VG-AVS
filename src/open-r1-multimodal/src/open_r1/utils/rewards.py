import re
from math_verify import parse, verify
from Levenshtein import ratio

from open_r1.utils.string_utils import (
    extract_choice,
    clean_text,
    extract_thinking_and_action_string,
    action_string_to_list,
)

import os
from datetime import datetime
from typing import List
import numpy as np
from open_r1.utils.procthor_utils import (
    build_additional_view,
    get_procthor_controller,
    _get_rank_info,
    _check_action_feasibility_fast,
)
import PIL
from PIL import Image
import torch
from typing import Optional
from open_r1.utils.model_load import initialize_verifier
from open_r1.utils.visualization import create_visualization
DEBUG_MODE = str(os.getenv("DEBUG_MODE", "0")) == "1"

def _get_local_rank():
    """Get local rank for debug output control."""
    return int(os.getenv("LOCAL_RANK", "0"))

def _get_debug_output_dir(**kwargs):
    """Get debug output directory: output/log/{model_name}/ and output/log/visualizations/{model_name}/"""
    # Get model name from kwargs
    model_name = "unknown_model"
    
    # Try to get from training_args
    training_args = kwargs.get("training_args")
    if training_args:
        if hasattr(training_args, "model_name_or_path"):
            model_name = training_args.model_name_or_path
        elif hasattr(training_args, "model_name"):
            model_name = training_args.model_name
        elif hasattr(training_args, "output_dir"):
            # Extract model name from output_dir if it contains model name
            output_dir = training_args.output_dir
            # Try to extract model name from path like "output/grpo-procthor-7b-..."
            if "output" in output_dir:
                parts = output_dir.split("/")
                for part in parts:
                    if part and part != "output" and not part.endswith(".txt"):
                        model_name = part
                        break
    
    # Try to get from kwargs directly
    if model_name == "unknown_model":
        model_name = kwargs.get("model_name_or_path") or kwargs.get("model_name", "unknown_model")
    
    # Extract just the model name (remove path)
    if "/" in str(model_name):
        model_name = str(model_name).split("/")[-1]
    
    # Create directories
    log_dir = os.path.join("output", "log", model_name)
    vis_dir = os.path.join("output", "log", "visualizations", model_name)
    
    return log_dir, vis_dir

def _add_question_type_prompt(question: str, question_type: Optional[str] = None) -> str:
    """Add question type specific prompt to question based on test_procthor_action_accuracy.py logic.
    
    Args:
        question: Original question text
        question_type: Type of question ('existence', 'counting', 'state', 'OCR', etc.)
    
    Returns:
        Question with type-specific prompt appended
    """
    if question_type is None:
        return question
    
    question_type_lower = question_type.strip().lower()
    if question_type_lower == 'existence':
        question += "\nPlease answer yes or no."
    elif question_type_lower == 'counting':
        question += "\nAnswer with a single number only."
    elif question_type_lower == 'state':
        question += "\nPlease answer yes or no."
    elif question_type_lower == 'ocr':
        question += "\nAnswer with a single number only."
    
    return question

@torch.no_grad()
def _verifier_answer(images: List, question: str, verifier_model=None, verifier_processor=None, max_new_tokens=48, question_type: Optional[str] = None) -> Optional[str]:
    """Get answer from verifier model."""
    if verifier_model is None or verifier_processor is None:
        model, processor = initialize_verifier()
    else:
        model = verifier_model
        processor = verifier_processor
    try:
        pil_images = []
        for image in images:
            if isinstance(image, str):
                pil_images.append(Image.open(image).convert('RGB'))
            elif isinstance(image, PIL.Image.Image):
                pil_images.append(image.convert('RGB'))
            else:
                raise ValueError(f"Invalid image type: {type(image)}")
        
        # Add question type specific prompt if provided
        question_with_prompt = _add_question_type_prompt(question, question_type)
        
        messages = [{
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in pil_images],
                # {"type": "text", "text": f"{question}\nOutput the single word (yes/no, number, etc.) only."}
                {"type": "text", "text": question_with_prompt}

            ]
        }]
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat_text], images=pil_images, return_tensors="pt").to(model.device)
        gen = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        out = processor.batch_decode(gen, skip_special_tokens=True)[0]
        
        ans_blocks = re.findall(r'<answer>(.*?)</answer>', out, re.DOTALL | re.IGNORECASE)
        if ans_blocks:
            ans = ans_blocks[-1].strip().rstrip('.').strip()
        else:
            ans = out.strip().split('\n')[-1].strip().rstrip('.').strip()
        
        return out, ans
    except Exception as e:
        if DEBUG_MODE and _get_local_rank() == 0:
            print(f"[verifier] Inference failed: {e}")
        return None, None


def default_accuracy_reward(content, sol, **kwargs):
    reward = 0.0
    # Extract answer from solution if it has think/answer tags
    sol_match = re.search(r"<answer>(.*?)</answer>", sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

    # Extract answer from content if it has think/answer tags
    content_matches = re.findall(r"<answer>(.*?)</answer>", content, re.DOTALL)
    student_answer = content_matches[-1].strip() if content_matches else content.strip()

    # Check for yes/no answers FIRST (before other methods)
    # This handles cases where verifier says "Yes, because..." or "No, because..."
    ground_yes_no = re.search(r"\b(yes|no)\b[.!?,;:]?", ground_truth.lower())
    student_yes_no = re.search(r"\b(yes|no)\b[.!?,;:]?", student_answer.lower())

    if ground_yes_no and student_yes_no:
        # Both have yes/no, compare them
        reward = 1.0 if ground_yes_no.group(1) == student_yes_no.group(1) else 0.0
        return reward
    elif ground_yes_no:
        # Ground truth is yes/no but student didn't answer yes/no
        return 0.0

    # Try symbolic verification for numeric answers
    try:
        answer = parse(student_answer)
        if float(verify(answer, parse(ground_truth))) > 0:
            reward = 1.0
    except Exception:
        pass  # Continue to next verification method if this fails

    # If symbolic verification failed, try string matching or fuzzy matching
    if reward == 0.0:
        try:
            # Check if ground truth contains numbers
            has_numbers = bool(re.search(r"\d+", ground_truth))
            # Check if it's a multiple choice question
            has_choices = extract_choice(ground_truth)

            if has_numbers:
                # For numeric answers, use exact matching
                reward = numeric_reward(student_answer, ground_truth)
                if reward is None:
                    reward = ratio(clean_text(student_answer), clean_text(ground_truth))
            elif has_choices:
                # For multiple choice, extract and compare choices
                correct_choice = has_choices.upper()
                student_choice = extract_choice(student_answer)
                if student_choice:
                    reward = 1.0 if student_choice == correct_choice else 0.0
            else:
                # For text answers, use fuzzy matching
                reward = ratio(clean_text(student_answer), clean_text(ground_truth))
        except Exception:
            pass  # Keep reward as 0.0 if all methods fail

    return reward


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using symbolic verification, exact string matching, or fuzzy matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol, accu_reward_method in zip(
        contents, solution, kwargs.get("accu_reward_method")
    ):
        # if accu_reward_method is defined, use the corresponding reward function, otherwise use the default reward function
        if accu_reward_method in [
            "mcq",
            "yes_no",
            "llm",
            "map",
            "math",
            "weighted_sum",
            "od_ap",
            "od_ap50",
            "odLength",
            "all_match",
        ]:
            raise ValueError(
                f"Unsupported accuracy reward method: {accu_reward_method}"
            )
        else:
            reward = default_accuracy_reward(content, sol)
        rewards.append(reward)

        if DEBUG_MODE and _get_local_rank() == 0:
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
            problem = kwargs.get("vqa_question")[0]
            if reward <= 1.0:  # this condition can be changed for debug
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"------------- {current_time} Accuracy reward: {reward} -------------\n"
                    )
                    f.write(f"accu_reward_method: {accu_reward_method}\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")

    return rewards


def format_reward(completions, **kwargs):
    """Format reward with different levels based on task type:

    For procthor_active_vqa task:
    - 1.0: Has <think> tag, all three special tags <head>, <fwd>, <view> with valid integers,
           AND all values in valid ranges
           - head: (-180, 180]
           - fwd: >= 0
           - view: (-180, 180]
    - 0.75: Has <think> tag, all three tags present with integers, but some values out of range
    - 0.5: Has <think> tag, all three tags present, but failed to parse integers
    - 0.25: Has <think> tag and at least one special tag (partial tags)
    - 0.0: Missing <think> tag or no valid special tags
    """
    completion_contents = [completion[0]["content"] for completion in completions]
    task_type = kwargs.get("task_type", "default")
    prompts = kwargs.get("prompts", [])
    rewards = []
    debug_infos = []

    for idx, content in enumerate(completion_contents):
        stripped = content.strip()

        # Check if this is procthor_active_vqa task - check for special tags
        has_head = re.search(
            r"<head>\s*(-?\d+)\s*</head>", stripped, re.IGNORECASE
        )
        has_fwd = re.search(
            r"<fwd>\s*(-?\d+)\s*</fwd>", stripped, re.IGNORECASE
        )
        has_view = re.search(r"<view>\s*(-?\d+)\s*</view>", stripped, re.IGNORECASE
        )
        has_think = re.search(
            r"<think>.*?</think>", stripped, re.DOTALL | re.IGNORECASE
        )

        # If any special tags are present, treat as procthor_active_vqa format
        if has_head or has_fwd or has_view:
            if has_think and has_head and has_fwd and has_view:
                # Extract values and check ranges
                try:
                    head_val = int(has_head.group(1))
                    fwd_val = int(has_fwd.group(1))
                    view_val = int(has_view.group(1))

                    # Check if all values are in valid ranges
                    head_valid = -180 < head_val <= 180
                    fwd_valid = fwd_val >= 0
                    view_valid = -180 < view_val <= 180

                    # Calculate base reward
                    if head_valid and fwd_valid and view_valid:
                        base_reward = 1.0
                        reason = "all-valid"
                    else:
                        base_reward = 0.75
                        reason = "out-of-range"

                    rewards.append(base_reward)
                    debug_infos.append(
                        (
                            content,
                            f"head={head_val},fwd={fwd_val},view={view_val}",
                            base_reward,
                            reason,
                        )
                    )

                except (ValueError, AttributeError):
                    # Failed to parse integers
                    base_reward = 0.5
                    rewards.append(base_reward)
                    debug_infos.append(
                        (
                            content,
                            f"parse-error",
                            base_reward,
                            "procthor-partial",
                        )
                    )
            elif has_think and (has_head or has_fwd or has_view):
                base_reward = 0.25
                rewards.append(base_reward)
                debug_infos.append(
                    (
                        content,
                        f"partial-tags",
                        base_reward,
                        "procthor-partial",
                    )
                )
            else:
                rewards.append(0.0)
                debug_infos.append(
                    (
                        content,
                        f"invalid",
                        0.0,
                        "procthor-invalid",
                    )
                )
        else:
            rewards.append(0.0)
            debug_infos.append(
                (
                    content,
                    f"invalid",
                    0.0,
                    "procthor-invalid",
                )
            )

    if DEBUG_MODE and _get_local_rank() == 0:
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        log_dir, vis_dir = _get_debug_output_dir(**kwargs)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "debug_format.txt")
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"------------- {current_time} Format reward (task_type={task_type}) -------------\n"
                )
                for i, debug_info in enumerate(debug_infos):
                    original, ans, rew, reason = debug_info[:4]
                    f.write(
                        f"\n[Sample {i+1}] reward={rew} reason={reason} answer_raw={ans}\n"
                    )
                    if i < len(prompts):
                        prompt_str = str(prompts[i])[:200] if prompts else "N/A"
                        f.write(f"Prompt (first 200 chars): {prompt_str}...\n")
                    f.write(f"Completion (len={len(original)}): {original}\n")
                    f.write("-" * 80 + "\n")
        except Exception as e:
            pass

    return rewards


# ============================================================================
# Helper Functions for format_reward_refine
# ============================================================================



def _find_refined_tag(all_action_matches, initial_match, think_matches):
    """
    Find a refined action tag that appears after the initial guess and is preceded by a think tag.
    
    A refined tag must:
    1. Appear after the initial guess ends (to be in Phase B)
    2. Have a think tag that starts after the initial guess and ends before this action tag
    """
    if not initial_match:
        return None
    
    initial_pos = initial_match.end()
    
    for action_match in all_action_matches:
        action_pos = action_match.start()
        
        # Must appear after initial guess
        if action_pos <= initial_pos:
            continue
        
        # Check if there's a think tag that precedes this action in Phase B
        for think_match in think_matches:
            think_start = think_match.start()
            think_end = think_match.end()
            # Think tag must start after initial guess and end before this action
            if think_start >= initial_pos and think_end <= action_pos:
                return action_match
    
    return None


def _find_refined_tag_new_format(all_action_matches, initial_match, think_matches):
    """
    Find a refined action tag that appears after the initial guess and after the reasoning block.
    
    New format: single reasoning block followed by all final guesses.
    
    A refined tag must:
    1. Appear after the initial guess ends (to be in Phase B)
    2. Appear after the reasoning block ends (reasoning comes before all final guesses)
    """
    if not initial_match:
        return None
    
    initial_pos = initial_match.end()
    
    # Find the reasoning block (should be single in new format)
    reasoning_match = think_matches[0] if think_matches else None
    if not reasoning_match:
        return None
    
    reasoning_end = reasoning_match.end()
    
    for action_match in all_action_matches:
        action_pos = action_match.start()
        
        # Must appear after initial guess
        if action_pos <= initial_pos:
            continue
        
        # Must appear after reasoning block
        if action_pos >= reasoning_end:
            return action_match
    
    return None


def _validate_tag_counts(think_matches, head_matches, fwd_matches, view_matches):
    """Validate that tag counts match expected format (1 think, 2 of each action)."""
    counts = {
        "think": len(think_matches),
        "head": len(head_matches),
        "fwd": len(fwd_matches),
        "view": len(view_matches),
    }
    
    is_valid = (
        counts["think"] == 1 and
        counts["head"] == 2 and
        counts["fwd"] == 2 and
        counts["view"] == 2
    )
    
    return is_valid, counts


def _validate_tag_order(
    initial_head, initial_fwd, initial_view,
    refined_head, refined_fwd, refined_view,
    think_matches
):
    """Validate that tags appear in the correct order."""
    order_valid = True
    order_issues = []
    
    # Check initial tags order: head < fwd < view
    if initial_head and initial_fwd and initial_view:
        if not (initial_head.start() < initial_fwd.start() < initial_view.start()):
            order_valid = False
            order_issues.append("initial-order")
    
    # Check refined tags order: reasoning must come after initial guesses and before final guesses
    # Final guesses must be in order: head < fwd < view
    if refined_head and refined_fwd and refined_view:
        # Find the reasoning block (should be single)
        reasoning_match = think_matches[0] if think_matches else None
        
        # Check that reasoning comes after initial guesses
        if initial_view and reasoning_match:
            if reasoning_match.start() < initial_view.end():
                # Reasoning should start after all initial guesses
                order_valid = False
                order_issues.append("reasoning-before-initial")
        
        # Check that final guesses come after reasoning
        if reasoning_match:
            if not (reasoning_match.end() <= refined_head.start()):
                order_valid = False
                order_issues.append("final-before-reasoning")
        
        # Check final tags order: head < fwd < view
        if not (refined_head.start() < refined_fwd.start() < refined_view.start()):
            order_valid = False
            order_issues.append("refined-order")
    
    return order_valid, order_issues




def _format_tag_info(tag_counts, tag_count_valid, order_valid, order_issues):
    """Format tag count and order information for debug output."""
    tag_info = (
        f",tags=think:{tag_counts['think']}"
        f"/head:{tag_counts['head']}"
        f"/fwd:{tag_counts['fwd']}"
        f"/view:{tag_counts['view']}"
    )
    
    if not tag_count_valid:
        tag_info += "(count-invalid)"
    if not order_valid:
        tag_info += f"(order-invalid:{','.join(order_issues)})"
    
    return tag_info


def _calculate_reward_new_rules(
    refined_head, refined_fwd, refined_view,
    has_all_actions, has_think,
    tag_count_valid, order_valid, order_issues,
    tag_counts
):
    """
    Calculate reward based on new rules:
    1. When all tags including think tag are present:
       - if tag order is wrong but tag count is correct: 0.75
       - if only tag count is wrong: 0.5
    2. If only think tag is missing but all action tags are present: 0.25
    3. Otherwise if parsing fails or any action tag is missing: 0
    """
    # Try to parse values
    try:
        head_val = int(refined_head.group(1)) if refined_head else None
        fwd_val = int(refined_fwd.group(1)) if refined_fwd else None
        view_val = int(refined_view.group(1)) if refined_view else None
        parse_success = head_val is not None and fwd_val is not None and view_val is not None
    except (ValueError, AttributeError):
        parse_success = False
        head_val = fwd_val = view_val = None
    
    # Rule 3: if parsing fails or any action tag is missing, return 0
    if not parse_success or not has_all_actions:
        tag_info = _format_tag_info(tag_counts, tag_count_valid, order_valid, order_issues)
        
        if not parse_success:
            reason = "parse-error"
        else:
            reason = "missing-action-tags"
        
        if not tag_count_valid:
            reason += "+tag-count-invalid"
        if not order_valid:
            reason += "+tag-order-invalid"
        
        return {
            "reward": 0.0,
            "answer": f"{reason}{tag_info}",
            "reason": reason,
        }
    
    # Rule 2: if only think tag is missing but all action tags are present, return 0.25
    if not has_think and has_all_actions:
        tag_info = _format_tag_info(tag_counts, tag_count_valid, order_valid, order_issues)
        
        reason = "no-think-all-actions"
        if not tag_count_valid:
            reason += "+tag-count-invalid"
        if not order_valid:
            reason += "+tag-order-invalid"
        
        return {
            "reward": 0.25,
            "answer": f"head={head_val},fwd={fwd_val},view={view_val}{tag_info}",
            "reason": reason,
        }
    
    # Rule 1: when all tags including think tag are present
    if has_think and has_all_actions:
        # if tag order is wrong but tag count is correct: 0.75
        if not order_valid and tag_count_valid:
            base_reward = 0.75
            reason = "order-invalid-count-valid"
        # if only tag count is wrong: 0.5
        elif not tag_count_valid:
            base_reward = 0.5
            reason = "tag-count-invalid"
        # if order and count are correct: 1.0 (perfect case)
        else:
            base_reward = 1.0
            reason = "all-valid"
        
        tag_info = _format_tag_info(tag_counts, tag_count_valid, order_valid, order_issues)
        
        return {
            "reward": base_reward,
            "answer": f"head={head_val},fwd={fwd_val},view={view_val}{tag_info}",
            "reason": reason,
        }
    
    # Fallback: should not reach here, but return 0.0
    tag_info = _format_tag_info(tag_counts, tag_count_valid, order_valid, order_issues)
    
    return {
        "reward": 0.0,
        "answer": f"unexpected-case{tag_info}",
        "reason": "unexpected-case",
    }


def _calculate_incomplete_reward(
    complete_pairs,
    tag_count_valid, order_valid, order_issues,
    tag_counts
):
    """
    Calculate reward when fewer than 3 action pairs are complete.
    
    Always returns 0.0 reward (no partial credit).
    
    Args:
        complete_pairs: Number of complete action pairs (0, 1, or 2)
        Other args: Tag validation information for debug output
    
    Returns:
        Dictionary with reward=0.0, answer string, and reason
    """
    tag_info = _format_tag_info(tag_counts, tag_count_valid, order_valid, order_issues)
    
    # Determine reason based on number of complete pairs
    if complete_pairs == 0:
        reason = "twostage-no-pairs"
        answer_prefix = "no-pairs"
    elif complete_pairs == 1:
        reason = "twostage-one-pair"
        answer_prefix = "one-pair"
    else:  # complete_pairs == 2
        reason = "twostage-two-pairs"
        answer_prefix = "two-pairs"
    
    # Add additional validation issues to reason
    if not tag_count_valid:
        reason += "+tag-count-invalid"
    if not order_valid:
        reason += "+tag-order-invalid"
    
    return {
        "reward": 0.0,
        "answer": f"{answer_prefix}{tag_info}",
        "reason": reason,
    }


def format_reward_refine(completions, **kwargs):
    """
    Format reward for two-stage sequential reasoning format.
    
    Expected Format (based on SFT_GRPO_FORMAT_PROMPT):
    ---------------
    Phase A - Initial guesses (quick predictions without reasoning):
        <head>INITIAL_GUESS</head> <fwd>INITIAL_GUESS</fwd> <view>INITIAL_GUESS</view>
    
    Phase B - Refined predictions (with single reasoning block):
        <think>REASONING PROCESS</think>
        <head>FINAL_GUESS</head> <fwd>FINAL_GUESS</fwd> <view>FINAL_GUESS</view>
    
    Validation Rules:
    ----------------
    1. Tag Count Requirements:
       - Exactly 1 <think> tag (single reasoning block for all actions)
       - Exactly 2 of each action tag: <head>, <fwd>, <view>
         (1 initial + 1 refined for each)
    
    2. Tag Order Requirements:
       - Initial phase: head < fwd < view (by position)
       - Reasoning must appear after initial guesses and before final guesses
       - Final phase: head < fwd < view (by position, after reasoning)
    
    3. Value Range Requirements:
       - view: (-180, 180] degrees
    
    Reward Structure:
    ----------------
    Perfect completion (all 3 action pairs present):
        - 1.0: All conditions met (correct tags, order, counts, valid ranges)
        - 0.75: All tags correct but values out of range
        - 0.5: All tags correct but cannot parse integer values OR tag count/order violations with valid values
        - 0.25: Cannot parse integers + tag count/order violations
        - 0.0: Parse error
    
    Incomplete (< 3 action pairs):
        - 0.0: Any case where all 3 action pairs are not complete (no partial credit)
    
    Args:
        completions: List of model completions
        **kwargs: Additional arguments including task_type, use_confidence_score, prompts
    
    Returns:
        List of reward scores (float) for each completion
    """
    completion_contents = [completion[0]["content"] for completion in completions]
    task_type = kwargs.get("task_type", "default")
    prompts = kwargs.get("prompts", [])
    
    rewards = []
    debug_infos = []

    for idx, content in enumerate(completion_contents):
        stripped = content.strip()

        # === 2. Find initial action tags (Phase A) ===
        initial_head_match = re.search(
            r"<head>\s*(-?\d+)\s*</head>", stripped, re.IGNORECASE
        )
        initial_fwd_match = re.search(
            r"<fwd>\s*(-?\d+)\s*</fwd>", stripped, re.IGNORECASE
        )
        initial_view_match = re.search(
            r"<view>\s*(-?\d+)\s*</view>", stripped, re.IGNORECASE
        )

        # === 3. Find thinking tags and all action tags (Phase B) ===
        think_matches = list(re.finditer(
            r"<think>.*?</think>", stripped, re.DOTALL | re.IGNORECASE
        ))
        all_head_matches = list(re.finditer(
            r"<head>\s*(-?\d+)\s*</head>", stripped, re.IGNORECASE
        ))
        all_fwd_matches = list(re.finditer(
            r"<fwd>\s*(-?\d+)\s*</fwd>", stripped, re.IGNORECASE
        ))
        all_view_matches = list(re.finditer(
            r"<view>\s*(-?\d+)\s*</view>", stripped, re.IGNORECASE
        ))

        # === 4. Find refined action tags (must appear after initial guesses and after reasoning) ===
        # New format: single reasoning block followed by all final guesses
        refined_head_match = _find_refined_tag_new_format(
            all_head_matches, initial_head_match, think_matches
        )
        refined_fwd_match = _find_refined_tag_new_format(
            all_fwd_matches, initial_fwd_match, think_matches
        )
        refined_view_match = _find_refined_tag_new_format(
            all_view_matches, initial_view_match, think_matches
        )

        # === 5. Validate tag counts and order ===
        tag_count_valid, tag_counts = _validate_tag_counts(
            think_matches, all_head_matches, all_fwd_matches, all_view_matches
        )
        order_valid, order_issues = _validate_tag_order(
            initial_head_match, initial_fwd_match, initial_view_match,
            refined_head_match, refined_fwd_match, refined_view_match,
            think_matches
        )

        # === 6. Check if all action tags are present ===
        has_all_initial = bool(initial_head_match and initial_fwd_match and initial_view_match)
        has_all_refined = bool(refined_head_match and refined_fwd_match and refined_view_match)
        has_all_actions = has_all_initial and has_all_refined
        
        # Check if think tag is present
        has_think = len(think_matches) > 0

        # === 7. Calculate reward based on new rules ===
        reward_info = _calculate_reward_new_rules(
            refined_head_match, refined_fwd_match, refined_view_match,
            has_all_actions, has_think,
            tag_count_valid, order_valid, order_issues,
            tag_counts
        )

        # === 8. Store results ===
        rewards.append(reward_info["reward"])
        debug_infos.append((
            content,
            reward_info["answer"],
            reward_info["reward"],
            reward_info["reason"],
        ))

    if DEBUG_MODE and _get_local_rank() == 0:
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        log_dir, vis_dir = _get_debug_output_dir(**kwargs)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "debug_format_twostage.txt")
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"------------- {current_time} Format reward twostage sequential (task_type={task_type}) -------------\n"
                )
                for i, debug_info in enumerate(debug_infos):
                    original, ans, rew, reason = debug_info[:4]
                    f.write(
                        f"\n[Sample {i+1}] reward={rew} reason={reason} answer_raw={ans}\n"
                    )
                    if i < len(prompts):
                        prompt_str = str(prompts[i])[:200] if prompts else "N/A"
                        f.write(f"Prompt (first 200 chars): {prompt_str}...\n")
                    f.write(f"Completion (len={len(original)}): {original}\n")
                    f.write("-" * 80 + "\n")
        except Exception as e:
            pass


    return rewards


def numeric_reward(content, sol, **kwargs):
    content = clean_text(content)
    sol = clean_text(sol)
    try:
        content, sol = float(content), float(sol)
        return 1.0 if content == sol else 0.0
    except:
        return None


def action_accuracy_reward(completions, solution, **kwargs):
    """The student output predicts the action parameters (heading rotation, forward distance in cm, view rotation) and uses them to create an additional view,
    then puts the view into the verifier Qwen2.5-VL-3B model to re-reason the answer and compare it with the GT to give an accuracy score.

    Args:
        completions: the student model's output (heading rotation, forward distance in centimeters, final view rotation)
        solution: ground truth answer

    Reward = default_accuracy_reward(verifier_answer, ground_truth)
    - By default, only the additional view is sent to the verifier (not the original input view)
    - If the verifier or image loading fails, return 0.0 (conservative)
    - If the additional view creation fails, fallback to using the primary input view
    """
    # Mini-batch size for verifier calls (only the verifier stage is batched)
    VERIFIER_MINIBATCH = int(os.getenv("VERIFIER_MINIBATCH", kwargs.get("verifier_minibatch", 16)))

    rewards = []
    contents = [c[0]["content"] for c in completions]
    # image_path: list[list[str]] or list[str]
    image_paths_arg = kwargs.get("image_path")
    problems_arg = kwargs.get("vqa_question")
    controller = kwargs.get("controller")

    # Fallback penalty: accept float or list[float]
    _fp = kwargs.get("fallback_penalty", 1.0)
    try:
        fallback_penalty = (_fp[-1] if isinstance(_fp, list) and _fp else float(_fp))
    except Exception:
        fallback_penalty = 1.0

    render_metadata = kwargs.get("render_metadata")

    # First pass: render additional views per-sample and collect verifier jobs
    per_item = []  # holds dicts per sample
    pending_jobs = []  # (global_idx, selected_images[0], question, question_type)
    for i, (content, sol) in enumerate(zip(contents, solution)):
        # Get original image path
        if isinstance(image_paths_arg, list) and len(image_paths_arg) == len(contents):
            orig_imgs = image_paths_arg[i]
        else:
            # Use first element if batch structure is unclear
            orig_imgs = image_paths_arg[0] if image_paths_arg else []
        if isinstance(orig_imgs, str):
            orig_imgs = [orig_imgs]
        if not orig_imgs:
            rewards.append(0.0)
            continue
        primary_img = PIL.Image.open(orig_imgs[0])

        # Extract predicted action parameters (head, fwd, view)
        # if True:
        #    print(f"Content: {content}")
        # actions = _extract_final_answer_action(content)
        thinking, actions = extract_thinking_and_action_string(content)
        # Extract question from problem text
        if isinstance(problems_arg, list) and len(problems_arg) == len(contents):
            raw_problem = problems_arg[i]
        else:
            raw_problem = problems_arg[0] if problems_arg else ""
        # vqa_question already contains the plain question text
        question = raw_problem.strip()

        # Extract render_metadata for this sample
        render_metadata_item = None
        if render_metadata is not None:
            if isinstance(render_metadata, list) and len(render_metadata) == len(
                contents
            ):
                render_metadata_item = render_metadata[i]
            else:
                render_metadata_item = (
                    render_metadata[0]
                    if isinstance(render_metadata, list)
                    else render_metadata
                )

        # Get question_type from kwargs or render_metadata
        question_type = None
        if isinstance(kwargs.get("question_type"), list) and len(kwargs.get("question_type")) == len(contents):
            question_type = kwargs.get("question_type")[i]
        elif kwargs.get("question_type") is not None:
            question_type = kwargs.get("question_type")[0] if isinstance(kwargs.get("question_type"), list) else kwargs.get("question_type")
        elif render_metadata_item is not None and render_metadata_item.get("question_type") is not None:
            question_type = render_metadata_item.get("question_type")

        # Ensure we have a controller per rank (GPU)
        if controller is None:
            controller = get_procthor_controller(headless=True)

        # Build additional view using predicted heading, forward distance, and view rotation

        add_view, result_metadata = build_additional_view(
            controller,
            action_string_to_list(actions),
            render_metadata=render_metadata_item,
        )

        # Extract metadata
        if result_metadata is None:
            result_metadata = {}
        verifier_render_ans = result_metadata.get("answer", None)
        used_fallback = result_metadata.get("used_fallback", False)


        # Use only the additional view; fallback to primary_img if creation fails
        # selected_images = [primary_img] + ([add_view] if add_view is not None else [])
        selected_images = [add_view] if add_view is not None else [primary_img]

        item = {
            "selected_images": selected_images,
            "question": question,
            "question_type": question_type,
            "solution": sol,
            "verifier_ans": None,
            "used_fallback": used_fallback,
            "actions": actions,  # Store for debug logging
            "content": content,  # Store for debug logging
            "add_view": add_view,  # Store for debug logging
        }
        if item["verifier_ans"] is None:
            # push single-image job (selected_images[0], question, question_type) for verifier batching
            pending_jobs.append((i, selected_images[0], question, question_type))
        per_item.append(item)

    # Second pass: run verifier in mini-batches for pending jobs
    model, processor = initialize_verifier()
    def _run_single(img, q, qt=None):
        try:
            pil = img if isinstance(img, PIL.Image.Image) else Image.open(img).convert("RGB")
            # Add question type specific prompt
            q_with_prompt = _add_question_type_prompt(q, qt)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": pil},
                    {"type": "text", "text": q_with_prompt},
                ],
            }]
            chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[chat_text], images=[pil], return_tensors="pt").to(model.device)
            gen = model.generate(**inputs, max_new_tokens=32, do_sample=False)
            out = processor.batch_decode(gen, skip_special_tokens=True)[0]
            ans_blocks = re.findall(r"<answer>(.*?)</answer>", out, re.DOTALL | re.IGNORECASE)
            ans = (ans_blocks[-1].strip() if ans_blocks else out.strip().split("\n")[-1].strip())
            return ans.rstrip(".").strip()
        except Exception:
            return None

    if model is not None and processor is not None and pending_jobs:
        for s in range(0, len(pending_jobs), max(1, VERIFIER_MINIBATCH)):
            chunk = pending_jobs[s:s + max(1, VERIFIER_MINIBATCH)]
            for (idx, img0, q0, qt0) in chunk:
                per_item[idx]["verifier_ans"] = _run_single(img0, q0, qt0)
    else:
        # Fallback: mark unanswered
        for (idx, _, _, _) in pending_jobs:
            per_item[idx]["verifier_ans"] = None

    # Final pass: compute rewards and logs
    for i, item in enumerate(per_item):
        verifier_ans = item["verifier_ans"]
        sol = item["solution"]
        used_fallback = item["used_fallback"]
        question = item["question"]
        primary_or_add = item["selected_images"][0]
        # Get stored values for debug logging
        actions = item.get("actions", "")
        content = item.get("content", "")
        add_view = item.get("add_view", None)
        selected_images = item.get("selected_images", [])

        if verifier_ans is None:
            reward = 0.0
        else:
            reward = default_accuracy_reward(verifier_ans, sol)
            if used_fallback:
                reward = reward * fallback_penalty
                if DEBUG_MODE and _get_local_rank() == 0:
                    print(
                        f"[action_accuracy] Applied fallback penalty: {fallback_penalty}, final reward: {reward:.3f}"
                    )

        rewards.append(reward)
        try:
            sol_print = (
                re.search(r"<answer>(.*?)</answer>", sol, re.DOTALL | re.IGNORECASE)
                .group(1)
                .strip()
            )
            if DEBUG_MODE and _get_local_rank() == 0:
                print(
                    f"[action_accuracy] predicted_actions={actions} question='{question[:80]}' verifier_ans='{verifier_ans}' gt='{sol_print[:80]}' reward={reward:.3f}"
                )
        except Exception:
            pass
        if DEBUG_MODE and _get_local_rank() == 0:
            log_dir, vis_dir = _get_debug_output_dir(**kwargs)
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(vis_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "debug_action_accuracy.txt")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"----- {current_time} action_accuracy reward: {reward} -----\n"
                    )
                    f.write(f"actions: {actions}, add_view: {add_view}\n")
                    f.write(f"selected_images: {selected_images}\n")
                    f.write(f"question: {question}\n")
                    f.write(f"verifier_ans: {verifier_ans}\n")
                    f.write(f"ground_truth: {sol}\n")
                    f.write(f"student_raw: {content}\n")
            except Exception as e:
                pass
            
            # Save visualization for first item in batch only
            if i == 0 and add_view is not None:
                try:
                    # Get primary image from first iteration
                    primary_img_for_vis = None
                    if isinstance(image_paths_arg, list) and len(image_paths_arg) > 0:
                        orig_imgs_vis = image_paths_arg[0] if isinstance(image_paths_arg[0], list) else [image_paths_arg[0]]
                        if orig_imgs_vis:
                            primary_img_for_vis = PIL.Image.open(orig_imgs_vis[0]).convert("RGB")
                    
                    if primary_img_for_vis is not None:
                        # Extract thinking process from content
                        thinking_process = content
                        
                        # Extract action question from vqa_question if available
                        action_question = question
                        vqa_question = question
                        
                        # Parse actions
                        predicted_actions = action_string_to_list(actions)
                        
                        # Extract GT answer text
                        gt_answer_text = ""
                        try:
                            gt_answer_text = re.search(r"<answer>(.*?)</answer>", sol, re.DOTALL | re.IGNORECASE).group(1).strip()
                        except:
                            gt_answer_text = sol
                        
                        # Determine if correct
                        is_correct = reward > 0.5  # Consider reward > 0.5 as correct
                        
                        # Generate filename with timestamp
                        vis_filename = f"action_accuracy_{current_time}_{i:05d}.png"
                        vis_path = os.path.join(vis_dir, vis_filename)
                        
                        create_visualization(
                            input_img_or_path=primary_img_for_vis,
                            generated_img_or_path=add_view,
                            predicted_actions=predicted_actions,
                            gt_actions=[],  # Not displayed
                            action_question=action_question,
                            vqa_question=vqa_question,
                            thinking_process=thinking_process,
                            verifier_answer=verifier_ans or "N/A",
                            gt_answer=gt_answer_text,
                            is_correct=is_correct,
                            output_path=vis_path,
                            gt_img_or_path=None,
                        )
                except Exception as e:
                    if DEBUG_MODE and _get_local_rank() == 0:
                        print(f"[DEBUG] Failed to save visualization: {e}")
    return rewards


