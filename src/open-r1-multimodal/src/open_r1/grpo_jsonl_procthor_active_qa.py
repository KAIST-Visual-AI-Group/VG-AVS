# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import pathlib
import re
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset
from open_r1.qwen2_5vl_monkey_patch import (
    monkey_patch_qwen2_5vl_flash_attn,
    monkey_patch_torch_load,
)
from open_r1.trainer import GRPOConfig, VLMGRPOTrainer
from open_r1.utils.model_load import get_vlm_module
from open_r1.utils.prompt_templates import ACTION_PROMPT_TEMPLATE, GRPO_FORMAT_PROMPT, SFT_GRPO_FORMAT_PROMPT
from open_r1.utils.rewards import (
    accuracy_reward,
    action_accuracy_reward,
    format_reward,
    format_reward_refine,
)

from transformers.utils import logging
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

monkey_patch_qwen2_5vl_flash_attn()
monkey_patch_torch_load()

logger = logging.get_logger(__name__)

DEBUG_MODE = str(os.getenv("DEBUG_MODE", "0")) == "1"


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """

    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    val_split_seed: Optional[int] = field(
        default=42,
        metadata={
            "help": "Random seed used for train/validation split (datasets.train_test_split)."
        },
    )
    save_validation_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "If set, save validation split to this path. Use a directory for Arrow (save_to_disk) or a file ending with .jsonl/.json for JSON."
        },
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format'"
        },
    )
    grpo_reward_weights: list[float] = field(
        default_factory=lambda: [0.3, 1],
        metadata={
            "help": "Weights for each reward function (same order as reward_funcs)"
        },
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={
            "help": "Maximum number of anyres blocks for the image (for InternVL)"
        },
    )
    use_fallback: bool = field(
        default=True,
        metadata={
            "help": "Use fallback position when predicted action is infeasible (default: True)"
        },
    )
    fallback_penalty: float = field(
        default=1.0,
        metadata={
            "help": "Penalty multiplier for using fallback position (default: 1.0)"
        },
    )
    fallback_num_grids: int = field(
        default=8,
        metadata={
            "help": "Number of grid points to check on the line from current to target for fallback (default: 8)"
        },
    )
    customized_scene_path: str = field(
        default=None,
        metadata={"help": "Path to customized scene (default: None)"},
    )
    vlm_lr: Optional[float] = field(
        default=1e-6,
        metadata={"help": "Learning rate for VLM backbone. If None, uses training_args.learning_rate"},
    )
    use_refine: bool = field(
        default=False,
        metadata={"help": "Whether to use refine format (initial guesses, then refined predictions with reasoning) (default: False)"},
    )


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "action_accuracy": action_accuracy_reward,
}


@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


def main(script_args, training_args, model_args):
    # Get local_rank for debug output control
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    if DEBUG_MODE and local_rank == 0:
        print("using vlm module:", vlm_module_cls.__name__)


    # overwrite the reward functions for two-stage sequential task
    if script_args.use_refine:
        reward_funcs_registry["format"] = format_reward_refine

    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    reward_weights = script_args.grpo_reward_weights
    if DEBUG_MODE and local_rank == 0:
        print("reward_funcs:", reward_funcs)
        print("reward_weights:", reward_weights)

    assert len(reward_funcs) == len(
        reward_weights
    ), "Length of reward_funcs and reward_weights must match"

    # ====== DATA PREPROCESSING ======
    # Load the JSONL datasets
    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")

    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")

    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")

    # Helper to convert raw jsonl item into conversation/example dict
    def make_conversation_from_jsonl(example, image_folder: str):
        required_fields = [
            "question",
            "answer",
            "scene_id",
            "question_image",
            "question_position",
            "question_rotation",
            "gt_action",
            "gt_position",
            "gt_rotation",
        ]
        for field in required_fields:
            if field not in example:
                raise ValueError(f"Field {field} is required in the example")

        # Resolve image paths from 'question_image'
        if isinstance(example.get("question_image"), str):
            img_path = (
                example["question_image"]
                if os.path.isabs(example["question_image"])
                else os.path.join(image_folder, example["question_image"])
            )
            image_paths = [img_path]
        elif isinstance(example.get("question_image"), list):
            image_paths = [
                (img if os.path.isabs(img) else os.path.join(image_folder, img))
                for img in example["question_image"]
            ]
        else:
            raise ValueError(
                f"Unsupported image type: {type(example.get('question_image'))}"
            )

        # Build action question (instruction + format guidance)
        # Select template based on use_refine flags
        if script_args.use_refine:
            action_question = (
                ACTION_PROMPT_TEMPLATE.format(question=example["question"])
                + SFT_GRPO_FORMAT_PROMPT 
            )
        else:
            action_question = (
                ACTION_PROMPT_TEMPLATE.format(question=example["question"])
                + GRPO_FORMAT_PROMPT
            )

        # Input camera pose from question_* fields
        position_list = example["question_position"]
        rotation_scalar = example["question_rotation"]
        render_position = {
            "x": float(position_list[0]),
            "y": float(position_list[1]),
            "z": float(position_list[2]),
        }
        render_rotation = {"x": 0.0, "y": float(rotation_scalar), "z": 0.0}

        # Ground-truth camera pose
        gt_position_list = example["gt_position"]
        gt_rotation_scalar = example["gt_rotation"]
        gt_position = {
            "x": float(gt_position_list[0]),
            "y": float(gt_position_list[1]),
            "z": float(gt_position_list[2]),
        }
        gt_rotation = {"x": 0.0, "y": float(gt_rotation_scalar), "z": 0.0}

        # Build render metadata (includes scene, fallbacks, thresholds)
        render_metadata = {
            "position": render_position,
            "rotation": render_rotation,
            "gt_position": gt_position,
            "gt_rotation": gt_rotation,
            "scene_index": int(re.search(r"house_(\d+)", example["scene_id"]).group(1)),
            "trans_scale": 100.0,
            "check_obj_existence": False,
            "pixel_threshold": 200,
            "use_fallback": script_args.use_fallback,
            "num_grids": script_args.fallback_num_grids,
        }

        # Include question_type in render_metadata if available
        if "question_type" in example:
            render_metadata["question_type"] = example["question_type"]

        if script_args.customized_scene_path is not None:
            custom_house_path = {
                "train": script_args.customized_scene_path,
                "val": None,
                "test": None,
            }
        else:
            custom_house_path = None
        render_metadata["custom_house_path"] = custom_house_path

        assert all(
            os.path.exists(p) for p in image_paths
        ), f"Image paths do not exist: {image_paths}"

        # Extract question_type if available
        question_type = example.get("question_type", None)

        return {
            "image_path": image_paths,
            "action_question": action_question,
            "vqa_question": example["question"],
            "gt_action": example["gt_action"],
            "render_metadata": render_metadata,
            "solution": f"<answer> {example['answer']} </answer>",
            "use_confidence_score": False,
            "question_type": question_type,
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        *(
                            {"type": "image", "text": None}
                            for _ in range(len(image_paths))
                        ),
                        {"type": "text", "text": action_question},
                    ],
                }
            ],
        }

    all_data = []
    for data_file, image_folder in zip(data_files, image_folders):
        with open(data_file, "r") as f:
            for line in f:
                item = json.loads(line)
                transformed = make_conversation_from_jsonl(item, image_folder)
                all_data.append(transformed)

    dataset = Dataset.from_list(all_data)

    # Split dataset for validation if requested
    splits = {"train": dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(
            test_size=script_args.val_split_ratio, seed=script_args.val_split_seed
        )
        splits["train"] = train_val_split["train"]
        splits["validation"] = train_val_split["test"]

        # Optionally persist validation split
        if script_args.save_validation_path:
            val_ds = splits.get("validation")
            save_path = script_args.save_validation_path
            try:
                if save_path.lower().endswith(".jsonl") or save_path.lower().endswith(
                    ".json"
                ):
                    # Prefer Dataset.to_json; fallback to manual JSONL
                    try:
                        # lines=True ensures JSON Lines when supported
                        val_ds.to_json(save_path, lines=True, force_ascii=False)
                    except Exception:
                        import json as _json

                        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                        with open(save_path, "w", encoding="utf-8") as _f:
                            for row in val_ds:
                                _f.write(_json.dumps(row, ensure_ascii=False) + "\n")
                    if DEBUG_MODE and local_rank == 0:
                        print(f"Saved validation split to JSON at: {save_path}")
                else:
                    # Treat as Arrow dataset directory
                    os.makedirs(save_path, exist_ok=True)
                    val_ds.save_to_disk(save_path)
                    if DEBUG_MODE and local_rank == 0:
                        print(f"Saved validation split (Arrow) to: {save_path}")
            except Exception as e:
                if DEBUG_MODE and local_rank == 0:
                    print(f"[warn] Failed to save validation split to {save_path}: {e}")
    elif script_args.save_validation_path:
        print(
            "[warn] save_validation_path is set but val_split_ratio == 0; no validation split to save."
        )


    else:
        trainer_cls = VLMGRPOTrainer
    if DEBUG_MODE and local_rank == 0:
        print("using trainer:", trainer_cls.__name__)
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        reward_weights=reward_weights,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=splits["train"],
        eval_dataset=(
            splits.get("validation") if training_args.eval_strategy != "no" else None
        ),
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
    )



    # Train and push the model to the Hub
    checkpoint_dirs = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if checkpoint_dirs:
        # Sort by checkpoint number (extract number from directory name)
        def get_checkpoint_number(path):
            try:
                return int(path.name.split("-")[1])
            except (IndexError, ValueError):
                return 0

        latest_checkpoint = max(checkpoint_dirs, key=get_checkpoint_number)
        print(f"Resuming from checkpoint from {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("Training from scratch")
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
