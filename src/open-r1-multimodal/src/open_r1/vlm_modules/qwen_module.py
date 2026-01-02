from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
from copy import deepcopy
from open_r1.vlm_modules.vlm_module import VLMBaseModule
from PIL import Image
import os

class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        import os
        import json
        
        # Check if it's a checkpoint directory
        if os.path.isdir(model_id):
            config_path = os.path.join(model_id, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # Check model_type or architectures in config
                    model_type = config.get("model_type", "").lower()
                    architectures = config.get("architectures", [])
                    arch_str = " ".join(architectures).lower()
                    
                    if "qwen2-vl" in model_type or "qwen2vl" in arch_str:
                        model_cls = Qwen2VLForConditionalGeneration
                    elif "qwen2.5-vl" in model_type or "qwen2_5_vl" in arch_str:
                        model_cls = Qwen2_5_VLForConditionalGeneration
                    else:
                        raise ValueError(f"Unsupported model type in config: {model_type}, architectures: {architectures}")
            else:
                # Check for adapter_config.json (LoRA adapter)
                adapter_config_path = os.path.join(model_id, "adapter_config.json")
                if os.path.exists(adapter_config_path):
                    with open(adapter_config_path, 'r') as f:
                        adapter_config = json.load(f)
                        # Get base model name from adapter config
                        base_model = adapter_config.get("base_model_name_or_path", "")
                        if base_model:
                            # Recursively check the base model
                            return self.get_model_class(base_model, model_init_kwargs)
                        else:
                            raise ValueError(f"No base_model_name_or_path found in adapter_config.json: {adapter_config_path}")
                else:
                    raise ValueError(f"Neither config.json nor adapter_config.json found in checkpoint: {model_id}")
        else:
            # Fallback to checking the model_id string
            if "Qwen2-VL" in model_id:
                model_cls = Qwen2VLForConditionalGeneration
            elif "Qwen2.5-VL" in model_id:
                model_cls = Qwen2_5_VLForConditionalGeneration
            elif "Qwen2-VL" in model_id or "qwen2-vl" in model_id.lower():
                model_cls = Qwen2VLForConditionalGeneration
            else:
                raise ValueError(f"Unsupported model: {model_id}")
        
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        # Expose resize hints and pixel caps to the trainer so it can set them on the processor
        return [
            ('image_processor', 'max_pixels'),
            ('image_processor', 'min_pixels'),
            ('image_processor', 'resize_width'),
            ('image_processor', 'resize_height'),
        ]
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        additional_output = None
        # Optional resizing prior to processor: use processor hints first, then env vars
        def _fixed_size_from_hints():
            rw = rh = None
            ip = getattr(processing_class, 'image_processor', None)
            if ip is not None:
                rw = getattr(ip, 'resize_width', None)
                rh = getattr(ip, 'resize_height', None)
            if rw is None:
                rw = getattr(processing_class, 'resize_width', None)
            if rh is None:
                rh = getattr(processing_class, 'resize_height', None)
            # Env fallback
            if rw is None:
                try:
                    rw = int(os.getenv('RESIZE_WIDTH', '0')) or None
                except Exception:
                    rw = None
            if rh is None:
                try:
                    rh = int(os.getenv('RESIZE_HEIGHT', '0')) or None
                except Exception:
                    rh = None
            return rw, rh

        def _maybe_resize_one(img: Image.Image, size_hint: tuple[int|None, int|None]):
            if not isinstance(img, Image.Image):
                return img
            try:
                w, h = img.size
                rw, rh = size_hint
                if rw and rh and (w != rw or h != rh):
                    return img.resize((int(rw), int(rh)), Image.Resampling.LANCZOS)
            except Exception:
                return img
            return img

        def _resize_structure(imgs):
            rw, rh = _fixed_size_from_hints()
            if rw is None or rh is None:
                return imgs
            # Preserve nested structure (list[list[Image]] or list[Image])
            if isinstance(imgs, list) and imgs and all(isinstance(x, list) for x in imgs):
                return [[_maybe_resize_one(im, (rw, rh)) for im in inner] for inner in imgs]
            elif isinstance(imgs, list):
                return [_maybe_resize_one(im, (rw, rh)) for im in imgs]
            return imgs

        if len(images) > 0:
            images = _resize_structure(images)
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
            additional_output = [{'image_grid_thw': image_grid_thw} for image_grid_thw in prompt_inputs['image_grid_thw']]    
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs, additional_output
    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case _:
                raise ValueError(f"Unsupported task type: {task_type}")
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. The text between <answer> and </answer> must be exactly one uppercase letter, A or B. No spaces, words, punctuation, or additional characters are allowed. If uncertain, choose the best single letter. Do not output anything after </answer>."
            
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the Qwen model output matches a specific format."""
        import re
        import os
        from datetime import datetime
        pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]

        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")
        return [1.0 if match else 0.0 for match in matches]
    
    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
        import re
        import os
        from datetime import datetime
        import json
        def iou(box1, box2):
            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2]-1, box2[2]-1)
            inter_y2 = min(box1[3]-1, box2[3]-1)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
            else:
                inter = 0
            union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
            return float(inter)/union
        def resize_bbox(bbox, input_height, input_width, image_height, image_width):
            bbox[0] = bbox[0] / input_width * image_width
            bbox[1] = bbox[1] / input_height * image_height
            bbox[2] = bbox[2] / input_width * image_width
            bbox[3] = bbox[3] / input_height * image_height
            return bbox
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'

        for i, (content, sol) in enumerate(zip(contents, solution)):
            image_grid_thw = kwargs.get("image_grid_thw")[i]
            image_path = kwargs.get("image_path")[i][0]
            image = Image.open(image_path)
            image_width, image_height = image.size
            input_height = int(image_grid_thw[1]*14)
            input_width = int(image_grid_thw[2]*14)
            
            sol = re.findall(answer_tag_pattern, sol, re.DOTALL)[-1]
            sol = json.loads(sol.strip())
            reward = 0.0
            # Try symbolic verification first
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                        bbox = resize_bbox(bbox, input_height, input_width, image_height, image_width)
                        # if iou(bbox, sol) > 0.5:
                        #     reward = 1.0
                        reward = iou(bbox, sol)
            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[i] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[i]
                if reward <= 1.0:  # this condition can be changed for debug
                    with open(log_path, "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"problem: {problem}\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Solution: {sol}\n") 
        return rewards

    @staticmethod
    def select_reward_func(func: str, task_type: str):
        if func == "accuracy":
            match task_type:
                case "rec":
                    return Qwen2VLModule.iou_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "format":
            match task_type:
                case "rec":
                    return Qwen2VLModule.format_reward_rec
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        else:
            raise ValueError(f"Unsupported reward function: {func}")
