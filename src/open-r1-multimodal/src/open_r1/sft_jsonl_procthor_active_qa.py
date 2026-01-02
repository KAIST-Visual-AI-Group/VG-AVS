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

"""
SFT trainer for ProcTHOR Active VQA task.
This script trains a VLM to predict action parameters (head, fwd, view)
based on gt_action field using standard language modeling loss.
"""

import json
import os
import pathlib
import re
from dataclasses import dataclass, field
from typing import Optional

import PIL
import torch
import numpy as np
from datasets import Dataset
from open_r1.qwen2_5vl_monkey_patch import (
    monkey_patch_qwen2_5vl_flash_attn,
    monkey_patch_qwen2_5vl_forward,
    monkey_patch_torch_load,
)
from open_r1.utils.model_load import get_vlm_module
from open_r1.utils.prompt_templates import (
    ACTION_PROMPT_TEMPLATE,
    SFT_FORMAT_PROMPT
)
from open_r1.vlm_modules import *
from transformers import Trainer, TrainingArguments
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from open_r1.qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn, monkey_patch_qwen2_5vl_forward, monkey_patch_torch_load

monkey_patch_qwen2_5vl_flash_attn()    
monkey_patch_torch_load()


@dataclass
class SFTScriptArguments(ScriptArguments):
    """Script arguments for SFT training with ProcTHOR Active VQA."""
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
        metadata={"help": "Random seed for train/validation split."},
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
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    vlm_lr: Optional[float] = field(
        default=1e-6,
        metadata={"help": "Learning rate for VLM backbone. If None, uses training_args.learning_rate"},
    )


@dataclass
class SFTModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


class ProcTHORSFTTrainer(Trainer):
    """Trainer for supervised fine-tuning on ProcTHOR Active VQA task.
    
    Trains the model to predict action parameters (head, fwd, view) in the format:
    <think> reasoning </think><head> X </head><fwd> Y </fwd><view> Z </view>
    """
    
    def __init__(
        self,
        model,
        args,
        vlm_module,
        processing_class,
        train_dataset=None,
        eval_dataset=None,
        ntl_weight=0.0,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            **kwargs
        )
        self.vlm_module = vlm_module
        # Store original processor separately
        self.vlm_processing_class = processing_class
        
        # Get tokenizer for pad_token_id
        self._tokenizer = processing_class.tokenizer if hasattr(processing_class, 'tokenizer') else processing_class
        
        # Suppress tokenization warning
        self.model.warnings_issued["estimate_tokens"] = True
    
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute language modeling loss for action prediction using teacher forcing."""
        # Extract data from inputs
        images = []
        prompts = []
        gt_actions = []
        
        for item in inputs:
            # Load images
            if "image_path" in item and item["image_path"] is not None:
                img_paths = item["image_path"] if isinstance(item["image_path"], list) else [item["image_path"]]
                # Load first image
                img_path = img_paths[0]
                img = PIL.Image.open(img_path).convert('RGB')
                # Ensure minimum dimensions
                w, h = img.size
                if w < 28 or h < 28:
                    if w < h:
                        new_w, new_h = 28, int(h * (28/w))
                    else:
                        new_h, new_w = 28, int(w * (28/h))
                    img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
                images.append(img)
            
            # Get prompt
            prompts.append(item["prompt"])
            
            # Get ground truth action
            if "gt_action" in item and item["gt_action"] is not None:
                gt_actions.append(item["gt_action"])
            else:
                raise ValueError(f"Item missing 'gt_action' field: {item}")
        
        # Prepare full input texts (prompt + target)
        full_texts = []
        prompts_text = self.vlm_module.prepare_prompt(self.vlm_processing_class, inputs)
        for prompt_text, gt_action in zip(prompts_text, gt_actions):
            full_texts.append(prompt_text + gt_action)
        
        # Prepare model inputs with full text
        model_inputs, additional_output = self.vlm_module.prepare_model_inputs(
            self.vlm_processing_class,
            full_texts,
            images,
            return_tensors="pt",
            padding=True,
            padding_side="right",  # Right padding for labels
            add_special_tokens=False,
        )
        
        # Move to device
        model_inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in model_inputs.items()}
        
        # Create labels (same as input_ids but with -100 for prompt tokens)
        input_ids = model_inputs["input_ids"]
        labels = input_ids.clone()
        
        # Get prompt lengths to mask them in labels
        prompt_inputs, _ = self.vlm_module.prepare_model_inputs(
            self.vlm_processing_class,
            prompts_text,
            images,
            return_tensors="pt",
            padding=True,
            padding_side="right",
            add_special_tokens=False,
        )
        prompt_lengths = (prompt_inputs["input_ids"] != self._tokenizer.pad_token_id).sum(dim=1)
        
        # Mask prompt tokens in labels
        for i, prompt_len in enumerate(prompt_lengths):
            labels[i, :prompt_len] = -100
        
        # Also mask padding tokens
        labels[labels == self._tokenizer.pad_token_id] = -100
        
        # Forward pass with labels
        outputs = model(
            input_ids=input_ids,
            attention_mask=model_inputs.get("attention_mask"),
            labels=labels,
            **{k: v for k, v in model_inputs.items() if k not in ["input_ids", "attention_mask"]}
        )
        
        # Get the main cross entropy loss
        ce_loss = outputs.loss
        
        # Log loss periodically
        if self.state.global_step % 10 == 0:
            self.log({"train/loss": ce_loss.item()})
        
        return (ce_loss, outputs) if return_outputs else ce_loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override prediction_step to handle list inputs during evaluation."""
        # Compute loss same as training
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        
        # Ensure loss is a tensor
        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, device=model.device)
        
        # Return loss, None for logits, None for labels
        return (loss, None, None)



def main(script_args, training_args, model_args):
    # Ensure we don't remove unused columns since we use custom data collator
    training_args.remove_unused_columns = False
    
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print("Using VLM module:", vlm_module_cls.__name__)
    
    # Load the JSONL datasets
    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")
    
    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    
    def make_conversation_from_jsonl(example: dict, image_folder: str) -> dict:
        required = [
            "question",
            "gt_action",
            "question_image",
        ]
        for k in required:
            if k not in example:
                raise ValueError(f"Missing required field '{k}' in data example")

        # Resolve image paths from 'question_image' (string or list)
        qi = example["question_image"]
        if isinstance(qi, str):
            image_paths = [qi if os.path.isabs(qi) else os.path.join(image_folder, qi)]
        elif isinstance(qi, list):
            image_paths = [p if os.path.isabs(p) else os.path.join(image_folder, p) for p in qi]
        else:
            raise ValueError(f"Unsupported question_image type: {type(qi)}")

        # Build action prompt similar to GRPO
        vqa_question = example["question"]
        action_question = ACTION_PROMPT_TEMPLATE.format(question=vqa_question) + SFT_FORMAT_PROMPT

        # Build SFT chat prompt (use first image)
        prompt = [{
            "role": "user",
            "content": [
                {"type": "image", "text": None},
                {"type": "text", "text": action_question},
            ]
        }]

        # Convert heading from [0, 360) to (-180, 180]
        rot0 = int(example['gt_action'][0])
        if rot0 > 180:
            rot0 -= 360
        view0 = int(example['gt_action'][2])
        if view0 > 180:
            view0 -= 360
        target_text = (
            f"<head> {rot0} </head> "
            f"<fwd> {int(example['gt_action'][1])} </fwd> "
            f"<view> {view0} </view> "
        ) 

        return {
            "image_path": image_paths,
            "vqa_question": vqa_question,
            "action_question": action_question,
            "gt_action": target_text,
            "prompt": prompt,
        }

    all_data = []
    for data_file, image_folder in zip(data_files, image_folders):
        with open(data_file, 'r') as f:
            for line in f:
                raw = json.loads(line)
                try:
                    ex = make_conversation_from_jsonl(raw, image_folder)
                except Exception as e:
                    print(f"Warning: skip item due to preprocessing error: {e}")
                    continue
                if ex.get("gt_action") is None:
                    print("Warning: Item missing 'gt_action' field, skipping item")
                    continue
                all_data.append(ex)
    
    print(f"Loaded {len(all_data)} training examples")
    
    if len(all_data) == 0:
        raise ValueError("No valid training examples found. Make sure 'gt_action' field exists in your data.")
    
    # Create dataset
    dataset = Dataset.from_list(all_data)
    
    # Split dataset for validation if requested
    splits = {'train': dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(
            test_size=script_args.val_split_ratio,
            seed=script_args.val_split_seed
        )
        splits['train'] = train_val_split['train']
        splits['validation'] = train_val_split['test']
        print(f"Train split: {len(splits['train'])} examples")
        print(f"Validation split: {len(splits['validation'])} examples")
    
    # Initialize model
    model_init_kwargs = getattr(training_args, 'model_init_kwargs', None) or {}
    model_init_kwargs["attn_implementation"] = "flash_attention_2"
    if model_init_kwargs.get("torch_dtype") is None:
        model_init_kwargs["torch_dtype"] = "bfloat16"
    
    torch_dtype = model_init_kwargs.get("torch_dtype")
    if isinstance(torch_dtype, str) and torch_dtype != "auto":
        torch_dtype = getattr(torch, torch_dtype)
    
    model_init_kwargs["use_cache"] = False if training_args.gradient_checkpointing else model_init_kwargs.get("use_cache")
    model_init_kwargs["trust_remote_code"] = True
    
    model_cls = vlm_module_cls().get_model_class(model_args.model_name_or_path, model_init_kwargs)
    model = model_cls.from_pretrained(model_args.model_name_or_path, **model_init_kwargs)
    
    # Apply LoRA if specified
    peft_config = get_peft_config(model_args)
    if peft_config is not None:
        from peft import get_peft_model
        print("Applying LoRA...")
        
        def find_all_linear_names(model, multimodal_keywords):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                    continue
                if isinstance(module, cls):
                    lora_module_names.add(name)
            lora_module_names.discard("embed_tokens")
            return list(lora_module_names)
        
        vision_keywords = vlm_module_cls().get_vision_modules_keywords()
        target_modules = find_all_linear_names(model, vision_keywords)
        peft_config.target_modules = target_modules
        print(f"LoRA target modules: {len(target_modules)} modules")
        model = get_peft_model(model, peft_config)
    
    # Freeze vision modules if requested
    if model_args.freeze_vision_modules:
        print("Freezing vision modules...")
        vision_keywords = vlm_module_cls().get_vision_modules_keywords()
        for n, p in model.named_parameters():
            if any(keyword in n for keyword in vision_keywords):
                p.requires_grad = False
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Initialize processing class
    processing_cls = vlm_module_cls().get_processing_class()
    processing_class = processing_cls.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    
    # Set max/min pixels for Qwen2VL
    if hasattr(processing_class, 'image_processor'):
        if script_args.max_pixels is not None:
            processing_class.image_processor.max_pixels = script_args.max_pixels
        if script_args.min_pixels is not None:
            processing_class.image_processor.min_pixels = script_args.min_pixels
    
    vlm_module_cls().post_model_init(model, processing_class)
    
    # Data collator - return items as-is
    def data_collator(features):
        return features
    
    # Initialize trainer
    # Initialize trainer
    trainer = ProcTHORSFTTrainer(
        model=model,
        args=training_args,
        vlm_module=vlm_module_cls(),
        processing_class=processing_class,
        train_dataset=splits['train'],
        eval_dataset=splits.get('validation') if training_args.eval_strategy != "no" else None,
        data_collator=data_collator,
    )
    
    # Train
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print("Resuming from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    # Save final model
    if training_args.save_strategy != "no":
        final_model_path = pathlib.Path(training_args.output_dir)
        trainer.save_model(final_model_path)
        print(f"Final model saved to: {final_model_path}")
    
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, TrainingArguments, SFTModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    if training_args.deepspeed and "zero3" in training_args.deepspeed:
        print("Zero3 is used, applying qwen2_5vl forward monkey patch")
        monkey_patch_qwen2_5vl_forward()
    
    main(script_args, training_args, model_args)
