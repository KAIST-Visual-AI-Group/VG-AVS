import os

import torch
from open_r1.vlm_modules import InvernVLModule, Qwen2VLModule
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)

DEBUG_MODE = str(os.getenv("DEBUG_MODE", "0")) == "1"
VERIFIER_MODEL_PATH = os.getenv(
    "VERIFIER_MODEL_PATH", "qwen2.5vl:7b"
)  # alias allowed (e.g., qwen2.5vl:3b, qwen2.5vl:7b)

ALIAS_MAP = {
    "qwen2.5vl:3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl:3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5_vl:3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5vl:7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl:7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5_vl:7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5vl:72b": "Qwen/Qwen2.5-VL-72B-Instruct",
    "qwen2.5-vl:72b": "Qwen/Qwen2.5-VL-72B-Instruct",
    "qwen2.5_vl:72b": "Qwen/Qwen2.5-VL-72B-Instruct",
}

# Globals for lazy-loaded verifier components
verifier_model = None
verifier_processor = None


def resolve_model_path(raw_path: str) -> str:
    key = raw_path.strip().lower()
    return ALIAS_MAP.get(key, raw_path)


def initialize_verifier():
    """Lazy load frozen verifier model (Qwen2.5-VL) for action-based accuracy reward.

    Can override model path via VERIFIER_MODEL_PATH env var. Supported aliases: qwen2.5vl:3b, qwen2.5vl:7b
    """
    global verifier_model, verifier_processor
    if verifier_model is None:
        target_path = resolve_model_path(VERIFIER_MODEL_PATH)
        tried = []
        candidates = [target_path]
        if "Instruct" not in target_path and "-Instruct" not in target_path:
            if target_path.endswith("-7B") or target_path.endswith("-3B"):
                candidates.append(target_path + "-Instruct")
        # Choose explicit device to avoid device_map/accelerate dispatch conflicts under ZeRO-3
        env_device = os.getenv("VERIFIER_DEVICE", "auto").lower()
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        if env_device == "cpu":
            chosen_device = "cpu"
        elif env_device.startswith("cuda") and torch.cuda.is_available():
            # Allow VERIFIER_DEVICE=cuda or cuda:<idx>
            if ":" in env_device:
                chosen_device = env_device
            else:
                chosen_device = f"cuda:{local_rank}"
        else:
            # auto: prefer current rank GPU if available else CPU
            chosen_device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

        print(
            f"[action_accuracy] Verifier device selection: {chosen_device} (env: {env_device}, local_rank: {local_rank})"
        )
        # Precision selection: allow override via VERIFIER_PRECISION in {bf16, fp16, float16, float32}
        env_prec = os.getenv("VERIFIER_PRECISION", "").lower()

        def _auto_dtype():
            if "cuda" in chosen_device and torch.cuda.is_available():
                bf16_ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
                return torch.bfloat16 if bf16_ok else torch.float16
            return torch.float32

        if env_prec in {"bf16", "bfloat16"}:
            dtype = torch.bfloat16
        elif env_prec in {"fp16", "float16"}:
            dtype = torch.float16
        elif env_prec in {"fp32", "float32"}:
            dtype = torch.float32
        else:
            dtype = _auto_dtype()
        for cand in candidates:
            verifier_processor = AutoProcessor.from_pretrained(
                cand, trust_remote_code=True
            )
            # Decide which loader
            if any(
                tag in cand.lower() for tag in ["2.5-vl", "2_5-vl", "2.5_vl", "2_5_vl"]
            ):
                loader_cls = AutoModelForVision2Seq
            else:
                loader_cls = Qwen2VLForConditionalGeneration
            # Important: Do NOT pass device_map or low_cpu_mem_usage=True when using DeepSpeed ZeRO-3
            # Load on host first, then move to explicit device to avoid accelerate/deepspeed integration
            verifier_model = loader_cls.from_pretrained(
                cand,
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
                trust_remote_code=True,
            )
            try:
                verifier_model.to(chosen_device)
            except Exception as _move_e:
                # Fallback to CPU if device move failed (e.g., OOM or invalid index)
                print(
                    f"[action_accuracy] Failed to move verifier to {chosen_device}: {_move_e}. Falling back to CPU."
                )
                verifier_model.to("cpu")
            verifier_model.eval()
            for p in verifier_model.parameters():
                p.requires_grad_(False)
            break
        if verifier_model is None:
            if DEBUG_MODE:
                print(f"[action_accuracy] All load attempts failed: {tried}")
    return verifier_model, verifier_processor


def get_vlm_module(model_name_or_path):
    import json
    import os

    # Check if it's a checkpoint directory
    if os.path.isdir(model_name_or_path):
        # First check for config.json (full model)
        config_path = os.path.join(model_name_or_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                # Check model_type or architectures in config
                model_type = config.get("model_type", "").lower()
                architectures = config.get("architectures", [])
                arch_str = " ".join(architectures).lower()

                if "qwen" in model_type or "qwen" in arch_str:
                    return Qwen2VLModule
                elif "internvl" in model_type or "internvl" in arch_str:
                    return InvernVLModule

        # Check for adapter_config.json (LoRA adapter)
        adapter_config_path = os.path.join(model_name_or_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)
                # Get base model name from adapter config
                base_model = adapter_config.get("base_model_name_or_path", "")
                if base_model:
                    # Recursively check the base model
                    return get_vlm_module(base_model)

    # Fallback to checking the path string
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")
