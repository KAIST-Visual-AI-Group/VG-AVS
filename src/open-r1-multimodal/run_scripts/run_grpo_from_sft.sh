#!/usr/bin/env bash
# set -euo pipefail

# Check Python environment
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

#CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
DATA_JSONL=${DATA_JSONL:-/home/daehyeonchoi/embodied/AVS_release/data/avs_procthor_train.jsonl}
IMG_ROOT=${IMG_ROOT:-/data/daehyeon/procthor_active_vqa} 

cd src/open-r1-multimodal

# Configuration
BASE_MODEL=${BASE_MODEL:-/data/andy2884/workspace/VG-AVS/test/VLM-R1/src/open-r1-multimodal/output/sft-procthor/checkpoint-413}
GRPO_RUN_NAME=${GRPO_RUN_NAME:-grpo-from-sft}

# Torch/distributed settings
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

if [[ -z "${DATA_JSONL:-}" ]]; then
  echo "ERROR: Please set DATA_JSONL to your dataset JSONL file." >&2
  exit 1
fi

if [[ -z "${IMG_ROOT:-}" ]]; then
  echo "ERROR: Please set IMG_ROOT to your image root directory." >&2
  exit 1
fi

GRPO_OUTPUT_DIR="/data/daehyeon/checkpoint/avs_release/${GRPO_RUN_NAME}"

export WANDB_PROJECT=${WANDB_PROJECT:-vgavs}
REPORT_TO=${REPORT_TO:-wandb}

echo ""
echo "========================================="
echo "GRPO Fine-tuning for Procthor Active VQA"
echo "========================================="
echo "Model: ${BASE_MODEL}"
echo "Output: ${GRPO_OUTPUT_DIR}"
echo ""

#export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export DEBUG_MODE=${DEBUG_MODE:-1}
export LOG_PATH="./debug_log_${GRPO_RUN_NAME}.txt"

TORCHRUN="$(dirname $(which python))/torchrun"
echo "Using torchrun: ${TORCHRUN}"

${TORCHRUN} --nproc_per_node="8" \
  --nnodes="1" \
  --node_rank="0" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="12346" \
  src/open_r1/grpo_jsonl_procthor_active_qa.py \
  --deepspeed local_scripts/zero2.json \
  --output_dir ${GRPO_OUTPUT_DIR} \
  --model_name_or_path ${BASE_MODEL} \
  --data_file_paths "${DATA_JSONL}" \
  ${IMG_ROOT:+--image_folders "${IMG_ROOT}"} \
  --dataset_name None \
  --reward_funcs format action_accuracy \
  --grpo_reward_weights 0.3 1 \
  --num_generations 16 \
  --beta 0.04 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --logging_steps 1 \
  --freeze_vision_modules true \
  --bf16 \
  --torch_dtype bfloat16 \
  --data_seed 42 \
  --report_to ${REPORT_TO} \
  --gradient_checkpointing true \
  --attn_implementation flash_attention_2 \
  --num_train_epochs 5 \
  --run_name ${GRPO_RUN_NAME} \
  --save_strategy epoch \
  --save_only_model false \
  --learning_rate 1e-6 \
  --num_iterations 1 \
  --use_refine \
  --max_completion_length 512 \
  # --save_steps 200 \
  # --save_total_limit 4 \

echo ""
echo "========================================="
echo "Training Pipeline Complete!"
echo "========================================="
echo ""