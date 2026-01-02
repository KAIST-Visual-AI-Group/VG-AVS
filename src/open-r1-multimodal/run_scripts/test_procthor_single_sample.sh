#!/usr/bin/env bash
set -euo pipefail

cd src/open-r1-multimodal

# ===== Project Root =====
# Change this path to your project root directory
PROJECT_ROOT=${PROJECT_ROOT:-/path/to/project}

MODEL_PATH=${MODEL_PATH:-${PROJECT_ROOT}/src/open-r1-multimodal/output/grpo-procthor}

IMG_ROOT=${IMG_ROOT:-/path/to/dataset}
SAMPLE_IDX=${SAMPLE_IDX:-0}

# Verifier settings
#CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
VERIFIER_MODEL=${VERIFIER_MODEL:-gemini-2.5-flash}
VERIFIER_DEVICE=${VERIFIER_DEVICE:-cuda:0}
GPU_DEVICE=${GPU_DEVICE:-0}
STUDENT_DEVICE=${STUDENT_DEVICE:-cuda:0}

# API Keys
export GEMINI_API_KEY="<GEMINI_API_KEY>"
#export OPENAI_API_KEY="<OPENAI_API_KEY>"  # Uncomment and set when using GPT

echo "Testing ProcTHOR action prediction model (Single Sample)..."
echo "Model: ${MODEL_PATH}"
echo "Image root: ${IMG_ROOT}"
echo "Sample Index: ${SAMPLE_IDX}"
echo ""

# existence
EXISTENCE_TEST_JSONL=${PROJECT_ROOT}/data/avs_procthor_existence.jsonl
MODEL_NAME=$(basename ${MODEL_PATH})
OUTPUT_DIR=./output/${MODEL_NAME}/test/${SAMPLE_IDX}/$(date +%Y%m%d_%H%M%S)

export PYTHONPATH=${PYTHONPATH:-}:${PROJECT_ROOT}/src/open-r1-multimodal/src

python src/open_r1/test_procthor_single_sample.py \
  --model_path "${MODEL_PATH}" \
  --test_jsonl "${EXISTENCE_TEST_JSONL}" \
  --output_dir "${OUTPUT_DIR}" \
  --sample_idx ${SAMPLE_IDX} \
  --verifier_model ${VERIFIER_MODEL} \
  --max_new_tokens 256 \
  --verifier_max_tokens 48 \
  --device ${STUDENT_DEVICE} \
  --verifier_device ${VERIFIER_DEVICE} \
  --gpu_device ${GPU_DEVICE} \
  --use_gemini_verifier \
  --image_root "${IMG_ROOT}" \
