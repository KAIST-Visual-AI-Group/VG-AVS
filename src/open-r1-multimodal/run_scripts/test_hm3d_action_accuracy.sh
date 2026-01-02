#!/usr/bin/env bash
set -euo pipefail

cd src/open-r1-multimodal

# ===== Project Root =====
# Change this path to your project root directory
PROJECT_ROOT=${PROJECT_ROOT:-/path/to/project}

MODEL_PATH=${MODEL_PATH:-${PROJECT_ROOT}/src/open-r1-multimodal/output/${MODEL_NAME}}

IMG_ROOT=${IMG_ROOT:-/path/to/dataset}
NUM_SAMPLES=${NUM_SAMPLES:--1} # use all samples

# Verifier settings
VERIFIER_MODEL=${VERIFIER_MODEL:-gemini-2.5-flash}
VERIFIER_DEVICE=${VERIFIER_DEVICE:-cuda:0}
GPU_DEVICE=${GPU_DEVICE:-0}
STUDENT_DEVICE=${STUDENT_DEVICE:-cuda:0}


export GEMINI_API_KEY="<GEMINI_API_KEY>"
export OPENAI_API_KEY="<OPENAI_API_KEY>"


echo "Testing action prediction model for HM3D scene..."
echo "Model: ${MODEL_PATH}"
echo "Image root: ${IMG_ROOT}"
echo "GPU device for rendering: ${GPU_DEVICE}"
echo ""

TEST_JSONL=${PROJECT_ROOT}/data/avs_hm3d_overall.jsonl

MODEL_NAME=$(basename ${MODEL_PATH})
OUTPUT_DIR=./output/${MODEL_NAME}/hm3d_eval/$(date +%Y%m%d_%H%M%S)

export PYTHONPATH=${PYTHONPATH:-}:${PROJECT_ROOT}/src/open-r1-multimodal/src

python src/open_r1/test_hm3d_action_accuracy.py \
  --model_path "${MODEL_PATH}" \
  --test_jsonl "${TEST_JSONL}" \
  --output_dir "${OUTPUT_DIR}" \
  --verifier_model ${VERIFIER_MODEL} \
  --max_new_tokens 512 \
  --verifier_max_tokens 48 \
  --device ${STUDENT_DEVICE} \
  --verifier_device ${VERIFIER_DEVICE} \
  --gpu_device ${GPU_DEVICE} \
  --use_gemini_verifier \
  --image_root "${IMG_ROOT}" \


echo ""
echo "Testing complete! Check results at: ${OUTPUT_DIR}"