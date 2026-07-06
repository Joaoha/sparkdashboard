#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR=${SPARK_MODEL_DIR:-$HOME/models/hf}
VLLM_IMAGE=${SPARK_VLLM_IMAGE:-vllm/vllm-openai:nightly}
PORT=${QWEN_PORT:-8000}
GPU_UTIL=${QWEN_GPU_UTIL:-0.48}
MAX_MODEL_LEN=${QWEN_MAX_MODEL_LEN:-262144}
MAX_BATCHED_TOKENS=${QWEN_MAX_BATCHED_TOKENS:-8192}
MODEL_NAME=${QWEN_MODEL_NAME:-Qwen3.6-35B-A3B-NVFP4}
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
if [ ! -e "$MODEL_PATH/config.json" ]; then
  echo "Qwen model missing at $MODEL_PATH. Run: sparkdashboard-download-models qwen" >&2
  exit 2
fi
systemctl --user stop mistral-medium-vllm.service ornith-vllm.service >/dev/null 2>&1 || true
docker rm -f mistral-medium-vllm ornith-vllm qwen-nvfp4-vllm >/dev/null 2>&1 || true
exec docker run --rm --init \
  --name qwen-nvfp4-vllm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$MODEL_DIR:/models:ro" \
  -p 0.0.0.0:${PORT}:${PORT} \
  "$VLLM_IMAGE" \
  "/models/$MODEL_NAME" \
    --served-model-name "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --quantization modelopt \
    --kv-cache-dtype fp8 \
    --attention-backend flashinfer \
    --moe-backend marlin \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs 1 \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --enable-chunked-prefill \
    --async-scheduling \
    --enable-prefix-caching \
    --load-format fastsafetensors \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_xml \
    --enable-auto-tool-choice
