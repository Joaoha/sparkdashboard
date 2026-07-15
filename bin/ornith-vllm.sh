#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DOCKER="$SCRIPT_DIR/docker-command.sh"
MODEL_DIR=${SPARK_MODEL_DIR:-$HOME/models/hf}
VLLM_IMAGE=${SPARK_VLLM_IMAGE:-vllm/vllm-openai:nightly}
PORT=${ORNITH_PORT:-8001}
GPU_UTIL=${ORNITH_GPU_UTIL:-0.82}
MAX_MODEL_LEN=${ORNITH_MAX_MODEL_LEN:-262144}
MAX_BATCHED_TOKENS=${ORNITH_MAX_BATCHED_TOKENS:-8192}
MODEL_NAME=${ORNITH_MODEL_NAME:-Ornith-1.0-35B}
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
if [ ! -e "$MODEL_PATH/config.json" ]; then
  echo "Ornith model missing at $MODEL_PATH. Run: sparkdashboard-download-models ornith" >&2
  exit 2
fi
# Ornith BF16 needs a clean unified-memory pool. Keep it mutually exclusive with Qwen/Mistral.
for svc in qwen-nvfp4-vllm.service mistral-medium-vllm.service krea-2.service qwen-image.service z-image.service flux2.service hidream-o1.service domainshuttle-web.service; do
  systemctl --user stop "$svc" >/dev/null 2>&1 || true
done
"$DOCKER" rm -f qwen-nvfp4-vllm ornith-vllm mistral-medium-vllm >/dev/null 2>&1 || true
exec "$DOCKER" run --rm --init \
  --name ornith-vllm \
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
    --dtype bfloat16 \
    --kv-cache-dtype fp8 \
    --attention-backend flashinfer \
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
