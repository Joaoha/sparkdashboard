#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DOCKER="$SCRIPT_DIR/docker-command.sh"
MODEL_DIR=${SPARK_MODEL_DIR:-$HOME/models/hf}
VLLM_IMAGE=${SPARK_VLLM_IMAGE:-vllm/vllm-openai:nightly}
MAX_MODEL_LEN=${MISTRAL_MAX_MODEL_LEN:-37888}
GPU_UTIL=${MISTRAL_GPU_UTIL:-0.80}
PORT=${MISTRAL_PORT:-8002}
MODEL_NAME=${MISTRAL_MODEL_NAME:-Mistral-Medium-3.5-128B-NVFP4}
EXTRA_ARGS=${MISTRAL_EXTRA_ARGS:-}
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
if [ ! -e "$MODEL_PATH/model.safetensors.index.json" ] && [ ! -e "$MODEL_PATH/config.json" ]; then
  echo "Mistral model missing at $MODEL_PATH. Run: sparkdashboard-download-models mistral" >&2
  exit 2
fi
for svc in qwen-nvfp4-vllm.service ornith-vllm.service krea-2.service qwen-image.service z-image.service flux2.service hidream-o1.service domainshuttle-web.service; do
  systemctl --user stop "$svc" >/dev/null 2>&1 || true
done
"$DOCKER" rm -f qwen-nvfp4-vllm ornith-vllm mistral-medium-vllm >/dev/null 2>&1 || true
exec "$DOCKER" run --rm --init \
  --name mistral-medium-vllm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e VLLM_ATTENTION_BACKEND=FLASHINFER \
  -e VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=${VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER:-1} \
  -e VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-1} \
  -e VLLM_TRITON_FORCE_FIRST_CONFIG=${VLLM_TRITON_FORCE_FIRST_CONFIG:-0} \
  -e VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE=${VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE:-1} \
  -e VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING=${VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING:-1} \
  -e VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=${VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE:-413138944} \
  -e HF_HOME=/models/.cache/huggingface \
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
    --kv-cache-dtype fp8_e4m3 \
    --attention-backend flashinfer \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs 1 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --async-scheduling \
    --enable-prefix-caching \
    --tokenizer-mode mistral \
    --tool-call-parser mistral \
    --enable-auto-tool-choice \
    --reasoning-parser mistral \
    --language-model-only \
    --disable-hybrid-kv-cache-manager \
    --load-format auto \
    $EXTRA_ARGS
