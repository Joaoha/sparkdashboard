#!/usr/bin/env bash
set -euo pipefail
cd /opt/domainshuttle/repo
export HF_HOME=/opt/domainshuttle/hf-cache
export VIDEOX_ATTENTION_TYPE=${VIDEOX_ATTENTION_TYPE:-SDPA}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

INPUT_JSON=${INPUT_JSON:-/opt/domainshuttle/inputs/smoke.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-/opt/domainshuttle/outputs/smoke}
MODEL_DIR=${MODEL_DIR:-/opt/domainshuttle/models/Diffusion_Transformers/Wan2.2-DomainShuttle-A14B}
HEIGHT=${HEIGHT:-256}
WIDTH=${WIDTH:-448}
VIDEO_LENGTH=${VIDEO_LENGTH:-17}
STEPS=${STEPS:-4}
FPS=${FPS:-8}
SEED=${SEED:-42}
SHIFT=${SHIFT:-5}
GUIDANCE_A=${GUIDANCE_A:-4.0}
GUIDANCE_B=${GUIDANCE_B:-3.0}
MEMORY_MODE=${MEMORY_MODE:-sequential_cpu_offload}

mkdir -p "$OUTPUT_DIR"
exec /opt/domainshuttle/.venv/bin/python examples/wan2.2_domainshuttle/predict_r2v_batch.py \
  --input_json "$INPUT_JSON" \
  --output_dir "$OUTPUT_DIR" \
  --domain_model_name "$MODEL_DIR" \
  --config_path config/wan2.2/wan_civitai_t2v.yaml \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --video_length "$VIDEO_LENGTH" \
  --fps "$FPS" \
  --num_inference_steps "$STEPS" \
  --guidance_scale "$GUIDANCE_A" "$GUIDANCE_B" \
  --shift "$SHIFT" \
  --seed "$SEED" \
  --ulysses_degree 1 \
  --ring_degree 1 \
  --memory_mode "$MEMORY_MODE" \
  --max_reference_num 1
