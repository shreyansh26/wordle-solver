#!/usr/bin/env bash
set -euo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

# Configuration (override via environment variables)
# Required: MODEL (HF hub id or local path)
MODEL=${MODEL:-"/mnt/ssd2/shreyansh/models/qwen3/exp_2025-08-08T00:53:20_qwen3_4b_fsdp_packing=ffd_flash_attn_fsdp2_torch_compile_dcp_kimi_k2_v2_sft/epoch_5/step_final"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-9900}
TP=${TP:-1}
DP=${DP:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
UVICORN_LOG_LEVEL=${UVICORN_LOG_LEVEL:-"info"}

echo "[vLLM] MODEL=${MODEL} HOST=${HOST} PORT=${PORT} TP=${TP} DP=${DP} MAX_MODEL_LEN=${MAX_MODEL_LEN}"

python -m verifiers.inference.vllm_server \
  --model "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --data-parallel-size "${DP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --dtype auto \
  --uvicorn-log-level "${UVICORN_LOG_LEVEL}"



