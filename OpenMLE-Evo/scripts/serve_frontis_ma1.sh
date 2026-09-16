#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# serve_frontis_ma1.sh — launch an OpenAI-compatible vLLM server for
# FrontisAI/Frontis-MA1-35B, to run ON THE RENTED GPU BOX.
#
# Install once:
#     pip install vllm
#
# Picking TP (tensor-parallel-size): set it to the number of GPUs you want to
# shard the model across. BF16 weights are ~70 GB:
#     1×H100-80GB / 1×A100-80GB  -> TP=1
#     2×A100/L40S-40-48GB        -> TP=2
#     FP8/AWQ (~35 GB)           -> TP=1 on a 40-48GB card
# (~3B params are active per token — MoE — so throughput is high and cheap.)
#
# Once this is serving, the OpenMLE-Evo runner connects two ways
# (see .cyberml/LOCAL_MODEL.md):
#   - SSH tunnel:  run_naturebench_local.py --model-ssh-host user@box \
#                    --model-ssh-port ${PORT} --model-id ${SERVED_NAME}
#   - Direct URL:  run_naturebench_local.py \
#                    --model-base-url http://box:${PORT}/v1 --model-id ${SERVED_NAME}
#                    (with PRIMARY_KEY=EMPTY in .env)
#
# All settings are env-var overridable, e.g.:  TP=2 PORT=8001 ./serve_frontis_ma1.sh
# ---------------------------------------------------------------------------

MODEL="${MODEL:-FrontisAI/Frontis-MA1-35B}"
SERVED_NAME="${SERVED_NAME:-frontis-ma1}"
PORT="${PORT:-8000}"
TP="${TP:-1}"                 # number of GPUs to shard across
MAX_LEN="${MAX_LEN:-32768}"   # trained SFT ceiling
GPU_UTIL="${GPU_UTIL:-0.90}"

echo "Serving Frontis-MA1-35B with vLLM:"
echo "  MODEL        = ${MODEL}"
echo "  SERVED_NAME  = ${SERVED_NAME}   (use this as --model-id in the runner)"
echo "  PORT         = ${PORT}"
echo "  TP           = ${TP}            (tensor-parallel-size / #GPUs)"
echo "  MAX_LEN      = ${MAX_LEN}"
echo "  GPU_UTIL     = ${GPU_UTIL}"

exec vllm serve "${MODEL}" \
  --served-model-name "${SERVED_NAME}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --language-model-only
