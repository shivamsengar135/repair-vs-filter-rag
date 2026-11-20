#!/usr/bin/env bash
# Start vLLM (official image, pinned tag) in Docker with the GPU, serving on 127.0.0.1:8000.
# Used because WSL Ubuntu has no C compiler (vLLM's JIT needs one); the image ships its own toolchain.
# Usage: bash scripts/serve_vllm_docker.sh [MODEL] [REVISION] [extra vllm args...]
# Stop:  docker rm -f vllm
set -euo pipefail
source "$(dirname "$0")/wsl_env.sh"
IMAGE="vllm/vllm-openai:v0.33.0"
MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
REV="${2:-a09a35458c702b33eeacc393d103063234e8bc28}"
shift $(( $# > 2 ? 2 : $# ))
LOG="logs/vllm_docker_$(date +%Y%m%d_%H%M%S).log"
echo "$LOG" > logs/vllm_serve_latest.txt
docker rm -f vllm > /dev/null 2>&1 || true
docker run -d --name vllm --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v "$HF_HOME:/root/.cache/huggingface" \
  -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  "$IMAGE" "$MODEL" --revision "$REV" \
  --host 0.0.0.0 --port 8000 --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" --seed 0 --dtype "${DTYPE:-bfloat16}" "$@" > /dev/null
echo "started container 'vllm' ($IMAGE); log: $LOG"
# Wait (max 15 min) for readiness.
for _ in $(seq 1 180); do
  docker logs vllm > "$LOG" 2>&1 || true
  if curl -sf -m 3 http://127.0.0.1:8000/health > /dev/null; then echo "VLLM_READY"; exit 0; fi
  if [ "$(docker inspect -f '{{.State.Running}}' vllm 2>/dev/null)" != "true" ]; then
    echo "VLLM_DIED"; tail -n 40 "$LOG"; exit 1
  fi
  sleep 5
done
echo "VLLM_TIMEOUT"; tail -n 40 "$LOG"; exit 1
