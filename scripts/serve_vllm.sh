#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server in a detached tmux session (survives the calling shell).
# Usage: bash scripts/serve_vllm.sh [MODEL] [REVISION] [extra vllm args...]
# Stop:  tmux kill-session -t vllm
set -euo pipefail
source "$(dirname "$0")/wsl_env.sh"
MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
REV="${2:-a09a35458c702b33eeacc393d103063234e8bc28}"
shift $(( $# > 2 ? 2 : $# ))
LOG="logs/vllm_serve_$(date +%Y%m%d_%H%M%S).log"
tmux kill-session -t vllm 2>/dev/null || true
tmux new-session -d -s vllm \
  "HF_HOME=$HF_HOME HF_HUB_OFFLINE=1 $UV_PROJECT_ENVIRONMENT/bin/vllm serve $MODEL --revision $REV \
     --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len 8192 \
     --gpu-memory-utilization 0.90 --seed 0 $* > $LOG 2>&1"
echo "$LOG" > logs/vllm_serve_latest.txt
echo "started tmux session 'vllm'; log: $LOG"
# Wait (max 15 min) for readiness.
for _ in $(seq 1 180); do
  if curl -sf -m 3 http://127.0.0.1:8000/health > /dev/null; then echo "VLLM_READY"; exit 0; fi
  if ! tmux has-session -t vllm 2>/dev/null; then echo "VLLM_DIED"; tail -n 40 "$LOG"; exit 1; fi
  sleep 5
done
echo "VLLM_TIMEOUT"; tail -n 40 "$LOG"; exit 1
