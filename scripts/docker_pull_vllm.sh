#!/usr/bin/env bash
# Pull the official vLLM image (pinned tag) in a detached tmux session with retries.
set -euo pipefail
source "$(dirname "$0")/wsl_env.sh"
IMAGE="${1:-vllm/vllm-openai:v0.33.0}"
LOG="logs/docker_pull_vllm_$(date +%Y%m%d_%H%M%S).log"
echo "$LOG" > logs/docker_pull_latest.txt
tmux kill-session -t pull 2>/dev/null || true
tmux new-session -d -s pull "
  for attempt in \$(seq 1 20); do
    echo \"=== attempt \$attempt \$(date -u +%FT%TZ)\" >> $LOG
    if docker pull -q $IMAGE >> $LOG 2>&1; then
      docker image inspect $IMAGE --format 'id={{.Id}} size={{.Size}} digests={{.RepoDigests}}' >> $LOG 2>&1
      echo PULL_OK >> $LOG; exit 0
    fi
    sleep 15
  done
  echo PULL_FAILED >> $LOG"
echo "started tmux session 'pull'; log: $LOG"
