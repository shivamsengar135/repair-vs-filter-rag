#!/usr/bin/env bash
# Progress of the confirmatory run: log tail, run folders, cache sizes, GPU.
cd "$(dirname "$0")/.."
tail -n 8 logs/confirm_runs.log
for d in results/confirm/bio/*/; do [ -d "$d" ] && echo "$d: $(ls "$d" | tr '\n' ' ')"; done 2>/dev/null
ls -la results/confirm/bio 2>/dev/null | head
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
tmux has-session -t confirm 2>/dev/null && echo "tmux confirm: running" || echo "tmux confirm: ended"
