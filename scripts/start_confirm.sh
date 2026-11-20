#!/usr/bin/env bash
# Start scripts/confirm_runs.sh detached in tmux session "confirm"; log logs/confirm_runs.log (ends with CONFIRM_EXIT).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/wsl_env.sh
tmux kill-session -t confirm 2>/dev/null || true
tmux new-session -d -s confirm "cd $PROJECT_ROOT && bash scripts/confirm_runs.sh >> logs/confirm_runs.log 2>&1; echo \"CONFIRM_EXIT \$? \$(date -u +%FT%TZ)\" >> logs/confirm_runs.log"
sleep 5; tail -n 3 logs/confirm_runs.log
