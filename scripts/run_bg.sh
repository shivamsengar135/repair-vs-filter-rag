#!/usr/bin/env bash
# Run a project Python script detached in tmux with a log. Usage: bash scripts/run_bg.sh SESSION script.py [args]
# Log: logs/bg_<SESSION>.log ; ends with "BG_EXIT <code>".
set -euo pipefail
source "$(dirname "$0")/wsl_env.sh"
S="$1"; shift
LOG="$PROJECT_ROOT/logs/bg_$S.log"
tmux kill-session -t "$S" 2>/dev/null || true
printf -v CMD '%q ' "$@"
tmux new-session -d -s "$S" "cd $PROJECT_ROOT && echo \"=== start \$(date -u +%FT%TZ) $CMD\" >> $LOG; /usr/bin/time -v bash scripts/py.sh $CMD >> $LOG 2>&1; echo \"BG_EXIT \$? \$(date -u +%FT%TZ)\" >> $LOG"
echo "started tmux '$S'; log $LOG"
