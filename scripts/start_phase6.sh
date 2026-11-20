#!/usr/bin/env bash
# (Re)start the Phase 6 run orchestration in tmux session p6, appending to logs/phase6_runs.log. Safe to rerun.
cd "$(dirname "$0")/.."
tmux has-session -t p6 2>/dev/null && { echo "p6 already running"; exit 0; }
echo "===== $(date -u +%FT%TZ) START phase6_runs.sh" >> logs/phase6_runs.log
tmux new-session -d -s p6 "cd $PWD && bash scripts/phase6_runs.sh >> logs/phase6_runs.log 2>&1"
sleep 2; tmux ls
