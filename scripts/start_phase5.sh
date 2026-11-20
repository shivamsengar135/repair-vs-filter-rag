#!/usr/bin/env bash
# (Re)start the Phase 5 master pipeline in tmux session p5, appending to logs/phase5_all.log. Safe to rerun (cached).
cd "$(dirname "$0")/.."
if tmux has-session -t p5 2>/dev/null; then echo "p5 already running"; exit 0; fi
echo "===== $(date -u +%FT%TZ) RESTART phase5_all.sh (after WSL restart)" >> logs/phase5_all.log
tmux new-session -d -s p5 "cd $PWD && bash scripts/phase5_all.sh >> logs/phase5_all.log 2>&1"
sleep 2; tmux ls
