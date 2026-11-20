#!/usr/bin/env bash
# Start the Phase 5 analyses in tmux session p5a (log: logs/phase5_analyze.log).
cd "$(dirname "$0")/.."
tmux has-session -t p5a 2>/dev/null && { echo "p5a already running"; exit 0; }
tmux new-session -d -s p5a "cd $PWD && bash scripts/phase5_analyze.sh > logs/phase5_analyze.log 2>&1"
sleep 1; tmux ls
