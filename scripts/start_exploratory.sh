#!/usr/bin/env bash
# Run the exploratory tests in tmux session 'expl' (log logs/phase6_exploratory.log).
cd "$(dirname "$0")/.."
tmux has-session -t expl 2>/dev/null && { echo "expl already running"; exit 0; }
tmux new-session -d -s expl "cd $PWD && bash scripts/py.sh scripts/exploratory_tests.py > logs/phase6_exploratory.log 2>&1; echo EXPL_EXIT \$? >> logs/phase6_exploratory.log"
sleep 1; tmux ls
