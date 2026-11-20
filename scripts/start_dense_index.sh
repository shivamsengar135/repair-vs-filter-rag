#!/usr/bin/env bash
# Build the bge-small dense index in tmux session 'dense' (log logs/build_dense_index.log). Resumable.
cd "$(dirname "$0")/.."
docker rm -f vllm > /dev/null 2>&1
tmux has-session -t dense 2>/dev/null && { echo "dense already running"; exit 0; }
tmux new-session -d -s dense "cd $PWD && HF_HUB_OFFLINE=1 bash scripts/py.sh scripts/build_dense_index.py >> logs/build_dense_index.log 2>&1; echo DENSE_EXIT \$? >> logs/build_dense_index.log"
sleep 1; tmux ls
