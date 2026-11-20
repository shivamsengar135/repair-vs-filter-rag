#!/usr/bin/env bash
# Show model-download progress: cache sizes, running hf processes, network rate.
du -sh ~/.cache/huggingface/* 2>/dev/null
find ~/.cache/huggingface -type f -size +50M -printf "%s %p\n" 2>/dev/null | awk '{printf "%.2f GB %s\n", $1/1e9, $2}'
echo "--- procs"; pgrep -af "hf download|vllm" || echo "no hf/vllm process"
echo "--- net rate (5s)"
r1=$(cat /sys/class/net/eth0/statistics/rx_bytes); sleep 5; r2=$(cat /sys/class/net/eth0/statistics/rx_bytes)
echo "$(( (r2 - r1) / 5 / 1024 / 1024 )) MiB/s"
date
