#!/usr/bin/env bash
# Run the long-form GPU score stage for one model/dataset with the full traceback (debugging the score stage).
cd "$(dirname "$0")/.."
HF_HUB_OFFLINE=1 bash scripts/py.sh scripts/run_longform.py --dataset "${2:-asqa}" --split test --model "${1:-Qwen/Qwen2.5-7B-Instruct}" \
  --stop-after score --verifier-device cuda 2>&1 | grep -vE 'it/s\]|s/it\]' | tail -n 45
