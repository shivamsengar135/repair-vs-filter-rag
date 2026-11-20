#!/usr/bin/env bash
# Test-set figures for all three generators (results/figures/test/).
cd "$(dirname "$0")/.."
for t in Qwen2.5-7B-Instruct Mistral-7B-Instruct-v0.3 Falcon3-7B-Instruct; do
  bash scripts/py.sh scripts/figures.py --split test --model-tag "$t" 2>&1 | tail -n 3
done
