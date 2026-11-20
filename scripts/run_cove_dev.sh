#!/usr/bin/env bash
# CoVe dev runs: both variants on both long-form datasets. Usage: bash scripts/run_cove_dev.sh MODEL
set -uo pipefail
cd "$(dirname "$0")/.."
for d in bio asqa; do for v in closed_book rag_verify; do
  echo "##### $d $v"
  bash scripts/py.sh scripts/run_cove.py --dataset "$d" --split dev --variant "$v" --model "$1" 2>&1 | grep -vE 'warn|Warning|^\s*$' | tail -n 3 | cut -c1-900
done; done
echo COVE_DEV_END
