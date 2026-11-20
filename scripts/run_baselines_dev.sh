#!/usr/bin/env bash
# RARR-style and FLARE dev runs on both long-form datasets (after CoVe). Usage: bash scripts/run_baselines_dev.sh MODEL
set -uo pipefail
cd "$(dirname "$0")/.."
while ! grep -q COVE_DEV_END logs/phase4_cove_dev.log 2>/dev/null; do sleep 60; done
for d in bio asqa; do
  for s in run_rarr run_flare; do
    echo "##### $d $s $(date -u +%FT%TZ)"
    bash scripts/py.sh scripts/$s.py --dataset "$d" --split dev --model "$1" 2>&1 | grep -vE 'warn|Warning|^\s*$' | tail -n 3 | cut -c1-900
  done
done
echo BASELINES_DEV_END
