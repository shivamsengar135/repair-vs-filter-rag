#!/usr/bin/env bash
# Phase 5, finishing stage for ONE generator's long-form runs (server must serve MODEL; verifier scores already cached
# by the GPU scoring stage). Then, for TIER 1, SelfCheckGPT samples for 3 seed sets. Do NOT edit while running.
# Usage: bash scripts/phase5_finish_stage.sh MODEL TIER
set -uo pipefail
cd "$(dirname "$0")/.."
M="$1"; TIER="$2"; TAG="${M##*/}"
for d in asqa bio; do
  echo "##### $(date -u +%FT%TZ) long-form pipeline $d test: full run"
  bash scripts/py.sh scripts/run_longform.py --dataset "$d" --split test --model "$M" --verifier-device cpu 2>&1 | grep -E '^\{|^wrote|Error|Traceback' | cut -c1-900
  if [ "$TIER" = 1 ]; then
    run=$(ls -td results/test/$d/longform__${TAG}__*/ | head -n 1)
    for k in 0 1 2; do
      echo "##### SelfCheckGPT samples $d seed set $k"
      bash scripts/py.sh scripts/run_selfcheck.py --run "$run" --stage sample --seed-set "$k" 2>&1 | grep -E '^samples|Error|Traceback' | cut -c1-300
    done
  fi
done
echo "##### FINISH_STAGE_END $M"
