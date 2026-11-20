#!/usr/bin/env bash
# Dev runs of the long-form pipeline on both datasets (verifiers on CPU while vLLM holds the GPU).
# Usage: bash scripts/run_longform_dev.sh MODEL [extra args]   -> logs/phase4_longform_dev.log (ends with LONGFORM_DEV_END)
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="$1"; shift
for d in bio asqa; do
  echo "##### $d $(date -u +%FT%TZ)"
  bash scripts/py.sh scripts/run_longform.py --dataset "$d" --split dev --model "$MODEL" --verifier-device "${VERIFIER_DEVICE:-cpu}" "$@" 2>&1 | grep --line-buffered -vE 'Loading weights|warn|Warning|^\s*$' | cut -c1-900
done
echo LONGFORM_DEV_END
