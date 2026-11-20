#!/usr/bin/env bash
# Phase 5, GPU scoring stage (no server): fill the verifier score cache for the long-form test runs of the given
# generators, on the GPU. Do NOT edit while running. Usage: bash scripts/phase5_score_stage.sh MODEL [MODEL ...]
set -uo pipefail
cd "$(dirname "$0")/.."
docker rm -f vllm > /dev/null 2>&1
for M in "$@"; do
  for d in asqa bio; do
    echo "##### $(date -u +%FT%TZ) verifier scores $M $d test (GPU)"
    HF_HUB_OFFLINE=1 bash scripts/py.sh scripts/run_longform.py --dataset "$d" --split test --model "$M" --stop-after score --verifier-device cuda 2>&1 \
      | grep -E 'STOP_AFTER|Error|Traceback|LLM request failed' | cut -c1-300
  done
done
echo "##### SCORE_STAGE_END"
