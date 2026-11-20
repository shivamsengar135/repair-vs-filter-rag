#!/usr/bin/env bash
# Phase 5: serve the judge and judge every long-form TEST run (prompts v2, no limit). Do NOT edit while running.
set -uo pipefail
cd "$(dirname "$0")/.."
JUDGE="RedHatAI/phi-4-FP8-dynamic"; JREV="8f62f4b85ccc1c7fda9f178976c1162f37e43651"
docker rm -f vllm > /dev/null 2>&1
GPU_UTIL=0.92 MAX_MODEL_LEN=4096 bash scripts/serve_vllm_docker.sh "$JUDGE" "$JREV" 2>&1 | tail -n 1
for d in asqa bio; do
  for run in results/test/$d/*__*/; do
    case "$run" in *longform__*|*cove-*|*rarr__*|*flare__*) ;; *) continue ;; esac
    grep -q '"prompts": "v2"' "$run/summary.json" 2>/dev/null || continue
    grep -q '"limit"' "$run/summary.json" && continue
    echo "##### $(date -u +%FT%TZ) judge $run"
    bash scripts/py.sh scripts/run_judge.py --run "$run" --judge-model "$JUDGE" 2>&1 | tail -n 1 | cut -c1-400
  done
done
docker rm -f vllm > /dev/null 2>&1
echo "##### JUDGE_STAGE_END"
