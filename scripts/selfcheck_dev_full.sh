#!/usr/bin/env bash
# SelfCheckGPT for the latest prompts-v2 long-form dev runs: sample stage on the Qwen server, then score on GPU.
# Do NOT edit while running.
set -uo pipefail
cd "$(dirname "$0")/.."
docker rm -f vllm > /dev/null 2>&1
bash scripts/serve_vllm_docker.sh 2>&1 | tail -n 1
runs=()
for ds in asqa bio; do
  runs+=("$(for d in results/dev/$ds/longform__*/; do grep -q '"prompts": "v2"' "$d/summary.json" 2>/dev/null && ! grep -q '"limit"' "$d/summary.json" && echo "$d"; done | xargs -r ls -td | head -n 1)")
done
for run in "${runs[@]}"; do
  echo "##### sample $run"
  bash scripts/py.sh scripts/run_selfcheck.py --run "$run" --stage sample 2>&1 | grep -E '^samples|Error|Traceback' | cut -c1-400
done
docker rm -f vllm > /dev/null 2>&1
for run in "${runs[@]}"; do
  echo "##### score $run"
  bash scripts/py.sh scripts/run_selfcheck.py --run "$run" --stage score --device cuda 2>&1 | grep -E '^\{|Error|Traceback' | cut -c1-400
done
echo SELFCHECK_DEV_END
