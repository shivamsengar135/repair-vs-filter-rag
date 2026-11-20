#!/usr/bin/env bash
# SelfCheckGPT-NLI scores (GPU) for the latest prompts-v2 long-form dev runs. Samples must already be cached.
# Do NOT edit while running.
set -uo pipefail
cd "$(dirname "$0")/.."
for ds in asqa bio; do
  run=$(for d in results/dev/$ds/longform__*/; do grep -q '"prompts": "v2"' "$d/summary.json" 2>/dev/null && ! grep -q '"limit"' "$d/summary.json" && echo "$d"; done | xargs -r ls -td | head -n 1)
  echo "##### $ds $run"
  bash scripts/py.sh scripts/run_selfcheck.py --run "$run" --stage score --device cuda 2>&1 | grep -E '^samples|^\{|Error|Traceback' | cut -c1-400
done
echo SELFCHECK_DEV_END
