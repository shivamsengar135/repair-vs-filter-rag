#!/usr/bin/env bash
# Phase 5 analyses on the TEST runs (after phase5_all.sh). Do NOT edit while running. Log: logs/phase5_analyze.log
set -uo pipefail
cd "$(dirname "$0")/.."
JTAG="phi-4-FP8-dynamic"
for d in asqa bio; do
  for run in results/test/$d/longform__*/; do
    grep -q '"prompts": "v2"' "$run/summary.json" 2>/dev/null || continue
    [ -f "$run/judge__$JTAG.jsonl" ] || { echo "MISSING judge file for $run"; continue; }
    echo "##### analyze_longform $run"
    bash scripts/py.sh scripts/analyze_longform.py --run "$run" --judge-tag "$JTAG" 2>&1 | tail -n 32
  done
  echo "##### compare_longform $d test (G1 methods)"
  bash scripts/py.sh scripts/compare_longform.py --dataset "$d" --split test --judge-tag "$JTAG" 2>&1 | tail -n 30
done
echo "##### CONFIRMATORY"
bash scripts/py.sh scripts/confirmatory.py --judge-tag "$JTAG"
for d in popqa nq triviaqa hotpotqa 2wiki; do
  for run in results/test/$d/shortform__*/; do
    echo "##### analyze_shortform $run"
    bash scripts/py.sh scripts/analyze_shortform.py --run "$run" 2>&1 | tail -n 5
  done
done
echo PHASE5_ANALYZE_END
