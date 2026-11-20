#!/usr/bin/env bash
# Short-form analyses + QA tables for the generator-size sensitivity runs (3B, 14B-AWQ). Log: logs/phase6_sizes.log
cd "$(dirname "$0")/.."
for t in Qwen2.5-3B-Instruct Qwen2.5-14B-Instruct-AWQ; do
  echo "##### QA table $t"; bash scripts/py.sh scripts/qa_table.py --model-tag "$t"
  for d in popqa nq triviaqa hotpotqa 2wiki; do
    for run in results/test/$d/shortform__${t}__*/; do
      echo "##### analyze_shortform $run"; bash scripts/py.sh scripts/analyze_shortform.py --run "$run" 2>&1 | tail -n 5
    done
  done
done
echo SIZES_END
