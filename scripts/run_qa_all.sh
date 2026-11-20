#!/usr/bin/env bash
# Run one method on all five QA sets of a split. Usage: bash scripts/run_qa_all.sh METHOD SPLIT MODEL [extra args]
set -euo pipefail
cd "$(dirname "$0")/.."
METHOD="$1"; SPLIT="$2"; MODEL="$3"; shift 3
for d in popqa nq triviaqa hotpotqa 2wiki; do
  printf "%-9s " "$d"
  bash scripts/py.sh scripts/run_qa.py --dataset "$d" --split "$SPLIT" --method "$METHOD" --model "$MODEL" "$@" 2>&1 | grep -E '^\{"n"|Error|Traceback|assert' | cut -c1-200
done
