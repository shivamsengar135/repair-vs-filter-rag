#!/usr/bin/env bash
# Phase 5, generator stage for ONE generator (server must already serve it). Do NOT edit while running.
# Usage: bash scripts/phase5_gen_stage.sh MODEL TIER   (TIER 1 = all methods, 2 = no-retrieval, RAG, pipelines)
# Steps are idempotent (cached); a failed step is logged and the script continues.
set -uo pipefail
cd "$(dirname "$0")/.."
M="$1"; TIER="$2"
step() { echo "##### $(date -u +%FT%TZ) $*"; }
step "QA closed-book + standard RAG (5 test sets)"
bash scripts/run_qa_all.sh closed_book test "$M"
bash scripts/run_qa_all.sh rag test "$M"
if [ "$TIER" = 1 ]; then
  step "serving non-determinism: PopQA test standard RAG x3 with fresh caches"
  for r in rep1 rep2 rep3; do
    bash scripts/py.sh scripts/run_qa.py --dataset popqa --split test --method rag --model "$M" --cache-tag "$r" 2>&1 | grep -E '^\{"n"|Error' | cut -c1-300
  done
fi
for d in popqa nq triviaqa hotpotqa 2wiki; do
  step "short-form pipeline $d test"
  # verifier scores are not cached in run_shortform.py: skip a finished run instead of recomputing it on CPU
  if ls results/test/"$d"/shortform__"${M##*/}"__*/summary.json > /dev/null 2>&1; then echo "SKIP finished"; continue; fi
  bash scripts/py.sh scripts/run_shortform.py --dataset "$d" --split test --model "$M" --verifier-device cpu 2>&1 | grep -E '^\{|^wrote|Error|Traceback' | cut -c1-500
done
for d in asqa bio; do
  step "long-form pipeline $d test: generate + decompose"
  bash scripts/py.sh scripts/run_longform.py --dataset "$d" --split test --model "$M" --stop-after decompose 2>&1 | grep -E 'STOP_AFTER|Error|Traceback' | cut -c1-300
done
if [ "$TIER" = 1 ]; then
  for d in asqa bio; do
    for v in closed_book rag_verify; do
      step "CoVe-$v $d test"
      bash scripts/py.sh scripts/run_cove.py --dataset "$d" --split test --variant "$v" --model "$M" 2>&1 | grep -E '^\{|Error|Traceback' | cut -c1-400
    done
    for s in run_rarr run_flare; do
      step "$s $d test"
      bash scripts/py.sh scripts/$s.py --dataset "$d" --split test --model "$M" 2>&1 | grep -E '^\{|Error|Traceback' | cut -c1-400
    done
  done
fi
step "GEN_STAGE_END $M"
