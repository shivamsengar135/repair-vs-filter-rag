#!/usr/bin/env bash
# Read-only: save the official Self-RAG short-form inference code and print its key parts + README commands.
set -euo pipefail
D=/mnt/c/Users/DIAT/Documents/shivam/trial_1_Q1/refs/raw/code; mkdir -p "$D"
B=https://raw.githubusercontent.com/AkariAsai/self-rag/main
curl -s -m 60 "$B/retrieval_lm/run_short_form.py" -o "$D/selfrag_run_short_form.py"
curl -s -m 60 "$B/retrieval_lm/utils.py" -o "$D/selfrag_utils.py"
curl -s -m 60 "$B/retrieval_lm/metrics.py" -o "$D/selfrag_metrics.py"
curl -s -m 60 "$B/README.md" -o "$D/selfrag_README.md"
wc -l "$D"/selfrag_*
echo "--- README: PopQA command"
grep -n -A 6 'popqa_longtail_w_gs' "$D/selfrag_README.md" | head -n 14
