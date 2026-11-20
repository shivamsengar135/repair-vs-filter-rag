#!/usr/bin/env bash
# Read-only: save FActScore's retrieval + scorer code and print the parts that define the knowledge-base schema
# and how evidence passages are chosen for each atomic fact.
set -euo pipefail
D=/mnt/c/Users/DIAT/Documents/shivam/trial_1_Q1/refs/raw/code; mkdir -p "$D"
B=https://raw.githubusercontent.com/shmsw25/FActScore/main/factscore
curl -s -m 60 "$B/retrieval.py" -o "$D/factscore_retrieval.py"
curl -s -m 60 "$B/factscorer.py" -o "$D/factscore_factscorer.py"
wc -l "$D"/factscore_*.py
echo "--- retrieval.py: schema, passage construction, retrieval"
grep -nE 'CREATE TABLE|SELECT|SPECIAL_SEPARATOR|MAX_LENGTH|def |bm25|BM25|gtr|k=|passages|retrieval_type|\.split' "$D/factscore_retrieval.py" | cut -c1-170 | head -n 60
echo "--- factscorer.py: evidence + prompt for each atomic fact"
grep -nE 'get_passages|retrieval|definition|True or False|Answer the question|knowledge_source|n=|k=|reversed' "$D/factscore_factscorer.py" | cut -c1-200 | head -n 40
