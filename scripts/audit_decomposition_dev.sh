#!/usr/bin/env bash
# Run the decomposition audit on the long-form dev runs of G1.
cd "$(dirname "$0")/.."
for r in results/dev/asqa/longform__Qwen2.5-7B-Instruct__5cdadc19a73e results/dev/bio/longform__Qwen2.5-7B-Instruct__6a6082d9435e; do
  echo "## $r"
  bash scripts/py.sh scripts/audit_decomposition.py "$r/items.jsonl" 4 2>&1 | cut -c1-330
done
