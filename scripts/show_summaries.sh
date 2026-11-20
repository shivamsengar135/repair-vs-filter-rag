#!/usr/bin/env bash
# Print the summary.json files matching a glob under results/. Usage: bash scripts/show_summaries.sh 'test/*/shortform__*'
cd "$(dirname "$0")/.."
for f in results/$1/summary.json; do
  echo "== $f"
  head -c "${2:-1500}" "$f"; echo
done
