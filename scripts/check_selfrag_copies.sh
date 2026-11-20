#!/usr/bin/env bash
# Read-only: list files of third-party copies of the Self-RAG eval data, and what the paper says about PopQA eval.
cd "$(dirname "$0")/.."
for d in jesse-tong/selfrag-eval-data awinml/popqa_longtail_w_gs; do
  echo "## $d"
  curl -s -m 40 "https://huggingface.co/api/datasets/$d/tree/main?recursive=true" | python3 -c '
import sys, json
d = json.load(sys.stdin)
for x in d:
    if x.get("type") == "file":
        print("   %10.2f MB  %s" % (x.get("size", 0) / 1e6, x["path"]))
'
done
echo "## Self-RAG paper on PopQA evaluation"
python3 scripts/lit/grep_text.py selfrag '(1,399|1399|long-tail|monthly|PopQA)' 0 0 8 300
