#!/usr/bin/env bash
# Phase 4: read-only checks of download sources (HTTP HEAD / small API calls). Nothing large is downloaded.
head_url() { printf "%s\n   " "$1"; curl -sIL -m 40 "$1" | grep -iE "^(HTTP/|content-length|content-type|x-linked-size|etag)" | tr -d '\r' | tr '\n' ' '; echo; }
echo "--- DPR Wikipedia passages"
head_url "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"
echo "--- HF dataset file listings"
for d in facebook/wiki_dpr akariasai/PopQA princeton-nlp/ALCE-data blue-guardrails/ragtruth-plus-plus xanhho/2WikiMultihopQA google-research-datasets/nq_open; do
  echo "## $d"
  curl -s -m 40 "https://huggingface.co/api/datasets/$d/tree/main?recursive=true" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
except Exception as e:
    print("  parse error", e); sys.exit()
if isinstance(d, dict):
    print("  ", d); sys.exit()
fs = [x for x in d if x.get("type") == "file"]
tot = sum(x.get("size", 0) for x in fs)
print("  %d files, %.2f GB" % (len(fs), tot / 1e9))
for x in sorted(fs, key=lambda x: -x.get("size", 0))[:8]:
    print("   %10.1f MB  %s" % (x.get("size", 0) / 1e6, x["path"]))
'
done
echo "--- FActScore repo: data download instructions"
curl -s -m 40 https://raw.githubusercontent.com/shmsw25/FActScore/main/README.md | grep -n -iE "download|drive.google|enwiki|\.db|labeled|unlabeled|license" | cut -c1-220 | head -n 25
echo "--- Self-RAG repo: eval data / retrieval files"
curl -s -m 40 https://raw.githubusercontent.com/AkariAsai/self-rag/main/README.md | grep -n -iE "drive.google|eval_data|popqa|download|wikipedia_embeddings|psgs_w100" | cut -c1-220 | head -n 20
