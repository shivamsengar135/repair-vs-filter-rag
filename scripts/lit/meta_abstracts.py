"""Print the arXiv abstracts stored in a fetch_meta.py output for the given keys (to read them, then record it).
Usage: python3 scripts/lit/meta_abstracts.py refs/raw/meta.json KEY [KEY ...]"""
import json
import sys
import textwrap

recs = {r["key"]: r for r in json.load(open(sys.argv[1]))["records"]}
for k in sys.argv[2:]:
    a = recs[k]["arxiv"]
    print(f"=== {k}: {a['title']} ({a['published'][:4]})")
    print(textwrap.fill(" ".join(a.get("abstract", a.get("summary", "")).split()), 150))
