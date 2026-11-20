"""Print abstracts (from saved arXiv API results) for a shortlist of IDs.
Usage: python3 scripts/lit/show_abstracts.py refs/raw/arxiv_search.json ID [ID ...]
"""
import json
import sys

data = json.load(open(sys.argv[1]))
by_id = {}
for r in data["results"]:
    for h in r["hits"]:
        by_id[h["id"].split("v")[0]] = h
for i in sys.argv[2:]:
    h = by_id.get(i)
    if not h:
        print(f"## {i} NOT IN SEARCH RESULTS\n")
        continue
    print(f"## {h['id']} | {h['published']} | {h['first_author']} | {h['title']}\n{h['abstract'][:1100]}\n")
