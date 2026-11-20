"""Diagnose retrieval on a dev run: show misses with retrieved titles, and recall by PopQA relation type.
Usage: python scripts/inspect_retrieval.py results/dev/popqa/rag__.../items.jsonl [n_examples]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from hallu.store import JsonlStore, read_jsonl

items = read_jsonl(sys.argv[1])
n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 8
cache = {r["query"]: r for r in JsonlStore(Path.home() / "halluc_data/cache/retrieval__bm25_psgs_w100.jsonl")}
split = {r["id"]: r for r in read_jsonl(f"data/splits/{Path(sys.argv[1]).parts[2]}_dev.jsonl")}

by_prop = defaultdict(list)
title_hit = []
for it in items:
    meta = split[it["id"]].get("meta", {})
    hits = cache[it["question"]]["hits"]
    by_prop[meta.get("prop", "-")].append(it["gold_in_passages"])
    if meta.get("subj"):
        title_hit.append(any(h["title"].lower() == meta["subj"].lower() for h in hits))
print("recall@5 by relation:", {k: f"{sum(v) / len(v):.2f} (n={len(v)})" for k, v in sorted(by_prop.items())})
if title_hit:
    print(f"subject's own page among top-5 titles: {sum(title_hit) / len(title_hit):.3f}")
shown = 0
for it in items:
    if it["gold_in_passages"] == 0.0 and shown < n_show:
        hits = cache[it["question"]]["hits"]
        print(f"\nQ: {it['question']}  gold={it['answers'][:3]}  pred={it['prediction']!r}")
        print("   titles:", [h["title"] for h in hits])
        shown += 1
