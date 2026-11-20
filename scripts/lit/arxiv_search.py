"""Phase 1: search arXiv for recent work (default: submitted 2025-01-01 onwards).

Each query is logged with its exact string and every hit is saved, so the search is reproducible.
Usage: python3 scripts/lit/arxiv_search.py refs/search_queries.txt refs/raw/arxiv_search.json [max_per_query]
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fetch_meta import ATOM, get  # noqa: E402

import os

# Default: 2025 onwards. Set ARXIV_DATE=none to search all years (used for Phase 2 novelty checks).
DATE = os.environ.get("ARXIV_DATE", "submittedDate:[202501010000 TO 202612312359]")


def search(q: str, n: int) -> list[dict]:
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "search_query": f"({q}) AND {DATE}" if DATE != "none" else q, "max_results": n,
        "sortBy": "relevance", "sortOrder": "descending"})
    root = ET.fromstring(get(url))
    hits = []
    for e in root.findall(f"{ATOM}entry"):
        hits.append({
            "id": (e.findtext(f"{ATOM}id") or "").rsplit("/abs/", 1)[-1],
            "published": (e.findtext(f"{ATOM}published") or "")[:10],
            "title": " ".join((e.findtext(f"{ATOM}title") or "").split()),
            "first_author": (e.find(f"{ATOM}author").findtext(f"{ATOM}name") if e.find(f"{ATOM}author") is not None else ""),
            "abstract": " ".join((e.findtext(f"{ATOM}summary") or "").split()),
        })
    return hits


def main() -> None:
    queries = [l.strip() for l in Path(sys.argv[1]).read_text().splitlines() if l.strip() and not l.startswith("#")]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    out, seen = [], set()
    for q in queries:
        hits = search(q, n)
        out.append({"query": q, "date_filter": DATE, "n_hits": len(hits), "hits": hits})
        print(f"\n### {q}  ({len(hits)} hits)")
        for h in hits:
            mark = " " if h["id"].split("v")[0] in seen else "*"
            seen.add(h["id"].split("v")[0])
            print(f"{mark} {h['id']:<13} {h['published']} {h['first_author'][:16]:<16} {h['title'][:110]}")
        time.sleep(3.5)
    Path(sys.argv[2]).write_text(json.dumps({"searched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "results": out}, indent=1))
    print(f"\n{len(seen)} unique papers; wrote {sys.argv[2]}  (* = first time seen)")


if __name__ == "__main__":
    main()
