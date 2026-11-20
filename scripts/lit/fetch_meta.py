"""Phase 1: verify candidate papers against primary metadata sources.

For each candidate arXiv ID (candidates come from memory or search and are NOT trusted),
fetch the record from the arXiv API (primary source for title/authors/abstract/date) and
from the Semantic Scholar Graph API (aggregator: venue, year, DOI, citation count).
A candidate is only "verified" if the arXiv title contains the expected keyword(s).

Output: refs/raw/meta.json  (full records)  and a printed summary table.
Stdlib only, so it runs with the system python3.

Usage: python3 scripts/lit/fetch_meta.py refs/candidates.tsv refs/raw/meta.json
candidates.tsv columns: arxiv_id <TAB> key <TAB> expected title keyword(s), '|' separated (all must match, case-insensitive)
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
UA = {"User-Agent": "lit-review-metadata-check/0.1 (research script)"}


def get(url: str, data: bytes | None = None, headers: dict | None = None, tries: int = 6) -> bytes:
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={**UA, **(headers or {})})
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"failed after {tries} tries: {url}: {last}")


def arxiv_records(ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(ids), 40):
        chunk = ids[i:i + 40]
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"id_list": ",".join(chunk), "max_results": len(chunk)})
        root = ET.fromstring(get(url))
        for e in root.findall(f"{ATOM}entry"):
            full = (e.findtext(f"{ATOM}id") or "").rsplit("/abs/", 1)[-1]
            base = full.split("v")[0]
            out[base] = {
                "arxiv_id_versioned": full,
                "title": " ".join((e.findtext(f"{ATOM}title") or "").split()),
                "authors": [a.findtext(f"{ATOM}name") for a in e.findall(f"{ATOM}author")],
                "published": e.findtext(f"{ATOM}published"),
                "updated": e.findtext(f"{ATOM}updated"),
                "abstract": " ".join((e.findtext(f"{ATOM}summary") or "").split()),
                "journal_ref": e.findtext(f"{ARXIV}journal_ref"),
                "doi": e.findtext(f"{ARXIV}doi"),
                "comment": e.findtext(f"{ARXIV}comment"),
                "primary_category": (e.find(f"{ARXIV}primary_category").attrib.get("term")
                                     if e.find(f"{ARXIV}primary_category") is not None else None),
            }
        time.sleep(3.5)  # arXiv API etiquette
    return out


def s2_records(ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    fields = "title,year,venue,publicationVenue,externalIds,citationCount,publicationDate"
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        try:
            raw = get("https://api.semanticscholar.org/graph/v1/paper/batch?fields=" + fields,
                      data=json.dumps({"ids": [f"ARXIV:{x}" for x in chunk]}).encode(),
                      headers={"Content-Type": "application/json"})
            for x, rec in zip(chunk, json.loads(raw)):
                if rec:
                    out[x] = rec
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: Semantic Scholar lookup failed ({e}); venue fields will be empty", file=sys.stderr)
        time.sleep(3)
    return out


def main() -> None:
    cand_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    cands = []
    for line in cand_path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        aid, key, kw = (line.split("\t") + ["", ""])[:3]
        cands.append({"arxiv_id": aid.strip(), "key": key.strip(), "expect": [k.strip().lower() for k in kw.split("|") if k.strip()]})
    ids = [c["arxiv_id"] for c in cands]
    ax = arxiv_records(ids)
    s2 = s2_records(ids)
    results = []
    for c in cands:
        a = ax.get(c["arxiv_id"])
        s = s2.get(c["arxiv_id"]) or {}
        ok = bool(a) and all(k in a["title"].lower() for k in c["expect"])
        results.append({**c, "verified_title_match": ok, "arxiv": a, "s2": s})
        venue = (s.get("venue") or "")[:38]
        print(f"{'OK  ' if ok else 'FAIL'} {c['arxiv_id']:<11} {c['key']:<18} "
              f"{(a['published'][:10] if a else '-'):<10} {(a['authors'][0] if a else '-')[:18]:<18} "
              f"n_auth={len(a['authors']) if a else 0:<3} S2[{s.get('year')} | {venue} | cites={s.get('citationCount')}] "
              f"{(a['title'] if a else 'NOT FOUND ON ARXIV')[:90]}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "records": results}, indent=1))
    print(f"\n{sum(r['verified_title_match'] for r in results)}/{len(results)} title-verified; wrote {out_path}")


if __name__ == "__main__":
    main()
