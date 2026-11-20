"""Confirm publication venues on the venue's own page / registry (Phase 7, before BibTeX is written).

Hints come from the Semantic Scholar record stored in refs/raw/meta*.json (aggregator — never trusted alone).
Confirmation sources, by hint:
  ACL Anthology id   -> https://aclanthology.org/<id>.bib (the venue's own BibTeX); title must match
  non-arXiv DOI      -> Crossref record (registry of record); title must match
  NeurIPS            -> proceedings index https://papers.nips.cc/paper_files/paper/<year>; title must appear
  anything else      -> OpenReview API title search; the note's `venue` field must be a non-submission venue
Anything not confirmed is reported as UNCONFIRMED and must be cited as an arXiv preprint.
(DBLP was tried first; it served an anti-bot page on 2026-11-18 and is not used.)
Output: refs/raw/venues.json + printed table. Stdlib only. Usage: python3 scripts/lit/venue_check.py KEY [KEY ...] | --all"""
from __future__ import annotations

import html
import json
import re
import sys
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fetch_meta import get  # noqa: E402

_cache: dict[str, str] = {}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", html.unescape(s or "").lower()).strip()


def fetch(url: str) -> str:
    if url not in _cache:
        _cache[url] = get(url, tries=3).decode("utf-8", "replace")
        time.sleep(1)
    return _cache[url]


def records() -> dict:
    t = {}
    for f in ("refs/raw/meta.json", "refs/raw/meta_phase7.json", "refs/raw/meta_phase7b.json"):
        if Path(f).exists():
            for r in json.load(open(f))["records"]:
                t.setdefault(r["key"], r)
    return t


NOT_A_VENUE = re.compile(r"\bCoRR\b|\bARR\b|submi|withdrawn|rejected|desk", re.I)


def check(rec: dict) -> dict:
    title = rec["arxiv"]["title"]
    s2 = rec.get("s2") or {}
    try:  # fresh Semantic Scholar record: external ids (ACL Anthology id, DOI) are added after publication
        fresh = json.loads(fetch(f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{rec['arxiv_id']}?fields=externalIds,venue,year"))
        s2 = {**s2, **{k: v for k, v in fresh.items() if v}}
    except Exception:  # noqa: BLE001
        pass
    ext = s2.get("externalIds") or {}
    out = {"title": title, "s2_venue": s2.get("venue"), "s2_year": s2.get("year"), "status": "UNCONFIRMED"}
    if ext.get("ACL"):
        bib = fetch(f"https://aclanthology.org/{ext['ACL']}.bib")
        m = re.search(r'title\s*=\s*"(.+?)",', bib, re.S)
        if m and norm(m.group(1)) == norm(title):
            book = re.search(r'(booktitle|journal)\s*=\s*"(.+?)",', bib, re.S)
            year = re.search(r'year\s*=\s*"(\d{4})"', bib)
            return {**out, "status": "CONFIRMED", "venue": " ".join(book.group(2).split()) if book else None,
                    "year": year.group(1) if year else None, "source": f"https://aclanthology.org/{ext['ACL']}", "bibtex": bib.strip()}
        out["note"] = f"ACL id {ext['ACL']} title mismatch"
    doi = ext.get("DOI", "")
    if doi and "arxiv" not in doi.lower():
        m = json.loads(fetch(f"https://api.crossref.org/works/{doi}"))["message"]
        if norm((m.get("title") or [""])[0]) == norm(title):
            return {**out, "status": "CONFIRMED", "venue": (m.get("container-title") or [""])[0], "year": (m.get("issued", {}).get("date-parts") or [[None]])[0][0],
                    "volume": m.get("volume"), "pages": m.get("page"), "doi": doi, "source": f"https://doi.org/{doi}"}
        out["note"] = f"DOI {doi} title mismatch"
    if "neural information processing" in norm(s2.get("venue")) and s2.get("year"):
        for y in (s2["year"], s2["year"] + 1):
            idx = fetch(f"https://papers.nips.cc/paper_files/paper/{y}")
            if norm(title) in norm(re.sub(r"<[^>]+>", " ", idx)):
                return {**out, "status": "CONFIRMED", "venue": f"Advances in Neural Information Processing Systems {y}", "year": y,
                        "source": f"https://papers.nips.cc/paper_files/paper/{y}"}
    try:
        res = json.loads(fetch("https://api2.openreview.net/notes/search?" + urllib.parse.urlencode(
            {"term": title, "type": "terms", "content": "title", "source": "forum", "limit": 10})))
        found = []
        for n in res.get("notes", []):
            c = n.get("content", {})
            t = c.get("title", {}).get("value") if isinstance(c.get("title"), dict) else c.get("title")
            v = c.get("venue", {}).get("value") if isinstance(c.get("venue"), dict) else c.get("venue")
            if norm(t) == norm(title) and v and not NOT_A_VENUE.search(v):
                found.append((bool(re.search(r"workshop", v, re.I)), v, n.get("forum") or n.get("id")))
        if found:
            ws, v, fid = sorted(found)[0]  # a main venue before a workshop
            return {**out, "status": "CONFIRMED", "venue": v, "workshop": ws, "source": f"https://openreview.net/forum?id={fid}"}
        out["openreview_rejected_venues"] = [
            (n.get("content", {}).get("venue", {}) or {}).get("value") for n in res.get("notes", []) if isinstance(n.get("content", {}).get("venue"), dict)][:5]
    except Exception as e:  # noqa: BLE001
        out["openreview_error"] = str(e)[:120]
    return out


def main() -> None:
    recs = records()
    keys = list(recs) if sys.argv[1:] == ["--all"] else sys.argv[1:]
    out_path = Path("refs/raw/venues.json")
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    for k in keys:
        try:
            r = check(recs[k])
        except Exception as e:  # noqa: BLE001
            r = {"title": recs[k]["arxiv"]["title"], "status": "ERROR", "error": str(e)[:200]}
        r["checked"] = "2026-11-18"
        out[k] = r
        out_path.write_text(json.dumps(out, indent=1, default=str))
        print(f"{k:24s} {r['status']:11s} {str(r.get('venue', ''))[:60]:60s} {r.get('year', '')} {r.get('source', r.get('note', r.get('error', '')))}", flush=True)


if __name__ == "__main__":
    main()
