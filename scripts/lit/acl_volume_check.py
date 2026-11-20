"""Confirm ACL-family venues on ACL Anthology volume pages (the venue's own site) and fetch the official BibTeX.

For each key, the verified title (refs/raw/meta*.json) is searched in the paper lists of the given Anthology volumes;
a match yields the paper id, whose .bib is downloaded from aclanthology.org and stored with the record.
Updates refs/raw/venues.json (status CONFIRMED_ACL). Usage: python3 scripts/lit/acl_volume_check.py KEY [KEY ...]"""
from __future__ import annotations

import html
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fetch_meta import get  # noqa: E402

VOLUMES = ["2022.tacl-1", "2023.tacl-1", "2024.tacl-1", "2023.acl-long", "2024.acl-long", "2025.acl-long", "2023.emnlp-main",
           "2024.emnlp-main", "2023.findings-emnlp", "2024.findings-acl", "2024.findings-emnlp", "2024.naacl-long",
           "2024.findings-naacl", "2025.naacl-long", "2025.findings-acl", "2024.eacl-demo", "2026.findings-acl", "2026.acl-long"]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", html.unescape(s or "").lower()).strip()


def main() -> None:
    titles = {}
    for f in ("refs/raw/meta.json", "refs/raw/meta_phase7.json", "refs/raw/meta_phase7b.json"):
        for r in json.load(open(f))["records"]:
            titles.setdefault(r["key"], r["arxiv"]["title"])
    index: dict[str, str] = {}  # normalised title -> paper id
    for v in VOLUMES:
        try:
            page = get(f"https://aclanthology.org/volumes/{v}/", tries=3).decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            print(f"volume {v}: {str(e)[:80]}")
            continue
        n = 0
        for pid, t in re.findall(r'<strong><a class=align-middle href=/(\d{4}\.[a-z0-9\-]+\.\d+)/>(.+?)</a></strong>', page):
            index[norm(re.sub(r"<[^>]+>", "", t))] = pid
            n += 1
        print(f"volume {v}: {n} papers", flush=True)
        time.sleep(1)
    if sys.argv[1] == "--pid":  # KEY PID: title changed between versions -> accept only if the author lists match
        k, pid = sys.argv[2], sys.argv[3]
        rec = next(r for f in ("refs/raw/meta.json", "refs/raw/meta_phase7.json") for r in json.load(open(f))["records"] if r["key"] == k)
        bib = get(f"https://aclanthology.org/{pid}.bib", tries=3).decode("utf-8", "replace").strip()
        bib_auth = re.search(r'author\s*=\s*"(.+?)",', bib, re.S).group(1)
        surname = lambda n: norm(n.split(",")[0] if "," in n else n.split()[-1])  # noqa: E731
        a_bib = [surname(x) for x in re.split(r"\s+and\s+", " ".join(bib_auth.split()))]
        a_arx = [surname(x) for x in rec["arxiv"]["authors"]]
        print("anthology authors:", a_bib, "\narXiv authors:    ", a_arx)
        if a_bib == a_arx:
            venues = json.loads(Path("refs/raw/venues.json").read_text())
            book = re.search(r'(booktitle|journal)\s*=\s*"(.+?)",', bib, re.S)
            year = re.search(r'year\s*=\s*"(\d{4})"', bib)
            venues[k] = {**venues.get(k, {}), "status": "CONFIRMED", "venue": " ".join(book.group(2).split()), "year": year.group(1),
                         "source": f"https://aclanthology.org/{pid}", "bibtex": bib, "checked": "2026-11-18",
                         "how": "published title differs from arXiv title; identical ordered author surname list; official .bib fetched"}
            Path("refs/raw/venues.json").write_text(json.dumps(venues, indent=1, default=str))
            print(k, "CONFIRMED", pid)
        else:
            print(k, "author lists differ: NOT confirmed")
        return
    if sys.argv[1] == "--grep":  # list Anthology titles containing a word (for titles that changed between versions)
        for w in sys.argv[2:]:
            for t, pid in index.items():
                if w.lower() in t:
                    print(f"{w}: {pid}  {t}")
        return
    venues = json.loads(Path("refs/raw/venues.json").read_text())
    for k in sys.argv[1:]:
        pid = index.get(norm(titles[k]))
        if not pid:
            print(f"{k:22s} not found in the scanned volumes")
            continue
        bib = get(f"https://aclanthology.org/{pid}.bib", tries=3).decode("utf-8", "replace").strip()
        book = re.search(r'(booktitle|journal)\s*=\s*"(.+?)",', bib, re.S)
        year = re.search(r'year\s*=\s*"(\d{4})"', bib)
        venues[k] = {**venues.get(k, {}), "title": titles[k], "status": "CONFIRMED", "venue": " ".join(book.group(2).split()) if book else None,
                     "year": year.group(1) if year else None, "source": f"https://aclanthology.org/{pid}", "bibtex": bib, "checked": "2026-11-18",
                     "how": "title found on the ACL Anthology volume page; official .bib fetched"}
        print(f"{k:22s} CONFIRMED {pid}  {venues[k]['venue'][:70]}", flush=True)
        time.sleep(1)
    Path("refs/raw/venues.json").write_text(json.dumps(venues, indent=1, default=str))


if __name__ == "__main__":
    main()
