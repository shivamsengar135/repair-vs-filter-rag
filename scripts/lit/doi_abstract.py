"""Fetch abstract + open-access links for a DOI from OpenAlex and Semantic Scholar (aggregators).
Usage: doi_abstract.py OUT.json DOI [DOI ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fetch_meta import get  # noqa: E402

out = {}
for doi in sys.argv[2:]:
    rec = {}
    try:
        m = json.loads(get(f"https://api.openalex.org/works/https://doi.org/{doi}", tries=2))
        inv = m.get("abstract_inverted_index") or {}
        words = sorted((p, w) for w, ps in inv.items() for p in ps)
        rec["openalex"] = {"title": m.get("title"), "abstract": " ".join(w for _, w in words),
                           "oa": m.get("open_access"), "locations": [l.get("landing_page_url") for l in m.get("locations", [])]}
    except Exception as e:  # noqa: BLE001
        rec["openalex"] = {"error": str(e)[:150]}
    try:
        s = json.loads(get(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=title,abstract,openAccessPdf,externalIds,tldr", tries=2))
        rec["s2"] = s
    except Exception as e:  # noqa: BLE001
        rec["s2"] = {"error": str(e)[:150]}
    out[doi] = rec
    print(json.dumps(rec, indent=1)[:3500])
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
