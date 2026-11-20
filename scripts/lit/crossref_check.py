"""Verify DOIs against the Crossref API (registry of record for DOIs). Usage: crossref_check.py OUT.json DOI [DOI ...]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fetch_meta import get  # noqa: E402

out = {}
for doi in sys.argv[2:]:
    try:
        m = json.loads(get(f"https://api.crossref.org/works/{doi}", tries=3))["message"]
        rec = {
            "title": (m.get("title") or [""])[0],
            "authors": [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in m.get("author", [])],
            "container": (m.get("container-title") or [""])[0],
            "issued": m.get("issued", {}).get("date-parts"),
            "volume": m.get("volume"), "page": m.get("page"), "type": m.get("type"), "publisher": m.get("publisher"),
        }
        out[doi] = rec
        print(f"OK {doi}\n   {rec['title']}\n   {rec['authors'][:4]} (+{max(0, len(rec['authors']) - 4)}) | {rec['container']} | {rec['issued']} | vol {rec['volume']} p {rec['page']}")
    except Exception as e:  # noqa: BLE001
        out[doi] = {"error": str(e)[:200]}
        print(f"FAIL {doi}: {str(e)[:150]}")
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
