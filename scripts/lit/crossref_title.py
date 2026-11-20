"""Search Crossref by bibliographic text (for works whose DOI is unknown). Prints the top hits; nothing is trusted
until the title and authors match. Usage: crossref_title.py OUT.json "query text" ["query" ...]"""
import json
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fetch_meta import get  # noqa: E402

out = {}
for q in sys.argv[2:]:
    url = "https://api.crossref.org/works?rows=5&query.bibliographic=" + urllib.parse.quote(q)
    items = json.loads(get(url, tries=3))["message"]["items"]
    out[q] = []
    print(f"== {q}")
    for m in items:
        rec = {"doi": m.get("DOI"), "title": (m.get("title") or [""])[0],
               "authors": [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in m.get("author", [])],
               "container": (m.get("container-title") or [""])[0], "issued": m.get("issued", {}).get("date-parts"),
               "volume": m.get("volume"), "page": m.get("page")}
        out[q].append(rec)
        print(f"   {rec['doi']} | {rec['title'][:90]} | {rec['authors'][:3]} | {rec['container'][:50]} | {rec['issued']} | {rec['volume']} {rec['page']}")
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
