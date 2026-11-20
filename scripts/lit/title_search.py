"""Search arXiv by title words (no date filter) to find a preprint of a known paper. Usage: title_search.py "words" ..."""
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fetch_meta import ATOM, get  # noqa: E402

for q in sys.argv[1:]:
    query = " AND ".join(f'ti:"{w}"' for w in q.split())
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": query, "max_results": 5})
    root = ET.fromstring(get(url))
    print(f"### {query}")
    for e in root.findall(f"{ATOM}entry"):
        print("  ", (e.findtext(f"{ATOM}id") or "").rsplit("/abs/", 1)[-1], (e.findtext(f"{ATOM}published") or "")[:10],
              "|", " ".join((e.findtext(f"{ATOM}title") or "").split())[:120],
              "|", [a.findtext(f"{ATOM}name") for a in e.findall(f"{ATOM}author")][:3])
    if not root.findall(f"{ATOM}entry"):
        print("   no arXiv hit")
