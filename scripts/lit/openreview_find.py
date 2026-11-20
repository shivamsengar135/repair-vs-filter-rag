"""List OpenReview notes matching a search term with their venue field (to locate a paper's accepted version).
Usage: python3 scripts/lit/openreview_find.py "term" ["term" ...]"""
import json
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fetch_meta import get  # noqa: E402

for term in sys.argv[1:]:
    url = "https://api2.openreview.net/notes/search?" + urllib.parse.urlencode({"term": term, "type": "terms", "content": "all", "source": "forum", "limit": 10})
    res = json.loads(get(url, tries=3))
    print("==", term)
    for n in res.get("notes", []):
        c = n.get("content", {})
        val = lambda f: (c.get(f) or {}).get("value") if isinstance(c.get(f), dict) else c.get(f)  # noqa: E731
        print(f"   {n.get('forum')} | {str(val('title'))[:95]} | venue={val('venue')} | authors={str(val('authors'))[:80]}")
