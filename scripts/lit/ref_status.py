"""Compact listing of refs/VERIFIED_REFS.md: key, year, read status, title (for planning which papers to cite).
Usage: python scripts/lit/ref_status.py"""
import collections
import re
from pathlib import Path

rows = [l for l in Path("refs/VERIFIED_REFS.md").read_text(encoding="utf-8").splitlines() if re.match(r"^\| [A-Za-z0-9_\-]+ \|", l)]
status = collections.Counter()
for l in rows:
    c = [x.strip() for x in l.strip("|").split("|")]
    key, title, year, read = c[0], c[1], c[3], c[-1]
    s = "metadata-only" if read.upper().startswith("TITLE/METADATA") else ("abstract-only" if read.lower().startswith("abstract only") else "read")
    status[s] += 1
    print(f"{key:28s} {year:5s} {s:14s} {title[:90]}")
print(dict(status), "total", len(rows))
