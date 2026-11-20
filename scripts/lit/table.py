"""Print one table (cells joined) from a saved full text, located by its caption 'Table N'.
Tables in arXiv HTML put the caption either after or before the cell block, so both blocks are tried.
Usage: python3 scripts/lit/table.py KEY:N[:MAXCHARS[:after]] ...
"""
import re
import sys
from pathlib import Path


def block(lines: list[str], i: int, step: int) -> list[str]:
    """Collect the contiguous cell block adjacent to line i in direction step (allowing short non-cell gaps)."""
    out, j, gap = [], i + step, 0
    while 0 <= j < len(lines) and gap <= 3:
        if lines[j].startswith("|") or (out and len(lines[j]) < 40 and not lines[j].startswith(("Table", "Figure", "##"))):
            out.append(lines[j] if len(lines[j]) > 2 else "|_")
            gap = 0
        elif out:
            break
        else:
            gap += 1
        j += step
    return out[::step]


for spec in sys.argv[1:]:
    p = spec.split(":")
    key, n = p[0], p[1]
    mx = int(p[2]) if len(p) > 2 else 1500
    prefer_after = len(p) > 3 and p[3] == "after"
    lines = Path(f"refs/raw/fulltext/{key}.txt").read_text().splitlines()
    idx = [i for i, l in enumerate(lines) if re.match(rf"Table {n}[:.]", l)]
    if not idx:
        print(f"==== {key} Table {n}: caption not found")
        continue
    i = idx[0]
    cap = lines[i] + (" " + lines[i + 1] if lines[i].rstrip().endswith(":") else "")
    before, after = block(lines, i, -1), block(lines, i, 1)
    b = after if (prefer_after or not before) else before
    print(f"==== {key} line {i + 1}: {cap[:300]}\n{' '.join(c[:50] for c in b)[:mx]}\n")
