"""Print lines of a saved full text that match a regex, with line numbers and optional context.
Usage: python3 scripts/lit/grep_text.py KEY REGEX [before] [after] [max_matches] [width]
"""
import re
import sys
from pathlib import Path

key, pat = sys.argv[1], re.compile(sys.argv[2])
before = int(sys.argv[3]) if len(sys.argv) > 3 else 0
after = int(sys.argv[4]) if len(sys.argv) > 4 else 0
mx = int(sys.argv[5]) if len(sys.argv) > 5 else 12
width = int(sys.argv[6]) if len(sys.argv) > 6 else 260
lines = Path(f"refs/raw/fulltext/{key}.txt").read_text().splitlines()
print(f"==== {key} ({lines[0]})")
n, last = 0, -1
for i, l in enumerate(lines):
    if pat.search(l):
        n += 1
        if n > mx:
            print("  ... more matches truncated")
            break
        for j in range(max(i - before, last + 1), min(i + after + 1, len(lines))):
            print(f"{j + 1:>5}: {lines[j][:width]}")
            last = j
