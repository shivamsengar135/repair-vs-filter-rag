"""Print line ranges of saved full texts, joining consecutive table-cell lines ('| x') into one line.
Usage: python3 scripts/lit/lines.py KEY:START-END[:WIDTH] ...
"""
import sys
from pathlib import Path

for spec in sys.argv[1:]:
    parts = spec.split(":")
    key, (a, b) = parts[0], map(int, parts[1].split("-"))
    w = int(parts[2]) if len(parts) > 2 else 240
    lines = Path(f"refs/raw/fulltext/{key}.txt").read_text().splitlines()
    print(f"==== {key} lines {a}-{b}")
    buf: list[str] = []
    start = None

    def flush() -> None:
        global buf, start
        if buf:
            print(f"{start}: " + " ".join(buf))
            buf = []

    for i in range(a - 1, min(b, len(lines))):
        l = lines[i]
        if l.startswith("|"):
            if not buf:
                start = i + 1
            buf.append(l[:60] if len(l) > 2 else "|_")
        else:
            flush()
            if len(l) > 2:
                print(f"{i + 1}: {l[:w]}")
    flush()
