"""Print section headings (with line numbers) and total line count of saved full texts. Usage: headings.py KEY ..."""
import sys
from pathlib import Path

for key in sys.argv[1:]:
    lines = Path(f"refs/raw/fulltext/{key}.txt").read_text().splitlines()
    print(f"==== {key} ({len(lines)} lines)")
    print("  ".join(f"[{i + 1}] {l[3:60]}" for i, l in enumerate(lines) if l.startswith("## ")))
