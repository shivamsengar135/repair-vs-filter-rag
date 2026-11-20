"""Print the per-claim fields of a long-form run and an estimate of each paper table's natural width.
Usage: python scripts/inspect_fig_data.py"""
import json
import re
from pathlib import Path

run = Path("results/test/bio/longform__Qwen2.5-7B-Instruct__5aa2ad1482e5")
row = json.loads((run / "items.jsonl").open(encoding="utf-8").readline())
for k, v in row.items():
    print(k, type(v).__name__, (json.dumps(v[0])[:400] if isinstance(v, list) and v else str(v)[:80]))

# natural width estimate: widest tabular row (LaTeX commands stripped), ~4.4 pt per char at \small + 8 pt per column
for f in sorted(Path("paper/tables").glob("*.tex")):
    if f.name == "prompts.tex":
        continue
    best, ncol = 0, 0
    for line in f.read_text(encoding="utf-8").splitlines():
        if "&" in line and line.rstrip().endswith("\\\\"):
            cells = [re.sub(r"\\[a-zA-Z]+|[{}$\\]", "", c).strip() for c in line.rstrip()[:-2].split("&")]
            w = sum(len(c) for c in cells) * 4.4 + 8 * len(cells)
            best, ncol = max(best, w), len(cells)
    print(f"{f.name}: {ncol} cols, ~{best:.0f} pt (text width ~466 pt, landscape ~680 pt)")
