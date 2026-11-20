"""Print a seeded sample of per-item outputs from a run. Usage: show_items.py ITEMS.jsonl [n] [wrong|right|all]"""
import random
import sys

from hallu.store import read_jsonl

rows = read_jsonl(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
which = sys.argv[3] if len(sys.argv) > 3 else "all"
if which != "all":
    rows = [r for r in rows if (r["match"] == 0.0) == (which == "wrong")]
random.Random(0).shuffle(rows)
for r in rows[:n]:
    print(f"match={r['match']:.0f} em={r['em']:.0f} | Q: {r['question'][:80]} | pred: {r['raw'][:90]!r} | gold: {r['answers'][:3]}")
