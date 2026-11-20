"""Compact view of a long-form run: zero-claim responses and every non-kept claim with its repair.
Usage: show_longform.py ITEMS.jsonl [max_rows]"""
import sys

from hallu.store import read_jsonl

rows = read_jsonl(sys.argv[1])
mx = int(sys.argv[2]) if len(sys.argv) > 2 else 40
n = 0
for r in rows:
    if not r["claims"]:
        print(f"ZERO CLAIMS | {r['question'][:60]} | answer: {r['answer'][:160]!r} | decomposition: {r['decomposition_raw'][:80]!r}")
for r in rows:
    for c in r["claims"]:
        if c["action"] != "keep" and n < mx:
            n += 1
            rep = c.get("repair")
            extra = ""
            if rep:
                extra = f" => {rep['text']!r} s_after={rep['s_after']:.2f}" if c["action"] == "revise" else f" => s_after={rep['s_after']:.2f} via {(rep.get('new_passage') or {}).get('title')}"
            print(f"{c['action']:10s} s={c['s']:.2f} con={c['con']:.2f} c={c['c']:.2f} | {r['question'][27:55]:28s} | {c['text'][:95]}{extra}")
