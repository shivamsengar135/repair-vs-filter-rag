"""Audit claim decomposition on a long-form run: how often does a response with factual content get zero or very few
claims? Heuristic content signal = number of sentences that are not pure "the passages do not say" meta-statements.
Usage: python scripts/audit_decomposition.py ITEMS.jsonl [n_examples]
"""
import re
import sys

from hallu.store import read_jsonl

META = re.compile(r"(passages?|context|information provided|provided (text|information)|not (explicitly )?(mention|contain|provide|specif))", re.I)
rows = read_jsonl(sys.argv[1])
n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 4
stats = {"responses": len(rows), "zero_claims": 0, "zero_but_content": 0, "low_ratio": 0}
examples = []
for r in rows:
    sents = [s for s in re.split(r"(?<=[.!?])\s+", r["answer"].strip()) if len(s.split()) >= 4]
    content = [s for s in sents if not META.search(s)]
    n = len(r["claims"])
    if n == 0:
        stats["zero_claims"] += 1
        if content:
            stats["zero_but_content"] += 1
            examples.append((r["question"], content[:2]))
    elif content and n < len(content):
        stats["low_ratio"] += 1   # fewer claims than content sentences: some sentences produced no claim
print(stats)
for q, c in examples[:n_show]:
    print(f"- Q: {q[:70]} | content sentences: {[s[:110] for s in c]}")
