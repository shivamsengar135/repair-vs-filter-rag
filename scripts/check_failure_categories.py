"""Validate a filled failure_categories.csv against the generated one (no pre-annotation is read here).
Usage: python3 scripts/check_failure_categories.py PATH"""
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

p = Path(sys.argv[1])
raw = p.read_bytes()
print("sha256", hashlib.sha256(raw).hexdigest()[:16], "BOM", raw[:3] == b"\xef\xbb\xbf", len(raw), "bytes")
rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines(keepends=True)))
ref = list(csv.DictReader(Path("labelling/failure_categories.csv").open(encoding="utf-8-sig")))
print("rows", len(rows), "columns", list(rows[0]))
key = lambda r: (r["case"].zfill(3), r["claim"].strip())
print("same cases and claims:", sorted(map(key, rows)) == sorted(map(key, ref)))
print("passages unchanged:", all(" ".join(a["best_retrieved_passage"].split()) == " ".join(b["best_retrieved_passage"].split())
                                 for a, b in zip(sorted(rows, key=key), sorted(ref, key=key))))
print("categories:", dict(Counter(r["category"].strip() for r in rows)))
print("invalid:", [(r["case"], r["category"]) for r in rows if r["category"].strip().upper() not in {"JE", "RM", "VFP", "VFN", "DE", "RE", "UNC"}][:15])
print("notes:", sum(bool(r["notes"].strip()) for r in rows))
print("by row type:", dict(Counter((r["what_went_wrong"][:7], r["category"].strip()) for r in rows)))
