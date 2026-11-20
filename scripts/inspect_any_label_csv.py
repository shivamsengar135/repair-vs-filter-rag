"""Read-only: columns, annotator values, label counts, row order and overlap with the other label files.
Does not read the judge key. Usage: python3 scripts/inspect_any_label_csv.py FILE.csv"""
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

p = Path(sys.argv[1])
raw = p.read_bytes()
rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines(keepends=True)))
print("sha256", hashlib.sha256(raw).hexdigest()[:16], "rows", len(rows), "columns", list(rows[0]))
for c in rows[0]:
    if c not in ("topic", "claim", "evidence", "note", "row"):
        print(f"{c}: {dict(Counter(r[c].strip() for r in rows).most_common(8))}")
a2 = [r["row"] for r in csv.DictReader(Path("labelling/human_labels_annotator2.csv").open(encoding="utf-8-sig"))]
print("same order as the blank annotator-2 file:", [r["row"] for r in rows] == a2)
h1 = {r["row"]: r["label"].strip() for r in csv.DictReader(Path("labelling/human_labels.csv").open(encoding="utf-8-sig"))}
hj = {r["row"]: r["label"].strip() for r in csv.DictReader(Path("labelling/human_labels_revised_joint.csv").open(encoding="utf-8-sig"))}
same1 = sum(r["label"].strip() == h1[r["row"]] for r in rows)
samej = sum(r["label"].strip() == hj[r["row"]] for r in rows)
print(f"identical labels vs annotator 1: {same1}/150, vs joint revision: {samej}/150")
print("notes filled:", sum(bool(r["note"].strip()) for r in rows), "| example notes:", [r["note"][:70] for r in rows[:3]])
