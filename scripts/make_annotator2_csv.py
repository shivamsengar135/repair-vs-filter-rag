"""Blank labelling file for the second, independent annotator (Bansidhar Joshi), 2026-11-20.

Same 150 statements and evidence as labelling/human_labels.csv but with EMPTY label and note columns (no labels or
notes of the first annotator, no judge labels). Row ids are kept; the row ORDER is reshuffled (seed 20261120) so the
sequence does not match the first annotator's file. UTF-8 BOM for Excel.
Scoring: python3 scripts/score_two_annotators.py
"""
import csv
import random
from pathlib import Path

src = list(csv.DictReader(Path("labelling/claims_to_label.csv").open(encoding="utf-8")))
assert len(src) == 150
random.Random(20261120).shuffle(src)
out = Path("labelling/human_labels_annotator2.csv")
with out.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f, lineterminator="\r\n")
    w.writerow(["row", "topic", "claim", "label", "note", "evidence"])
    for r in src:
        w.writerow([r["row"], r["topic"], r["claim"], "", "", r["evidence"]])
back = list(csv.DictReader(out.open(encoding="utf-8-sig")))
assert len(back) == 150 and not any(b["label"] or b["note"] for b in back)
print("wrote", out, "first rows:", [b["row"] for b in back[:5]])
