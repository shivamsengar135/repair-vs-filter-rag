"""Print the human's notes for selected failure cases (RM, DE, VFP, JE-on-removed, UNC, and re-retrieved rows)."""
import csv
from pathlib import Path

for r in csv.DictReader(Path("labelling/failure_categories.csv").open(encoding="utf-8-sig")):
    c = r["category"].strip().upper()
    kept = r["what_went_wrong"].startswith("KEPT")
    if c in ("RM", "DE", "VFP", "UNC") or (c == "JE" and not kept) or r["repair"].startswith("re-retrieved"):
        print(f"{r['case']} {'KEPT' if kept else 'REMOVED'} {r['dataset'][:4]} {c:4s} | {r['claim'][:90]} | NOTE: {r['notes'][:160]}"
              + (" | RE-RETRIEVED" if r["repair"].startswith("re-retrieved") else ""))
