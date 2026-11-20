"""Write labelling/human_labels.csv: the 150 statements for labelling by hand in Excel / LibreOffice.

Same rows and text as labelling/claims_to_label.csv, but with the columns the labeller fills (label, note) placed
right after the claim, and a UTF-8 BOM so Excel shows non-English characters correctly. No judge labels inside.
Scoring: bash scripts/py.sh scripts/score_labelling.py labelling/human_labels.csv
"""
import csv
from pathlib import Path

src = list(csv.DictReader(Path("labelling/claims_to_label.csv").open(encoding="utf-8")))
out = Path("labelling/human_labels.csv")
with out.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f, lineterminator="\r\n")
    w.writerow(["row", "topic", "claim", "label", "note", "evidence"])
    for r in src:
        w.writerow([r["row"], r["topic"], r["claim"], "", "", r["evidence"]])
back = list(csv.DictReader(out.open(encoding="utf-8-sig")))
assert len(back) == 150 and all(a["claim"] == b["claim"] and a["evidence"] == b["evidence"] for a, b in zip(src, back))
print("wrote", out, "with", len(back), "rows")
