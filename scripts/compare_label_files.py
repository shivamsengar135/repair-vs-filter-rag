"""Compare a new human-labels CSV with the scored labelling/human_labels.csv (no judge labels are read here).
Shows columns, label counts, and which rows changed. Usage: python3 scripts/compare_label_files.py NEW.csv"""
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

new_p = Path(sys.argv[1])
raw = new_p.read_bytes()
print("sha256", hashlib.sha256(raw).hexdigest()[:16], "BOM", raw[:3] == b"\xef\xbb\xbf", len(raw), "bytes")
new = list(csv.DictReader(raw.decode("utf-8-sig").splitlines(keepends=True)))
old = {r["row"]: r for r in csv.DictReader(Path("labelling/human_labels.csv").open(encoding="utf-8-sig"))}
cols = list(new[0].keys())
print("columns:", cols, "rows:", len(new))
print("same rows/claims:", sorted(r["row"] for r in new) == sorted(old) and all(r["claim"].strip() == old[r["row"]]["claim"].strip() for r in new))
label_cols = [c for c in cols if "label" in c.lower() or "annot" in c.lower() or "author" in c.lower()]
print("label-like columns:", label_cols)
for c in label_cols:
    print(f"  {c}: {dict(Counter(r[c].strip() for r in new))}")
if "label" in cols:
    ch = [(r["row"], old[r["row"]]["label"].strip(), r["label"].strip()) for r in new if r["label"].strip().upper() != old[r["row"]]["label"].strip().upper()]
    print("rows whose 'label' differs from the scored file:", len(ch), ch[:20])
other = [c for c in cols if c not in ("row", "topic", "claim", "label", "note", "evidence")]
print("extra columns:", other)
for c in other:
    vals = Counter(r[c].strip() for r in new)
    print(f"  {c}: {len(vals)} distinct, e.g. {list(vals.items())[:6]}")
nd = [(r["row"], old[r["row"]]["note"][:60], r.get("note", "")[:60]) for r in new if r.get("note", "").strip() != old[r["row"]]["note"].strip()]
print("rows whose note differs:", len(nd), nd[:5])
