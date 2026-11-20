"""Validate a filled human_labels.csv against the original statements (no judge labels are read here).
Checks: 150 rows, same row ids / claims / evidence as labelling/claims_to_label.csv, every label in {S, NS, U}.
Usage: python3 scripts/check_human_labels.py PATH"""
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
raw = path.read_bytes()
print("file:", path, len(raw), "bytes, sha256", hashlib.sha256(raw).hexdigest()[:16], "BOM:", raw[:3] == b"\xef\xbb\xbf")
for enc in ("utf-8-sig", "cp1252"):
    try:
        text = raw.decode(enc); print("decoded as", enc); break
    except UnicodeDecodeError:
        continue
rows = list(csv.DictReader(text.splitlines(keepends=True)))
print("columns:", list(rows[0].keys()), "rows:", len(rows))
orig = {r["row"]: r for r in csv.DictReader(Path("labelling/claims_to_label.csv").open(encoding="utf-8"))}
bad_text = [r["row"] for r in rows if r["row"] not in orig or r["claim"].strip() != orig[r["row"]]["claim"].strip()]
ev_changed = [r["row"] for r in rows if r["row"] in orig and " ".join(r["evidence"].split()) != " ".join(orig[r["row"]]["evidence"].split())]
labels = Counter(r["label"].strip() for r in rows)
invalid = [(r["row"], r["label"]) for r in rows if r["label"].strip().upper() not in ("S", "NS", "U")]
print("row ids match:", sorted(orig) == sorted(r["row"] for r in rows), "| claims differing:", bad_text[:10], "| evidence differing:", ev_changed[:10])
print("labels:", dict(labels))
print("invalid labels:", invalid[:15], "count", len(invalid))
print("notes written:", sum(bool(r.get("note", "").strip()) for r in rows))
