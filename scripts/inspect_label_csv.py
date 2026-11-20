"""Read-only: shape of labelling/claims_to_label.csv (rows, topics, evidence length, labels already filled)."""
import csv
from collections import Counter
from pathlib import Path

rows = list(csv.DictReader(Path("labelling/claims_to_label.csv").open(encoding="utf-8")))
print("rows:", len(rows), "columns:", list(rows[0].keys()))
print("labels filled:", Counter(r["label"].strip() for r in rows))
ev = [len(r["evidence"]) for r in rows]
print("evidence chars: min", min(ev), "median", sorted(ev)[len(ev) // 2], "max", max(ev))
print("claim chars max:", max(len(r["claim"]) for r in rows))
print("example topics:", [r["topic"][:40] for r in rows[:5]])
print("key file present:", Path("labelling/.key_DO_NOT_OPEN.json").exists())
