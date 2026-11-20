"""Read-only: shape of labelling/failure_sample.csv."""
import csv
from collections import Counter
from pathlib import Path

rows = list(csv.DictReader(Path("labelling/failure_sample.csv").open(encoding="utf-8")))
print("rows:", len(rows)); print("columns:", list(rows[0]))
print(Counter((r["dataset"], r["error_type"]) for r in rows))
print("origins:", Counter(r["origin"] for r in rows), "actions:", Counter(r["action"] for r in rows))
print("claim changed by repair:", sum(r["claim_final"] != r["claim_original"] for r in rows))
print("best_passage empty:", sum(not r["best_passage"] for r in rows), "max len", max(len(r["best_passage"]) for r in rows))
print("ids:", rows[0]["sample_id"], "...", rows[-1]["sample_id"])
