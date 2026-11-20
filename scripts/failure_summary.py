"""Counts of the failure-analysis categories (assistant pre-annotation; plan §8) by dataset x error type.
Writes results/test/failure_analysis_preannotation.json. Usage: python scripts/failure_summary.py"""
import collections
import csv
import json
from pathlib import Path

rows = list(csv.DictReader(open("labelling/failure_sample.csv", encoding="utf-8")))
pre = json.loads(Path("labelling/failure_preannotation_assistant.json").read_text())
cnt: dict = collections.defaultdict(collections.Counter)
for r in rows:
    code = pre["labels"][r["sample_id"].rsplit("-", 1)[-1]][0]
    cnt[f"{r['dataset']}/{r['error_type']}"][code] += 1
out = {"annotator": pre["annotator"], "codes": pre["codes"], "n": len(rows), "counts": {k: dict(v) for k, v in cnt.items()}}
Path("results/test/failure_analysis_preannotation.json").write_text(json.dumps(out, indent=1))
for k, v in out["counts"].items():
    print(k, sum(v.values()), dict(sorted(v.items(), key=lambda x: -x[1])))
