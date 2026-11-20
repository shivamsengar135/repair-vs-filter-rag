"""Score the human failure categorisation (labelling/failure_categories.csv) and compare it with the assistant's
pre-annotation. Writes results/test/failure_analysis_human.json: human counts per dataset x error type (these are
what the paper reports), agreement and Cohen's kappa with the pre-annotation, and the confusion table.
Usage: python3 scripts/score_failure_categories.py [labelling/failure_categories.csv]"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

CODES = ["JE", "RM", "VFP", "VFN", "DE", "RE", "UNC"]
path = Path(sys.argv[1] if len(sys.argv) > 1 else "labelling/failure_categories.csv")
rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
sample = {r["sample_id"].rsplit("-", 1)[-1]: r for r in csv.DictReader(Path("labelling/failure_sample.csv").open(encoding="utf-8"))}
if len(rows) != 96 or {r["case"].zfill(3) for r in rows} != set(sample):
    raise SystemExit(f"expected the 96 cases 000-095, got {len(rows)} rows")
human = {r["case"].zfill(3): r["category"].strip().upper() for r in rows}
bad = {k: v for k, v in human.items() if v not in CODES}
if bad:
    raise SystemExit(f"{len(bad)} rows without a valid category {CODES}: {dict(list(bad.items())[:10])}")
pre = {k: v[0] for k, v in json.loads(Path("labelling/failure_preannotation_assistant.json").read_text())["labels"].items()}
ids = sorted(human)
po = sum(human[i] == pre[i] for i in ids) / len(ids)
ph, pa = Counter(human[i] for i in ids), Counter(pre[i] for i in ids)
pe = sum(ph[c] * pa[c] for c in CODES) / len(ids) ** 2
counts = defaultdict(Counter)
for i in ids:
    s = sample[i]
    counts[f"{s['dataset']}/{s['error_type']}"][human[i]] += 1
out = {"annotator": "human (see results/RESULTS_LEDGER.md)", "n": len(ids),
       "counts_human": {k: dict(v) for k, v in sorted(counts.items())},
       "totals_human": dict(ph), "totals_assistant": dict(pa),
       "agreement_with_assistant": po, "cohen_kappa_vs_assistant": (po - pe) / (1 - pe),
       "confusion_human_x_assistant": {h: {a: sum(human[i] == h and pre[i] == a for i in ids) for a in CODES if
                                           any(human[i] == h and pre[i] == a for i in ids)} for h in CODES if ph[h]},
       "notes_written": sum(bool(r["notes"].strip()) for r in rows)}
Path("results/test/failure_analysis_human.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
