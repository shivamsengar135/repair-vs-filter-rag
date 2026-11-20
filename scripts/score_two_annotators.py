"""Two-annotator judge validation: annotator 1 = labelling/human_labels.csv (Shivam Sengar, blind, scored 2026-11-20),
annotator 2 = labelling/human_labels_annotator2.csv (Bansidhar Joshi, independent, blank file).

Reports: human-human Cohen's kappa (rows where both gave S/NS), each annotator vs the judge (kappa + bootstrap 95% CI,
10,000 resamples, seed 0), and the judge vs the rows where both humans agree. The jointly revised file
labelling/human_labels_revised_joint.csv is NOT used (made after the first result was known).
Writes labelling/agreement_two_annotators.json. Usage: python3 scripts/score_two_annotators.py"""
import csv
import json
from pathlib import Path

import numpy as np


def read(p: str) -> dict[int, str]:
    rows = list(csv.DictReader(Path(p).open(encoding="utf-8-sig")))
    lab = {int(r["row"]): r["label"].strip().upper() for r in rows}
    if len(lab) != 150 or any(v not in ("S", "NS", "U") for v in lab.values()):
        raise SystemExit(f"{p}: need 150 rows labelled S / NS / U; invalid: "
                         f"{[(k, v) for k, v in lab.items() if v not in ('S', 'NS', 'U')][:10]}")
    return lab


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def kci(a: np.ndarray, b: np.ndarray) -> dict:
    rng = np.random.default_rng(0)
    boot = [kappa(a[i], b[i]) for i in rng.integers(0, len(a), size=(10_000, len(a)))]
    lo, hi = np.nanquantile(boot, [0.025, 0.975])
    return {"n": int(len(a)), "kappa": kappa(a, b), "ci95": [float(lo), float(hi)], "raw_agreement": float((a == b).mean())}


h1, h2 = read("labelling/human_labels.csv"), read("labelling/human_labels_annotator2.csv")
key = {r["row"]: r for r in json.loads(Path("labelling/.key_DO_NOT_OPEN.json").read_text())["rows"]}
judge = {i: bool(key[i]["judge_label"]) for i in key}
both = [i for i in sorted(h1) if h1[i] != "U" and h2[i] != "U"]
out = {"annotator_1": "Shivam Sengar (labelling/human_labels.csv)", "annotator_2": "Bansidhar Joshi (labelling/human_labels_annotator2.csv)",
       "counts": {"annotator_1": {k: list(h1.values()).count(k) for k in ("S", "NS", "U")},
                  "annotator_2": {k: list(h2.values()).count(k) for k in ("S", "NS", "U")}},
       "human_vs_human": kci(np.array([h1[i] == "S" for i in both], float), np.array([h2[i] == "S" for i in both], float))}
for name, h in (("annotator_1_vs_judge", h1), ("annotator_2_vs_judge", h2)):
    use = [i for i in sorted(h) if h[i] != "U"]
    out[name] = kci(np.array([h[i] == "S" for i in use], float), np.array([judge[i] for i in use], float))
agree = [i for i in both if h1[i] == h2[i]]
out["judge_vs_consensus_rows"] = kci(np.array([h1[i] == "S" for i in agree], float), np.array([judge[i] for i in agree], float))
Path("labelling/agreement_two_annotators.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
