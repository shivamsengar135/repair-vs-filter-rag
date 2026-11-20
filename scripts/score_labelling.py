"""Agreement between the user's labels and the judge (EXPERIMENT_PLAN.md §7): Cohen's kappa with a bootstrap 95% CI.
Rows labelled U are excluded from kappa and reported. Acceptance rule (fixed in the plan): kappa >= 0.60 accept;
0.40-0.60 accept with warning + human-only sensitivity analysis; < 0.40 reject.
Usage: python scripts/score_labelling.py [labelling/claims_to_label.csv]
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "labelling/claims_to_label.csv")
    key = {r["row"]: r for r in json.loads((path.parent / ".key_DO_NOT_OPEN.json").read_text())["rows"]}
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))  # -sig: a file re-saved by Excel starts with a BOM
    if len(rows) != len(key):
        raise SystemExit(f"expected {len(key)} rows, found {len(rows)} (was the file cut or re-sorted?)")
    labels = {int(r["row"]): r["label"].strip().upper() for r in rows}
    missing = [i for i, l in labels.items() if l not in ("S", "NS", "U")]
    if missing:
        raise SystemExit(f"{len(missing)} rows without a valid label (S / NS / U), e.g. rows {missing[:10]}")
    use = [i for i, l in labels.items() if l != "U"]
    h = np.array([labels[i] == "S" for i in use], dtype=float)
    j = np.array([key[i]["judge_label"] for i in use], dtype=float)
    k = kappa(h, j)
    rng = np.random.default_rng(0)
    boot = [kappa(h[ix], j[ix]) for ix in rng.integers(0, len(use), size=(10_000, len(use)))]
    lo, hi = np.nanquantile(boot, [0.025, 0.975])
    by_src = {}
    for src in ("rag", "repaired", "kept_by_A"):
        ix = [n for n, i in enumerate(use) if key[i]["source"] == src]
        by_src[src] = {"n": len(ix), "agreement": float((h[ix] == j[ix]).mean()) if ix else None,
                       "human_supported_rate": float(h[ix].mean()) if ix else None, "judge_supported_rate": float(j[ix].mean()) if ix else None}
    verdict = "ACCEPT" if k >= 0.60 else ("ACCEPT WITH WARNING" if k >= 0.40 else "REJECT")
    out = {"n_labelled": len(labels), "n_unclear": len(labels) - len(use), "n_used": len(use), "cohen_kappa": k,
           "kappa_95ci": [float(lo), float(hi)], "raw_agreement": float((h == j).mean()), "by_source": by_src, "verdict": verdict}
    (path.parent / "agreement.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
