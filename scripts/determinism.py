"""Residual serving non-determinism (EXPERIMENT_PLAN.md §10): compare the cached PopQA test standard-RAG run with
its three cache-free repeats (rep1..rep3): per-item identical predictions and accuracy spread.
Usage: python scripts/determinism.py"""
import glob
import json
from pathlib import Path

import numpy as np

from hallu.store import read_jsonl

base = [d for d in glob.glob("results/test/popqa/rag__Qwen2.5-7B-Instruct__*") if not d.rsplit("__", 1)[-1].startswith("rep")]
reps = sorted(glob.glob("results/test/popqa/rag__Qwen2.5-7B-Instruct__*__rep*"))
runs = {"cached": base[0], **{Path(r).name.rsplit("__", 1)[-1]: r for r in reps}}
preds = {k: {r["id"]: r for r in read_jsonl(Path(v) / "items.jsonl")} for k, v in runs.items()}
ids = sorted(preds["cached"])
acc = {k: float(np.mean([p[i]["match"] for i in ids])) for k, p in preds.items()}
same = {k: float(np.mean([preds[k][i]["prediction"] == preds["cached"][i]["prediction"] for i in ids])) for k in preds if k != "cached"}
all_same = float(np.mean([len({preds[k][i]["prediction"] for k in preds}) == 1 for i in ids]))
out = {"runs": runs, "accuracy": acc, "accuracy_mean": float(np.mean(list(acc.values()))), "accuracy_std": float(np.std(list(acc.values()), ddof=1)),
       "identical_prediction_rate_vs_cached": same, "identical_across_all_4": all_same, "n": len(ids)}
Path("results/test/determinism_popqa_rag.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
