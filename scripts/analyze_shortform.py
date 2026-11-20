"""Selective-answering analysis of a short-form run (EXPERIMENT_PLAN.md §6): standard RAG vs CF / C / A.

For each method: accuracy without abstention, AURC, selective accuracy at 50% / 80% coverage, ECE of its score
used as a confidence, and (single-claim conformal filter = selective answering) at targets 0.80-0.95:
answer rate (coverage), accuracy among answered, and P(abstain or correct) — the conformal target quantity.
Correctness = `match` accuracy. Usage: python scripts/analyze_shortform.py --run results/dev/popqa/shortform__...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hallu import analysis as A
from hallu import metrics as M
from hallu.runutil import git_commit
from hallu.store import read_jsonl

TARGETS = [0.80, 0.85, 0.90, 0.95]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-partitions", type=int, default=100)
    a = ap.parse_args()
    run = Path(a.run)
    rows = read_jsonl(run / "items.jsonl")
    methods = {
        "CF (score s)": ([r["s"] for r in rows], [bool(r["rag_match"]) for r in rows]),
        "C (two-signal)": ([r["score_c"] for r in rows], [bool(r["rag_match"]) for r in rows]),
        "A (repair)": ([r["A_score"] for r in rows], [bool(r["A_match"]) for r in rows]),
    }
    out = {"run": str(run), "n": len(rows), "git_commit": git_commit(),
           "standard_rag": {"accuracy": float(np.mean([r["rag_match"] for r in rows])), "coverage": 1.0}, "methods": {}}
    for name, (score, corr) in methods.items():
        resp = [[(s, c)] for s, c in zip(score, corr)]
        m = {"accuracy_no_abstention": float(np.mean(corr)), "aurc": M.aurc(score, corr),
             "selective_acc_cov50": M.selective_accuracy(score, corr, 0.5), "selective_acc_cov80": M.selective_accuracy(score, corr, 0.8),
             "ece": M.ece(score, corr), "targets": {}}
        for t in TARGETS:
            e = A.evaluate(resp, 1 - t, a.n_partitions)
            m["targets"][str(t)] = {"coverage": e["non_empty_rate"], "accuracy_answered": e["factual_precision_micro"],
                                    "abstain_or_correct": e["all_retained_correct_rate"]}
        out["methods"][name] = m
    (run / "analysis.json").write_text(json.dumps(out, indent=1))
    print(f"run {run}  n={len(rows)}  standard RAG accuracy {out['standard_rag']['accuracy']:.3f}")
    print(f"{'method':16s} {'acc':>6s} {'AURC':>6s} {'sel@50':>7s} {'sel@80':>7s} {'ECE':>6s} | " +
          " ".join(f"{'cov/acc@' + str(t):>18s}" for t in TARGETS))
    for name, m in out["methods"].items():
        cells = [f"{m['targets'][str(t)]['coverage']['mean']:.2f}/{m['targets'][str(t)]['accuracy_answered']['mean']:.2f}" for t in TARGETS]
        print(f"{name:16s} {m['accuracy_no_abstention']:6.3f} {m['aurc']:6.3f} {m['selective_acc_cov50']:7.3f} {m['selective_acc_cov80']:7.3f} "
              f"{m['ece']:6.3f} | " + " ".join(f"{c:>18s}" for c in cells))


if __name__ == "__main__":
    main()
