"""Matched-target evaluation of claim-filtering methods (EXPERIMENT_PLAN.md §6, clarifications C1 and C5).

Input per method: for every response, the list of its candidate claims as (score, judge_label).
For a target 1 - alpha, the filter threshold is calibrated by split conformal prediction on a random half of the
responses and applied to the other half; this is repeated over seeded partitions. The SAME partitions are used
for every method, so per-question results are paired across methods.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from hallu import conformal as C

Resp = Sequence[tuple[float, bool]]  # one response: [(score, judge says supported), ...]


def partition(n: int, seed: int, cal_frac: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Seeded calibration / evaluation index split of n responses."""
    perm = np.random.default_rng(seed).permutation(n)
    n_cal = int(round(cal_frac * n))
    return np.sort(perm[:n_cal]), np.sort(perm[n_cal:])


def apply_threshold(resp: Resp, q_hat: float) -> list[bool]:
    """Judge labels of the claims that survive the filter (score strictly above q_hat)."""
    return [lab for (s, lab) in resp if s > q_hat]


def per_response_stats(kept: Sequence[bool]) -> dict:
    n, sup = len(kept), int(sum(kept))
    return {"supported": sup, "kept": n, "non_empty": float(n > 0), "all_correct": float(sup == n),  # empty counts as correct
            "precision": (sup / n) if n else np.nan}


def evaluate_once(responses: Sequence[Resp], alpha: float | None, seed: int, cal_frac: float = 0.5) -> dict:
    """One partition. alpha=None means no filtering (all claims kept) but the same evaluation half is used."""
    cal, ev = partition(len(responses), seed, cal_frac)
    if alpha is None:
        q = C.NEG_INF
    else:
        q = C.calibrate_from_responses([[s for s, _ in responses[i]] for i in cal], [[lab for _, lab in responses[i]] for i in cal], alpha)
    stats = [per_response_stats(apply_threshold(responses[i], q)) for i in ev]
    tot_kept = sum(s["kept"] for s in stats)
    return {
        "q_hat": q, "eval_index": ev,
        "supported_per_response": np.array([s["supported"] for s in stats], dtype=float),
        "precision_per_response": np.array([s["precision"] for s in stats], dtype=float),
        "summary": {
            "supported_claims_per_response": float(np.mean([s["supported"] for s in stats])),
            "claims_kept_per_response": float(np.mean([s["kept"] for s in stats])),
            "non_empty_rate": float(np.mean([s["non_empty"] for s in stats])),
            "all_retained_correct_rate": float(np.mean([s["all_correct"] for s in stats])),   # the conformal target quantity
            "factual_precision_micro": (sum(s["supported"] for s in stats) / tot_kept) if tot_kept else float("nan"),
            "factual_precision_macro": float(np.nanmean([s["precision"] for s in stats])) if tot_kept else float("nan"),
        },
    }


def evaluate(responses: Sequence[Resp], alpha: float | None, n_partitions: int = 100, cal_frac: float = 0.5) -> dict:
    """Mean and std of every summary metric over partitions seeded 0..n_partitions-1."""
    runs = [evaluate_once(responses, alpha, seed, cal_frac)["summary"] for seed in range(n_partitions)]
    out = {}
    for k in runs[0]:
        vals = np.array([r[k] for r in runs], dtype=float)
        out[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0}
    out["n_partitions"] = n_partitions
    return out


def original_retention(responses: Sequence[Resp], q_hat: float) -> float:
    """Fraction of candidate claims kept at a threshold (the 'claim retention' reported by conformal-RAG papers)."""
    total = sum(len(r) for r in responses)
    return (sum(sum(1 for s, _ in r if s > q_hat) for r in responses) / total) if total else float("nan")
