"""Evaluation metrics and statistical tests (EXPERIMENT_PLAN.md §6 and §10).

Pure functions over plain Python / numpy data so they can be unit-tested without models.
"""
from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------- short-form correctness

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    """SQuAD-style normalisation: lower-case, strip punctuation and articles, collapse whitespace."""
    s = "".join(ch for ch in s.lower() if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: Sequence[str]) -> float:
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def match_accuracy(pred: str, golds: Sequence[str]) -> float:
    """1 if any normalised gold answer string occurs inside the normalised prediction ("match" accuracy)."""
    p = normalize_answer(pred)
    return float(any(normalize_answer(g) and normalize_answer(g) in p for g in golds))


def token_f1(pred: str, golds: Sequence[str]) -> float:
    """Maximum token-level F1 over the gold answers."""
    best = 0.0
    p = normalize_answer(pred).split()
    for g in golds:
        gt = normalize_answer(g).split()
        common = Counter(p) & Counter(gt)
        n = sum(common.values())
        if not p or not gt:
            f = float(p == gt)
        elif n == 0:
            f = 0.0
        else:
            prec, rec = n / len(p), n / len(gt)
            f = 2 * prec * rec / (prec + rec)
        best = max(best, f)
    return best


# ---------------------------------------------------------------- long-form (claim-level)


def factual_precision(labels: Sequence[bool]) -> float | None:
    """Fraction of claims labelled supported; None for an empty (abstained) response."""
    return (sum(bool(x) for x in labels) / len(labels)) if len(labels) else None


def response_summary(per_response_labels: Sequence[Sequence[bool]]) -> dict:
    """Aggregate claim-level judge labels of the FINAL answers of one method over a dataset."""
    n = len(per_response_labels)
    supported = [sum(bool(x) for x in r) for r in per_response_labels]
    total = [len(r) for r in per_response_labels]
    nonempty = [t > 0 for t in total]
    precs = [s / t for s, t in zip(supported, total) if t > 0]
    return {
        "n_responses": n,
        "supported_claims_per_response": float(np.mean(supported)) if n else float("nan"),
        "claims_per_response": float(np.mean(total)) if n else float("nan"),
        "non_empty_rate": float(np.mean(nonempty)) if n else float("nan"),
        # macro over non-empty responses (FActScore convention) and micro over all claims
        "factual_precision_macro": float(np.mean(precs)) if precs else float("nan"),
        "factual_precision_micro": (sum(supported) / sum(total)) if sum(total) else float("nan"),
        # fraction of ALL responses that are non-empty and contain no unsupported claim
        "fully_supported_rate": float(np.mean([t > 0 and s == t for s, t in zip(supported, total)])) if n else float("nan"),
        "responses_with_unsupported_claim_rate": float(np.mean([s < t for s, t in zip(supported, total)])) if n else float("nan"),
    }


# ---------------------------------------------------------------- calibration and selective prediction


def ece(conf: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins on [0, 1]."""
    conf_a, corr_a = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    if len(conf_a) == 0:
        return float("nan")
    bins = np.minimum((conf_a * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(corr_a[m].mean() - conf_a[m].mean())
    return float(total)


def risk_coverage(conf: Sequence[float], correct: Sequence[bool]) -> tuple[np.ndarray, np.ndarray]:
    """Coverage and selective risk when answering the top-k most confident items, k = 1..n (ties broken stably)."""
    conf_a, corr_a = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    order = np.argsort(-conf_a, kind="stable")
    errors = 1.0 - corr_a[order]
    k = np.arange(1, len(conf_a) + 1)
    return k / len(conf_a), np.cumsum(errors) / k


def aurc(conf: Sequence[float], correct: Sequence[bool]) -> float:
    """Area under the risk-coverage curve (mean selective risk over all coverage levels k/n)."""
    if len(conf) == 0:
        return float("nan")
    _, risk = risk_coverage(conf, correct)
    return float(risk.mean())


def selective_accuracy(conf: Sequence[float], correct: Sequence[bool], coverage: float) -> float:
    """Accuracy on the most confident `coverage` fraction of items (at least one item)."""
    cov, risk = risk_coverage(conf, correct)
    k = max(1, int(math.floor(coverage * len(cov) + 1e-9)))
    return float(1.0 - risk[k - 1])


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the ROC curve = P(score of a random positive > score of a random negative), ties count 1/2
    (Mann-Whitney U / (n_pos * n_neg)). NaN if one class is empty."""
    s, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):                      # average ranks over ties
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    u = ranks[y].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


# ---------------------------------------------------------------- statistical tests


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_resamples: int = 10_000, seed: int = 0,
                     alpha: float = 0.05) -> dict:
    """Paired bootstrap over items for mean(a) - mean(b).

    Returns the observed difference, a percentile CI and a two-sided p-value computed as
    2 * min(P(diff* <= 0), P(diff* >= 0)) over the bootstrap distribution, capped at 1.
    """
    a_a, b_a = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a_a.shape != b_a.shape or a_a.ndim != 1 or len(a_a) == 0:
        raise ValueError("a and b must be non-empty 1-D arrays of equal length (paired by item)")
    d = a_a - b_a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_resamples, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    return {"diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "p_value": float(p),
            "n": int(len(d)), "n_resamples": n_resamples, "seed": seed}


def bootstrap_lower_bound(a: Sequence[float], b: Sequence[float], n_resamples: int = 10_000, seed: int = 0,
                          alpha: float = 0.05) -> float:
    """One-sided (1 - alpha) lower confidence bound for mean(a) - mean(b); used for the non-inferiority guard."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(n_resamples, len(d)))].mean(axis=1)
    return float(np.quantile(boot, alpha))


def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict:
    """Exact (binomial) McNemar test for paired binary outcomes."""
    a_a, b_a = np.asarray(a_correct, dtype=bool), np.asarray(b_correct, dtype=bool)
    if a_a.shape != b_a.shape:
        raise ValueError("paired inputs must have equal length")
    n01 = int((~a_a & b_a).sum())  # a wrong, b right
    n10 = int((a_a & ~b_a).sum())  # a right, b wrong
    n = n01 + n10
    if n == 0:
        return {"n10": n10, "n01": n01, "p_value": 1.0}
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"n10": n10, "n01": n01, "p_value": float(min(1.0, 2 * tail))}


def holm(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm step-down procedure; returns reject decisions in the original order."""
    order = np.argsort(p_values)
    m = len(p_values)
    reject = [False] * m
    for rank, i in enumerate(order):
        if p_values[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject
