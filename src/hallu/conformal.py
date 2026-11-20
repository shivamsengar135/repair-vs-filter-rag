"""Split-conformal claim filtering ("conformal factuality", Mohri & Hashimoto, arXiv:2402.10978).

Definitions follow the paper (Algorithm and Section 3-5, read from the arXiv HTML on 2026-11-13):

* Each response is a set of sub-claims with a confidence score; filtering at threshold t keeps the claims whose
  score is >= t (higher score = more likely correct).
* For a calibration response, r is the *minimum strictly safe threshold*: the smallest t such that for every
  t' >= t the retained set contains only correct claims. With "keep score >= t", any threshold strictly above the
  highest-scoring incorrect claim is safe, so we represent r by that maximum incorrect score and keep, at test
  time, only claims with score STRICTLY GREATER than the calibrated quantile. A response with no incorrect claim
  has r = -inf (nothing needs removing).
* q_hat is the ceil((n + 1)(1 - alpha)) / n empirical quantile of the calibration r values, i.e. the
  ceil((n + 1)(1 - alpha))-th smallest value. If that rank exceeds n the threshold is +inf (remove everything).

Guarantee (under exchangeability of calibration and test responses): P(all retained claims correct) >= 1 - alpha.
This is a RESPONSE-level guarantee. The partial-correctness variant (at least a fraction `a` of retained claims
correct) is implemented through `min_correct_fraction`.
"""
from __future__ import annotations

import math
from typing import Sequence

NEG_INF, POS_INF = float("-inf"), float("inf")


def min_safe_threshold(scores: Sequence[float], correct: Sequence[bool], min_correct_fraction: float = 1.0) -> float:
    """Return r: keeping claims with score > r is safe for every threshold >= r.

    Safe means the fraction of correct claims among the retained ones is >= min_correct_fraction (an empty
    retained set counts as safe). Scanning thresholds from high to low, r is the largest score value at which
    keeping "score >= that value" would be unsafe; -inf if no threshold is unsafe.
    """
    if len(scores) != len(correct):
        raise ValueError("scores and correct must have equal length")
    if not 0.0 < min_correct_fraction <= 1.0:
        raise ValueError("min_correct_fraction must be in (0, 1]")
    pairs = sorted(zip(scores, correct), key=lambda p: -p[0])
    r = NEG_INF
    kept = good = 0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:  # ties enter together
            kept += 1
            good += bool(pairs[j][1])
            j += 1
        if good / kept < min_correct_fraction - 1e-12 and r == NEG_INF:
            r = pairs[i][0]  # highest unsafe level; every threshold must be strictly above it
        i = j
    return r


def calibrate(r_values: Sequence[float], alpha: float) -> float:
    """q_hat = the ceil((n + 1)(1 - alpha))-th smallest calibration score; +inf if that rank exceeds n."""
    n = len(r_values)
    if n == 0:
        raise ValueError("need at least one calibration response")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    rank = math.ceil((n + 1) * (1 - alpha) - 1e-12)
    if rank > n:
        return POS_INF
    return sorted(r_values)[rank - 1]


def keep_mask(scores: Sequence[float], q_hat: float) -> list[bool]:
    """Claims retained at test time: score strictly greater than q_hat."""
    return [s > q_hat for s in scores]


def calibrate_from_responses(cal_scores: Sequence[Sequence[float]], cal_correct: Sequence[Sequence[bool]],
                             alpha: float, min_correct_fraction: float = 1.0) -> float:
    r = [min_safe_threshold(s, c, min_correct_fraction) for s, c in zip(cal_scores, cal_correct)]
    return calibrate(r, alpha)
