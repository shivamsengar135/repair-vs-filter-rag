import math

import numpy as np
import pytest

from hallu import metrics as M


def test_normalize_answer():
    assert M.normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert M.normalize_answer("An apple, a day.") == "apple day"


def test_exact_match_and_match_accuracy():
    assert M.exact_match("the Paris", ["Paris"]) == 1.0
    assert M.exact_match("Paris, France", ["Paris"]) == 0.0
    assert M.match_accuracy("It is Paris, France.", ["Lyon", "Paris"]) == 1.0
    assert M.match_accuracy("It is Lyon.", ["Paris"]) == 0.0
    assert M.match_accuracy("anything", ["the"]) == 0.0  # gold that normalises to empty never matches


def test_token_f1():
    assert M.token_f1("Barack Obama", ["Barack Obama"]) == 1.0
    assert M.token_f1("Obama", ["Barack Obama"]) == pytest.approx(2 * 1.0 * 0.5 / 1.5)
    assert M.token_f1("Trump", ["Barack Obama"]) == 0.0
    assert M.token_f1("", [""]) == 1.0
    assert M.token_f1("x", [""]) == 0.0
    assert M.token_f1("Obama", ["Trump", "Barack Obama"]) == pytest.approx(2 / 3)


def test_factual_precision_and_summary():
    assert M.factual_precision([True, False, True, True]) == 0.75
    assert M.factual_precision([]) is None
    s = M.response_summary([[True, True], [True, False, False], []])
    assert s["n_responses"] == 3
    assert s["supported_claims_per_response"] == pytest.approx(1.0)
    assert s["claims_per_response"] == pytest.approx(5 / 3)
    assert s["non_empty_rate"] == pytest.approx(2 / 3)
    assert s["factual_precision_macro"] == pytest.approx((1.0 + 1 / 3) / 2)
    assert s["factual_precision_micro"] == pytest.approx(3 / 5)
    assert s["fully_supported_rate"] == pytest.approx(1 / 3)  # the empty response does not count as supported
    assert s["responses_with_unsupported_claim_rate"] == pytest.approx(1 / 3)


def test_ece_perfect_and_worst():
    assert M.ece([1.0, 1.0, 0.0, 0.0], [True, True, False, False]) == pytest.approx(0.0)
    assert M.ece([1.0, 1.0], [False, False]) == pytest.approx(1.0)
    # two bins: conf .25 (acc 0) and .75 (acc 1) -> 0.5*0.25 + 0.5*0.25
    assert M.ece([0.25, 0.75], [False, True]) == pytest.approx(0.25)


def test_risk_coverage_and_aurc():
    conf, corr = [0.9, 0.8, 0.7, 0.6], [True, True, False, True]
    cov, risk = M.risk_coverage(conf, corr)
    assert cov.tolist() == [0.25, 0.5, 0.75, 1.0]
    assert risk.tolist() == pytest.approx([0, 0, 1 / 3, 0.25])
    assert M.aurc(conf, corr) == pytest.approx((0 + 0 + 1 / 3 + 0.25) / 4)
    assert M.selective_accuracy(conf, corr, 0.5) == 1.0
    assert M.selective_accuracy(conf, corr, 1.0) == 0.75
    # a perfectly ranked scorer has lower AURC than an inverted one
    assert M.aurc([0.9, 0.1], [True, False]) < M.aurc([0.1, 0.9], [True, False])


def test_auroc():
    assert M.auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert M.auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
    assert M.auroc([0.5, 0.5], [True, False]) == 0.5              # tie counts one half
    assert M.auroc([0.9, 0.3, 0.6, 0.1], [True, False, False, True]) == pytest.approx(0.5)  # pos {.9,.1} vs neg {.3,.6}: 2 of 4 pairs
    assert np.isnan(M.auroc([0.1, 0.2], [True, True]))
    rng = np.random.default_rng(0)
    y = rng.random(2000) < 0.3
    s = rng.normal(np.where(y, 1.0, 0.0), 1.0)
    # brute-force pairwise check
    pos, neg = s[y], s[~y]
    brute = ((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    assert M.auroc(s, y) == pytest.approx(brute)


def test_paired_bootstrap_detects_difference_and_null():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 400)
    r = M.paired_bootstrap(base + 0.5, base, n_resamples=2000, seed=0)
    assert r["diff"] == pytest.approx(0.5) and r["p_value"] < 0.01 and r["ci_low"] > 0
    noise = rng.normal(0, 1, 400)
    r0 = M.paired_bootstrap(base + noise - noise.mean(), base, n_resamples=2000, seed=0)
    assert abs(r0["diff"]) < 1e-9 and r0["p_value"] > 0.5 and r0["ci_low"] < 0 < r0["ci_high"]
    with pytest.raises(ValueError):
        M.paired_bootstrap([1, 2], [1])


def test_paired_bootstrap_is_deterministic():
    a, b = [1, 0, 1, 1, 0, 1], [0, 0, 1, 0, 0, 1]
    assert M.paired_bootstrap(a, b, 500, seed=3) == M.paired_bootstrap(a, b, 500, seed=3)


def test_bootstrap_lower_bound():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1, 500)
    lb = M.bootstrap_lower_bound(base + 0.3, base, 2000, seed=0)
    assert lb == pytest.approx(0.3)  # constant paired difference -> bound equals the difference
    lb2 = M.bootstrap_lower_bound(base + rng.normal(0, 1, 500), base, 2000, seed=0)
    assert lb2 < 0.1


def test_mcnemar_exact():
    # 8 discordant pairs all in one direction: p = 2 * (1/2)^8
    a = [True] * 8 + [True] * 5
    b = [False] * 8 + [True] * 5
    r = M.mcnemar(a, b)
    assert (r["n10"], r["n01"]) == (8, 0)
    assert r["p_value"] == pytest.approx(2 * 0.5 ** 8)
    assert M.mcnemar([True, False], [True, False])["p_value"] == 1.0
    # balanced discordance -> p = 1
    assert M.mcnemar([True, False], [False, True])["p_value"] == 1.0


def test_holm():
    assert M.holm([0.01, 0.04]) == [True, True]      # 0.01 <= 0.025, then 0.04 <= 0.05
    assert M.holm([0.03, 0.04]) == [False, False]    # smallest fails 0.025 -> stop
    assert M.holm([0.04, 0.01, 0.5]) == [False, True, False]
    assert M.holm([]) == []
