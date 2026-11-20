import numpy as np
import pytest

from hallu import conformal as C


def test_min_safe_threshold_full_correctness():
    # incorrect claims at .7 and .2 -> must keep only scores strictly above .7
    assert C.min_safe_threshold([0.9, 0.7, 0.5, 0.2], [True, False, True, False]) == 0.7
    assert C.min_safe_threshold([0.9, 0.5], [True, True]) == C.NEG_INF
    assert C.min_safe_threshold([], []) == C.NEG_INF
    assert C.min_safe_threshold([0.9], [False]) == 0.9


def test_min_safe_threshold_ties_enter_together():
    # a correct and an incorrect claim share score .8 -> the tie level itself is unsafe
    assert C.min_safe_threshold([0.8, 0.8, 0.3], [True, False, True]) == 0.8


def test_min_safe_threshold_partial_is_strictly_safe():
    # a = 0.5: keeping {.9 F} is unsafe (0/1); adding .8 T gives 1/2 (safe); adding .7 T gives 2/3 (safe).
    # "Strictly safe" needs every HIGHER threshold to be safe too, so r is the highest unsafe level, .9.
    assert C.min_safe_threshold([0.9, 0.8, 0.7], [False, True, True], min_correct_fraction=0.5) == 0.9
    # a = 0.5 with the error lower down: {.9 T} ok, {.9 T,.8 F} = 1/2 ok, {+.7 F} = 1/3 unsafe -> r = .7
    assert C.min_safe_threshold([0.9, 0.8, 0.7], [True, False, False], min_correct_fraction=0.5) == 0.7


def test_min_safe_threshold_input_checks():
    with pytest.raises(ValueError):
        C.min_safe_threshold([0.1], [True, False])
    with pytest.raises(ValueError):
        C.min_safe_threshold([0.1], [True], min_correct_fraction=0.0)


def test_calibrate_rank():
    r = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # n = 9
    assert C.calibrate(r, alpha=0.2) == 0.8   # ceil(10 * 0.8) = 8th smallest
    assert C.calibrate(r, alpha=0.5) == 0.5   # ceil(10 * 0.5) = 5th
    assert C.calibrate(r, alpha=0.1) == 0.9   # ceil(10 * 0.9) = 9th
    assert C.calibrate(r, alpha=0.05) == C.POS_INF  # rank 10 > n -> remove everything
    assert C.calibrate(list(reversed(r)), alpha=0.2) == 0.8  # order of inputs is irrelevant
    with pytest.raises(ValueError):
        C.calibrate([], 0.1)
    with pytest.raises(ValueError):
        C.calibrate([0.1], 1.0)


def test_keep_mask_is_strict():
    assert C.keep_mask([0.9, 0.7, 0.5], 0.7) == [True, False, False]
    assert C.keep_mask([0.9], C.POS_INF) == [False]
    assert C.keep_mask([0.1], C.NEG_INF) == [True]


def _simulate(rng, n_resp, informative=True):
    """Responses with 1-8 claims; correct claims tend to score higher when the scorer is informative."""
    scores, correct = [], []
    for _ in range(n_resp):
        m = int(rng.integers(1, 9))
        c = rng.random(m) < 0.7
        s = rng.normal(loc=np.where(c, 1.0, 0.0) if informative else 0.0, scale=1.0, size=m)
        scores.append(s.tolist())
        correct.append(c.tolist())
    return scores, correct


@pytest.mark.parametrize("alpha", [0.1, 0.2])
@pytest.mark.parametrize("informative", [True, False])
def test_marginal_guarantee_holds_in_simulation(alpha, informative):
    """P(all retained claims correct) >= 1 - alpha, averaged over calibration/test draws.

    The guarantee must hold even for an uninformative scorer (it then removes almost everything).
    """
    rng = np.random.default_rng(12345)
    ok = total = 0
    for _ in range(300):
        cs, cc = _simulate(rng, 100, informative)
        ts, tc = _simulate(rng, 50, informative)
        q = C.calibrate_from_responses(cs, cc, alpha)
        for s, c in zip(ts, tc):
            kept = C.keep_mask(s, q)
            ok += all(ci for ci, k in zip(c, kept) if k)
            total += 1
    rate = ok / total
    # 15,000 test responses: binomial s.e. < 0.004, so a 0.015 tolerance is ample
    assert rate >= 1 - alpha - 0.015, rate
    # and the procedure is not vacuous by construction: coverage should not be far above 1 - alpha + 1/(n+1)
    assert rate <= 1 - alpha + 1 / 101 + 0.02, rate


def test_informative_scorer_retains_more_than_uninformative():
    rng = np.random.default_rng(7)
    kept = {}
    for informative in (True, False):
        cs, cc = _simulate(rng, 400, informative)
        ts, _ = _simulate(rng, 400, informative)
        q = C.calibrate_from_responses(cs, cc, alpha=0.2)
        kept[informative] = np.mean([sum(C.keep_mask(s, q)) for s in ts])
    assert kept[True] > kept[False]
