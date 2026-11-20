import numpy as np
import pytest

from hallu import analysis as A


def test_partition_is_seeded_disjoint_and_complete():
    cal, ev = A.partition(11, seed=3)
    cal2, ev2 = A.partition(11, seed=3)
    assert (cal == cal2).all() and (ev == ev2).all()
    assert len(cal) == 6 and len(ev) == 5  # round(5.5) = 6 (numpy/py round-half-even gives 6)
    assert set(cal) | set(ev) == set(range(11)) and not set(cal) & set(ev)
    assert not (A.partition(11, 3)[0] == A.partition(11, 4)[0]).all()


def test_apply_threshold_and_stats():
    resp = [(0.9, True), (0.6, False), (0.3, True)]
    assert A.apply_threshold(resp, 0.6) == [True]          # strict
    assert A.apply_threshold(resp, float("-inf")) == [True, False, True]
    s = A.per_response_stats([True, False, True])
    assert s == {"supported": 2, "kept": 3, "non_empty": 1.0, "all_correct": 0.0, "precision": pytest.approx(2 / 3)}
    e = A.per_response_stats([])
    assert e["kept"] == 0 and e["non_empty"] == 0.0 and e["all_correct"] == 1.0 and np.isnan(e["precision"])


def _responses(rng, n, quality):
    out = []
    for _ in range(n):
        m = int(rng.integers(3, 12))
        lab = rng.random(m) < 0.75
        sc = rng.normal(np.where(lab, quality, 0.0), 1.0, size=m)
        out.append(list(zip(sc.tolist(), lab.tolist())))
    return out


def test_no_filter_keeps_everything_on_the_eval_half():
    resp = _responses(np.random.default_rng(0), 40, 2.0)
    r = A.evaluate_once(resp, alpha=None, seed=0)
    ev = r["eval_index"]
    assert r["summary"]["claims_kept_per_response"] == pytest.approx(np.mean([len(resp[i]) for i in ev]))
    assert r["summary"]["supported_claims_per_response"] == pytest.approx(np.mean([sum(l for _, l in resp[i]) for i in ev]))
    assert r["summary"]["non_empty_rate"] == 1.0


def test_filter_reaches_target_and_better_scorer_keeps_more():
    rng = np.random.default_rng(1)
    good, bad = _responses(rng, 600, 3.0), _responses(rng, 600, 0.3)
    eg, eb = A.evaluate(good, alpha=0.1, n_partitions=40), A.evaluate(bad, alpha=0.1, n_partitions=40)
    for e in (eg, eb):
        assert e["all_retained_correct_rate"]["mean"] >= 0.9 - 0.03
    assert eg["supported_claims_per_response"]["mean"] > 2 * eb["supported_claims_per_response"]["mean"]
    assert eg["non_empty_rate"]["mean"] > eb["non_empty_rate"]["mean"]
    assert eg["n_partitions"] == 40 and eg["supported_claims_per_response"]["std"] > 0


def test_same_seed_gives_paired_eval_sets_across_methods():
    rng = np.random.default_rng(2)
    a, b = _responses(rng, 50, 2.0), _responses(rng, 50, 1.0)
    ra, rb = A.evaluate_once(a, 0.2, seed=7), A.evaluate_once(b, 0.2, seed=7)
    assert (ra["eval_index"] == rb["eval_index"]).all()
    assert len(ra["supported_per_response"]) == len(rb["supported_per_response"]) == 25


def test_original_retention():
    resp = [[(0.9, True), (0.1, False)], [(0.5, True)], []]
    assert A.original_retention(resp, 0.4) == pytest.approx(2 / 3)
    assert A.original_retention(resp, float("-inf")) == 1.0
    assert np.isnan(A.original_retention([[]], 0.0))
