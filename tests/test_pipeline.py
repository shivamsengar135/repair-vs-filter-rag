import pytest

from hallu.pipeline import KEEP, LOW, RERETRIEVE, REVISE, Thresholds, final_claims, repairs_applied, score_two_signal, triage

TH = Thresholds(0.5, 0.5, 0.5)


def test_triage_order_of_tests():
    assert triage(0.9, 0.9, 0.0, TH) == KEEP          # supported wins even if another passage contradicts
    assert triage(0.2, 0.8, 0.9, TH) == REVISE        # contradicted beats confident
    assert triage(0.2, 0.1, 0.7, TH) == RERETRIEVE    # unsupported, not contradicted, model is confident
    assert triage(0.2, 0.1, 0.3, TH) == LOW
    assert triage(0.5, 0.0, 0.0, TH) == KEEP          # thresholds are inclusive
    assert triage(0.49, 0.5, 0.0, TH) == REVISE
    assert triage(0.49, 0.49, 0.5, TH) == RERETRIEVE


def test_triage_respects_thresholds():
    assert triage(0.6, 0.0, 0.0, Thresholds(tau_s=0.7)) == LOW
    assert triage(0.2, 0.6, 0.0, Thresholds(tau_con=0.7)) == LOW
    assert triage(0.2, 0.0, 0.6, Thresholds(tau_c=0.7)) == LOW


def test_score_two_signal():
    assert score_two_signal(1.0, 0.0) == 1.0
    assert score_two_signal(0.0, 0.3) == pytest.approx(0.3)
    assert score_two_signal(0.5, 0.5) == pytest.approx(0.75)
    assert score_two_signal(0.2, 0.9) > score_two_signal(0.2, 0.1)


CLAIMS = [
    {"text": "a", "s": 0.9, "action": KEEP},
    {"text": "b", "s": 0.3, "action": REVISE, "repair": {"text": "b2", "s_after": 0.8}},
    {"text": "c", "s": 0.1, "action": RERETRIEVE, "repair": {"text": "c", "s_after": 0.7}},
    {"text": "d", "s": 0.2, "action": REVISE, "repair": {"text": None, "s_after": 0.0}},
    {"text": "e", "s": 0.05, "action": LOW},
    {"text": "f", "s": 0.4, "action": RERETRIEVE, "repair": {"text": "f", "s_after": 0.1}},
]


def test_final_claims_full_repair():
    out = final_claims(CLAIMS)
    assert [(o["text"], o["origin"]) for o in out] == [("a", "original"), ("b2", "revised"), ("c", "reretrieved"),
                                                       ("e", "original"), ("f", "reretrieved")]
    assert [o["score"] for o in out] == [0.9, 0.8, 0.7, 0.05, 0.4]  # re-retrieval never lowers a score; "d" deleted
    assert repairs_applied(CLAIMS) == 4


def test_final_claims_budget_takes_riskiest_first():
    # attempted repairs by ascending s: c (0.1), d (0.2), b (0.3), f (0.4)
    out1 = final_claims(CLAIMS, budget=1)
    assert [(o["text"], o["score"]) for o in out1] == [("a", 0.9), ("b", 0.3), ("c", 0.7), ("d", 0.2), ("e", 0.05), ("f", 0.4)]
    out2 = final_claims(CLAIMS, budget=2)
    assert [o["text"] for o in out2] == ["a", "b", "c", "e", "f"]  # d's deletion now applied
    assert final_claims(CLAIMS, budget=0) == [{"text": c["text"], "score": c["s"], "origin": "original", "source_index": i}
                                               for i, c in enumerate(CLAIMS)]
    assert final_claims(CLAIMS, budget=99) == final_claims(CLAIMS)
    assert [repairs_applied(CLAIMS, b) for b in (0, 1, 4, 99)] == [0, 1, 4, 4]


def test_final_claims_ignores_unattempted_repairs_and_empty():
    cl = [{"text": "x", "s": 0.1, "action": RERETRIEVE}]  # repair stage not run (e.g. baseline CF)
    assert final_claims(cl) == [{"text": "x", "score": 0.1, "origin": "original", "source_index": 0}]
    assert final_claims([]) == []


def test_one_axis_triage_has_no_low_bucket():
    assert triage(0.2, 0.1, 0.3, TH, one_axis=True) == RERETRIEVE  # would be LOW with the confidence axis
    assert triage(0.2, 0.8, 0.0, TH, one_axis=True) == REVISE
    assert triage(0.9, 0.0, 0.0, TH, one_axis=True) == KEEP


def test_final_claims_branch_restriction():
    claims = [
        {"text": "a", "s": 0.9, "action": KEEP},
        {"text": "b", "s": 0.1, "action": REVISE, "repair": {"text": "b2", "s_after": 0.8}},
        {"text": "c", "s": 0.2, "action": RERETRIEVE, "repair": {"text": "c", "s_after": 0.7}},
    ]
    rev_only = final_claims(claims, branches=(REVISE,))
    assert [(c["text"], c["score"], c["origin"]) for c in rev_only] == [("a", 0.9, "original"), ("b2", 0.8, "revised"), ("c", 0.2, "original")]
    rer_only = final_claims(claims, branches=(RERETRIEVE,))
    assert [(c["text"], c["score"], c["origin"]) for c in rer_only] == [("a", 0.9, "original"), ("b", 0.1, "original"), ("c", 0.7, "reretrieved")]
    both = final_claims(claims)
    assert [c["origin"] for c in both] == ["original", "revised", "reretrieved"]
