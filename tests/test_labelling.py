import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sc = load("score_labelling")
ms = load("make_labelling_sample")


def test_kappa_known_values():
    a = np.array([1, 1, 0, 0], float)
    assert sc.kappa(a, a) == 1.0
    assert sc.kappa(a, 1 - a) == pytest.approx(-1.0)
    # 2x2 table: both yes 20, a-yes/b-no 5, a-no/b-yes 10, both no 15 -> po = .7, pe = .5*.6 + .5*.4 = .5, kappa = .4
    a = np.array([1] * 20 + [1] * 5 + [0] * 10 + [0] * 15, float)
    b = np.array([1] * 20 + [0] * 5 + [1] * 10 + [0] * 15, float)
    assert sc.kappa(a, b) == pytest.approx(0.4)
    assert np.isnan(sc.kappa(np.ones(3), np.ones(3)))  # degenerate: no variation


def test_stratified_sampling():
    pool = [{"judge_label": i < 50, "i": i} for i in range(100)]
    picked = ms.stratified(pool, 50, random.Random(0))
    assert len(picked) == 50 and sum(p["judge_label"] for p in picked) == 25
    pool = [{"judge_label": i < 80, "i": i} for i in range(100)]   # only 20 negatives: all of them + 30 positives
    picked = ms.stratified(pool, 50, random.Random(0))
    assert len(picked) == 50 and sum(not p["judge_label"] for p in picked) == 20
    few_neg = [{"judge_label": i < 95, "i": i} for i in range(100)]
    picked = ms.stratified(few_neg, 50, random.Random(0))
    assert len(picked) == 50 and sum(not p["judge_label"] for p in picked) == 5   # all 5 negatives, rest positives
    assert len(ms.stratified(pool[:10], 50, random.Random(0))) == 10              # small pool: take everything
