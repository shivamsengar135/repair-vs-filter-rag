"""EXPLORATORY significance tests on the TEST runs (EXPERIMENT_PLAN.md §10: unadjusted, labelled exploratory).

Long-form (each generator x dataset): C vs CF, SelfCheckGPT (each seed set) vs CF, C vs SelfCheckGPT, at every
target. Primary statistic: paired bootstrap (10,000, seed 0) of supported claims / answer on the evaluation half of
partition 0 - the same procedure as the confirmatory A-vs-CF test - plus the micro-precision difference with its
one-sided 95% lower bound. Robustness: the paired difference of means on each of the 100 partitions (mean, std,
share of partitions with diff > 0).
Short-form (each generator x set): McNemar (exact) for A vs standard RAG accuracy (plan §10); paired bootstrap of
AURC(C) - AURC(CF) over items (10,000, seed 0; negative = C better).
Writes results/<split>/exploratory_tests.json and prints compact tables.
Usage: python scripts/exploratory_tests.py [--split test]"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from hallu import analysis as A
from hallu import metrics as M
from hallu.runutil import git_commit
from hallu.store import read_jsonl, text_key

JT = "phi-4-FP8-dynamic"
GENS = ["Qwen2.5-7B-Instruct", "Mistral-7B-Instruct-v0.3", "Falcon3-7B-Instruct"]
TARGETS = [0.80, 0.85, 0.90, 0.95]
N_PART = 100


def one(pattern: str) -> Path:
    c = [Path(p) for p in glob.glob(pattern)]
    if len(c) != 1:
        raise SystemExit(f"expected exactly 1 match for {pattern}, got {c}")
    return c[0]


def longform_methods(run: Path) -> dict[str, list]:
    rows = read_jsonl(run / "items.jsonl")
    labels = {r["id"]: r["label"] for r in read_jsonl(run / f"judge__{JT}.jsonl")}
    lab = lambda iid, t: labels[text_key(iid, t)]  # noqa: E731
    m = {"CF": [[(c["s"], lab(r["id"], c["text"])) for c in r["claims"]] for r in rows],
         "C": [[(c["score_c"], lab(r["id"], c["text"])) for c in r["claims"]] for r in rows]}
    for k, p in enumerate([run / "selfcheck_nli.jsonl"] + [run / f"selfcheck_nli_seedset{i}.jsonl" for i in (1, 2)]):
        if p.exists():
            sc = {r["id"]: r["claim_scores"] for r in read_jsonl(p)}
            m[f"SelfCheck{k}"] = [[(1.0 - s, lab(r["id"], c["text"])) for c, s in zip(r["claims"], sc[r["id"]])] for r in rows]
    return m


def compare(ma: list, mb: list, alpha: float) -> dict:
    ra, rb = A.evaluate_once(ma, alpha, seed=0), A.evaluate_once(mb, alpha, seed=0)
    boot = M.paired_bootstrap(ra["supported_per_response"], rb["supported_per_response"], 10_000, seed=0)
    ev = ra["eval_index"]
    ka = np.array([sum(1 for s, _ in ma[i] if s > ra["q_hat"]) for i in ev], float)
    kb = np.array([sum(1 for s, _ in mb[i] if s > rb["q_hat"]) for i in ev], float)
    sa, sb = ra["supported_per_response"], rb["supported_per_response"]
    idx = np.random.default_rng(0).integers(0, len(ev), size=(10_000, len(ev)))
    with np.errstate(invalid="ignore", divide="ignore"):
        d = sa[idx].sum(1) / ka[idx].sum(1) - sb[idx].sum(1) / kb[idx].sum(1)
    d = d[np.isfinite(d)]
    prec = {"diff": float((sa.sum() / ka.sum() if ka.sum() else np.nan) - (sb.sum() / kb.sum() if kb.sum() else np.nan)),
            "one_sided_95_lower": float(np.quantile(d, 0.05)) if len(d) else float("nan")}
    diffs = []
    for seed in range(N_PART):
        xa, xb = A.evaluate_once(ma, alpha, seed), A.evaluate_once(mb, alpha, seed)
        diffs.append(xa["summary"]["supported_claims_per_response"] - xb["summary"]["supported_claims_per_response"])
    diffs = np.array(diffs)
    return {"supported_per_response": boot, "precision_micro": prec,
            "partitions": {"mean_diff": float(diffs.mean()), "std_diff": float(diffs.std(ddof=1)), "share_positive": float((diffs > 0).mean())}}


def aurc_boot(conf_a, conf_b, correct, n=10_000, seed=0) -> dict:
    ca, cb, y = np.asarray(conf_a, float), np.asarray(conf_b, float), np.asarray(correct, bool)
    obs = M.aurc(ca, y) - M.aurc(cb, y)
    idx = np.random.default_rng(seed).integers(0, len(y), size=(n, len(y)))
    boot = np.array([M.aurc(ca[i], y[i]) - M.aurc(cb[i], y[i]) for i in idx])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    return {"diff": float(obs), "ci_low": float(lo), "ci_high": float(hi), "p_value": float(p), "n": int(len(y))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    out: dict = {"label": "EXPLORATORY (unadjusted; plan §10)", "git_commit": git_commit(), "judge_tag": JT,
                 "longform": {}, "shortform": {}}
    for ds in ("bio", "asqa"):
        for g in GENS:
            run = one(f"results/{a.split}/{ds}/longform__{g}__*/")
            m = longform_methods(run)
            pairs = [("C", "CF")] + [(k, "CF") for k in m if k.startswith("SelfCheck")] + [("C", k) for k in m if k.startswith("SelfCheck")]
            res = {str(t): {f"{x} vs {y}": compare(m[x], m[y], 1 - t) for x, y in pairs} for t in TARGETS}
            out["longform"][f"{ds}/{g}"] = {"run": str(run), "results": res}
            for pair, r in res["0.9"].items():
                b, pp = r["supported_per_response"], r["partitions"]
                print(f"LF {ds:4s} {g:26s} @0.90 {pair:22s} diff {b['diff']:+.3f} [{b['ci_low']:+.3f},{b['ci_high']:+.3f}] p={b['p_value']:.4f} "
                      f"| prec {r['precision_micro']['diff']:+.3f} (lb {r['precision_micro']['one_sided_95_lower']:+.3f}) "
                      f"| partitions {pp['mean_diff']:+.3f}±{pp['std_diff']:.3f}, >0 in {pp['share_positive']:.2f}", flush=True)
    for ds in ("popqa", "nq", "triviaqa", "hotpotqa", "2wiki"):
        for g in GENS:
            run = one(f"results/{a.split}/{ds}/shortform__{g}__*/")
            rows = read_jsonl(run / "items.jsonl")
            mc = M.mcnemar([bool(r["A_match"]) for r in rows], [bool(r["rag_match"]) for r in rows])
            au = aurc_boot([r["score_c"] for r in rows], [r["s"] for r in rows], [bool(r["rag_match"]) for r in rows])
            out["shortform"][f"{ds}/{g}"] = {"run": str(run), "A_vs_RAG_accuracy_mcnemar": mc, "AURC_C_minus_CF": au,
                                             "acc_A": float(np.mean([bool(r["A_match"]) for r in rows])),
                                             "acc_RAG": float(np.mean([bool(r["rag_match"]) for r in rows]))}
            print(f"SF {ds:8s} {g:26s} A-RAG acc {np.mean([bool(r['A_match']) for r in rows]) - np.mean([bool(r['rag_match']) for r in rows]):+.3f} "
                  f"(A right/RAG wrong {mc['n10']}, A wrong/RAG right {mc['n01']}, McNemar p={mc['p_value']:.4f}) "
                  f"| AURC C-CF {au['diff']:+.3f} [{au['ci_low']:+.3f},{au['ci_high']:+.3f}] p={au['p_value']:.4f}", flush=True)
    p = Path(f"results/{a.split}/exploratory_tests.json")
    p.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", p)


if __name__ == "__main__":
    main()
