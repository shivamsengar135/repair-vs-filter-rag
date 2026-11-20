"""Matched-target comparison of standard RAG / CF / C / A / B on one judged long-form run (EXPERIMENT_PLAN.md §1, §6).

Reads <run>/items.jsonl and <run>/judge__<tag>.jsonl. Writes <run>/analysis__<tag>.json and prints a table.
On DEV runs this also reports the pilot quantity R (§1). The hypothesis test printed here is exploratory on dev;
it is confirmatory only when run once on the frozen test split with the frozen configuration.

Usage: python scripts/analyze_longform.py --run results/dev/asqa/longform__... --judge-tag phi-4-FP8-dynamic
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hallu import analysis as A
from hallu import conformal as C
from hallu import metrics as M
from hallu.pipeline import final_claims
from hallu.runutil import git_commit
from hallu.store import read_jsonl, text_key

TARGETS = [0.80, 0.85, 0.90, 0.95]
BUDGETS = [0, 1, 2, 4, None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--judge-tag", required=True)
    ap.add_argument("--n-partitions", type=int, default=100)
    a = ap.parse_args()
    run = Path(a.run)
    rows = read_jsonl(run / "items.jsonl")
    labels = {r["id"]: r["label"] for r in read_jsonl(run / f"judge__{a.judge_tag}.jsonl")}

    def lab(item_id: str, text: str) -> bool:
        return labels[text_key(item_id, text)]

    methods: dict[str, list] = {
        "CF (score s)": [[(c["s"], lab(r["id"], c["text"])) for c in r["claims"]] for r in rows],
        "C (two-signal)": [[(c["score_c"], lab(r["id"], c["text"])) for c in r["claims"]] for r in rows],
    }
    for b in BUDGETS:
        name = "A (repair all)" if b is None else f"B (budget {b})"
        methods[name] = [[(fc["score"], lab(r["id"], fc["text"])) for fc in final_claims(r["claims"], b)] for r in rows]
    # SelfCheckGPT-NLI baseline: same claims, score = 1 - mean P(contradiction with samples); one row per seed set
    for k, sc_path in enumerate([run / "selfcheck_nli.jsonl"] + [run / f"selfcheck_nli_seedset{i}.jsonl" for i in (1, 2)]):
        if sc_path.exists():
            sc = {r["id"]: r["claim_scores"] for r in read_jsonl(sc_path)}
            name = "SelfCheckGPT-NLI" if k == 0 else f"SelfCheckGPT-NLI (seed set {k})"
            methods[name] = [[(1.0 - s, lab(r["id"], c["text"])) for c, s in zip(r["claims"], sc[r["id"]])] for r in rows]

    out: dict = {"run": str(run), "judge_tag": a.judge_tag, "n_responses": len(rows), "git_commit": git_commit(),
                 "n_partitions": a.n_partitions, "cal_frac": 0.5, "unfiltered": {}, "targets": {}}
    out["unfiltered"]["standard RAG (all claims)"] = A.evaluate(methods["CF (score s)"], None, a.n_partitions)
    out["unfiltered"]["A without filter"] = A.evaluate(methods["A (repair all)"], None, a.n_partitions)
    for t in TARGETS:
        out["targets"][str(t)] = {name: A.evaluate(resp, 1 - t, a.n_partitions) for name, resp in methods.items()}

    # ---- paired comparison A vs CF at target 0.90, partition 0 (plan §1): primary endpoint + non-inferiority guard
    ra = A.evaluate_once(methods["A (repair all)"], 0.10, seed=0)
    rc = A.evaluate_once(methods["CF (score s)"], 0.10, seed=0)
    boot = M.paired_bootstrap(ra["supported_per_response"], rc["supported_per_response"], 10_000, seed=0)
    ev = ra["eval_index"]

    def micro_prec_boot(seed: int = 0, n: int = 10_000) -> dict:
        ka = np.array([sum(1 for s, _ in methods["A (repair all)"][i] if s > ra["q_hat"]) for i in ev], float)
        kc = np.array([sum(1 for s, _ in methods["CF (score s)"][i] if s > rc["q_hat"]) for i in ev], float)
        sa, sc = ra["supported_per_response"], rc["supported_per_response"]
        idx = np.random.default_rng(seed).integers(0, len(ev), size=(n, len(ev)))
        with np.errstate(invalid="ignore", divide="ignore"):
            d = sa[idx].sum(1) / ka[idx].sum(1) - sc[idx].sum(1) / kc[idx].sum(1)
        d = d[np.isfinite(d)]
        point = (sa.sum() / ka.sum() if ka.sum() else float("nan")) - (sc.sum() / kc.sum() if kc.sum() else float("nan"))
        return {"diff": float(point), "one_sided_95_lower": float(np.quantile(d, 0.05)) if len(d) else float("nan"), "n_valid_resamples": int(len(d))}

    guard = micro_prec_boot()
    out["A_vs_CF_at_0.90"] = {"supported_per_response": boot, "precision_micro_diff": guard,
                              "non_inferiority_margin": -0.02, "guard_holds": bool(guard["one_sided_95_lower"] > -0.02)}

    # ---- pilot R (dev only): of the claims CF would drop at 0.90, how many does repair turn into a judge-supported claim?
    q_cf = C.calibrate_from_responses([[s for s, _ in r] for r in methods["CF (score s)"]], [[l for _, l in r] for r in methods["CF (score s)"]], 0.10)
    dropped = attempted = recovered = dropped_true = rec_net = rec_kept = 0
    for r in rows:
        for c in r["claims"]:
            if not (c["s"] > q_cf):
                dropped += 1
                orig_true = lab(r["id"], c["text"])
                dropped_true += orig_true
                rep = c.get("repair")
                if rep is not None:
                    attempted += 1
                    if rep.get("text") and lab(r["id"], rep["text"]):
                        recovered += 1
                        rec_net += not orig_true
                        rec_kept += rep["s_after"] > q_cf
    out["pilot"] = {"q_cf_full_dev_at_0.90": q_cf, "claims_dropped_by_CF": dropped, "of_which_judge_supported_originally": dropped_true,
                    "repair_attempted": attempted, "repair_output_judge_supported": recovered,
                    "R": (recovered / dropped) if dropped else float("nan"),
                    "R_net": (rec_net / dropped) if dropped else float("nan"),
                    "R_kept": (rec_kept / dropped) if dropped else float("nan"),
                    "note": "R = preregistered definition (repair output judge-supported); R_net = original was unsupported; "
                            "R_kept = also passes the CF threshold after repair"}

    (run / f"analysis__{a.judge_tag}.json").write_text(json.dumps(out, indent=1, default=float))

    def fmt(e: dict, k: str) -> str:
        return f"{e[k]['mean']:.3f}±{e[k]['std']:.3f}"

    cols = ["supported_claims_per_response", "claims_kept_per_response", "non_empty_rate", "all_retained_correct_rate", "factual_precision_micro"]
    print(f"run: {run}  (n={len(rows)}, {a.n_partitions} partitions, mean±std)")
    print(f"{'method':28s} " + " ".join(f"{c[:24]:>26s}" for c in cols))
    for name, e in out["unfiltered"].items():
        print(f"{name:28s} " + " ".join(f"{fmt(e, c):>26s}" for c in cols))
    for t in TARGETS:
        print(f"--- target {t}")
        for name, e in out["targets"][str(t)].items():
            print(f"{name:28s} " + " ".join(f"{fmt(e, c):>26s}" for c in cols))
    print("A vs CF @0.90, partition 0:", json.dumps(out["A_vs_CF_at_0.90"], default=float))
    print("pilot:", json.dumps(out["pilot"], default=float))


if __name__ == "__main__":
    main()
