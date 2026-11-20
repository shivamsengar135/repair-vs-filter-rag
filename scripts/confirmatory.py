"""Confirmatory tests H1 (biographies) and H2 (ASQA), EXPERIMENT_PLAN.md §1 and §10 — run ONCE on the test split.

For each hypothesis: A vs CF at target 0.90 on G1's test run, primary endpoint = judge-supported claims per response,
paired bootstrap over questions (10,000 resamples, seed 0) on calibration/evaluation partition 0 (cal fraction 0.5),
two-sided α = 0.05, Holm across {H1, H2}. Non-inferiority guard: one-sided 95% lower bound of A − CF claim-level
factual precision (micro) > −0.02. A hypothesis is SUPPORTED only if Holm rejects AND the difference is positive
AND the guard holds. Reads the analysis files written by scripts/analyze_longform.py.

Usage: python scripts/confirmatory.py --judge-tag phi-4-FP8-dynamic
"""
import argparse
import glob
import json
from pathlib import Path

from hallu.metrics import holm
from hallu.runutil import git_commit


def g1_run(ds: str) -> Path:
    c = []
    for d in glob.glob(f"results/test/{ds}/longform__Qwen2.5-7B-Instruct__*"):
        s = Path(d) / "summary.json"
        if s.exists() and '"limit"' not in s.read_text() and json.loads(s.read_text())["config"].get("prompts") == "v2":
            c.append((s.stat().st_mtime, Path(d)))
    if not c:
        raise SystemExit(f"no G1 test run for {ds}")
    return max(c)[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-tag", required=True)
    a = ap.parse_args()
    hyps = {"H1": "bio", "H2": "asqa"}
    res = {}
    for h, ds in hyps.items():
        run = g1_run(ds)
        an = json.loads((run / f"analysis__{a.judge_tag}.json").read_text())
        cmp_ = an["A_vs_CF_at_0.90"]
        res[h] = {"dataset": ds, "run": str(run), "n_eval": cmp_["supported_per_response"]["n"],
                  "diff_supported_per_response": cmp_["supported_per_response"]["diff"],
                  "ci95": [cmp_["supported_per_response"]["ci_low"], cmp_["supported_per_response"]["ci_high"]],
                  "p_value": cmp_["supported_per_response"]["p_value"],
                  "precision_diff": cmp_["precision_micro_diff"]["diff"],
                  "precision_lower_bound_95": cmp_["precision_micro_diff"]["one_sided_95_lower"],
                  "guard_holds": cmp_["guard_holds"],
                  "A_at_0.90": an["targets"]["0.9"]["A (repair all)"], "CF_at_0.90": an["targets"]["0.9"]["CF (score s)"]}
    rej = holm([res[h]["p_value"] for h in hyps], alpha=0.05)
    for h, r in zip(hyps, rej):
        res[h]["holm_reject"] = bool(r)
        res[h]["SUPPORTED"] = bool(r and res[h]["diff_supported_per_response"] > 0 and res[h]["guard_holds"])
    out = {"alpha": 0.05, "correction": "Holm over {H1, H2}", "bootstrap": "paired, 10,000 resamples, seed 0, partition 0",
           "judge_tag": a.judge_tag, "git_commit": git_commit(), "results": res}
    Path("results/test/confirmatory.json").write_text(json.dumps(out, indent=1))
    for h in hyps:
        r = res[h]
        print(f"{h} ({r['dataset']}): A−CF supported claims/resp = {r['diff_supported_per_response']:+.3f} "
              f"95% CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}], p = {r['p_value']:.4f}, Holm reject = {r['holm_reject']}; "
              f"precision diff {r['precision_diff']:+.3f} (lower bound {r['precision_lower_bound_95']:+.3f}, guard {r['guard_holds']}) "
              f"→ SUPPORTED = {r['SUPPORTED']}")
    print("wrote results/test/confirmatory.json")


if __name__ == "__main__":
    main()
