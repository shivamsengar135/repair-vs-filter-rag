"""Confirmatory tests H3-H5 (docs/PREREG_C_CONFIRMATORY.md) — run ONCE, after all three bio_confirm runs are judged.

For each generator (H3 = Qwen2.5-7B, H4 = Mistral-7B, H5 = Falcon3-7B): C vs CF at target 0.90 on the bio_confirm
run; primary endpoint = judge-supported claims per answer; paired bootstrap over answers (10,000 resamples, seed 0)
on the evaluation half of calibration/evaluation partition 0 (calibration fraction 0.5) — the procedure and code of
the H1/H2 test and of the exploratory C-vs-CF tests (scripts/exploratory_tests.py: compare, longform_methods).
Two-sided alpha = 0.05, Holm over {H3, H4, H5}. A hypothesis is SUPPORTED only if Holm rejects AND the difference is
positive AND the precision guard holds (one-sided 95% lower bound of C - CF micro precision > -0.02).
Also reported (descriptive, not tested): the 100-partition mean/std/share>0, the other targets, AUROC of s, c and C.
Writes results/confirm/confirmatory_c.json. Usage: python scripts/confirmatory_c.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from exploratory_tests import JT, compare, longform_methods, one  # noqa: E402  (identical procedure)
from hallu.metrics import holm  # noqa: E402
from hallu.runutil import git_commit  # noqa: E402
from hallu.store import read_jsonl, text_key  # noqa: E402

HYPS = {"H3": "Qwen2.5-7B-Instruct", "H4": "Mistral-7B-Instruct-v0.3", "H5": "Falcon3-7B-Instruct"}
MARGIN = -0.02


def auroc(score: list[float], y: list[bool]) -> float:
    s, y = np.asarray(score, float), np.asarray(y, bool)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="confirm", help="'test' only to check the code reproduces the exploratory numbers")
    split = ap.parse_args().split
    # code check on test: the three DEFAULT test runs (results/test also holds ablation runs)
    TEST_DEFAULT = {"Qwen2.5-7B-Instruct": "5aa2ad1482e5", "Mistral-7B-Instruct-v0.3": "3f20b417147d", "Falcon3-7B-Instruct": "0b9ff4861d3b"}
    res = {}
    for h, g in HYPS.items():
        run = one(f"results/{split}/bio/longform__{g}__{TEST_DEFAULT[g] if split == 'test' else '*'}/")
        summ = json.loads((run / "summary.json").read_text())
        if "limit" in json.dumps(summ["config"]) and summ["config"].get("limit"):
            raise SystemExit(f"{run} is a limited run")
        m = longform_methods(run)
        rows = read_jsonl(run / "items.jsonl")
        lab = {r["id"]: r["label"] for r in read_jsonl(run / f"judge__{JT}.jsonl")}
        claims = [(c, lab[text_key(r["id"], c["text"])]) for r in rows for c in r["claims"]]
        r9 = compare(m["C"], m["CF"], 0.10)
        b = r9["supported_per_response"]
        res[h] = {"generator": g, "run": str(run), "n_answers": len(rows), "n_claims": len(claims),
                  "n_eval": b["n"], "diff_supported_per_answer": b["diff"], "ci95": [b["ci_low"], b["ci_high"]],
                  "p_value": b["p_value"], "precision_diff": r9["precision_micro"]["diff"],
                  "precision_lower_bound_95": r9["precision_micro"]["one_sided_95_lower"],
                  "guard_holds": bool(r9["precision_micro"]["one_sided_95_lower"] > MARGIN),
                  "partitions_0.90": r9["partitions"],
                  "other_targets_descriptive": {str(t): compare(m["C"], m["CF"], 1 - t)["supported_per_response"] for t in (0.80, 0.85, 0.95)},
                  "auroc": {"s": auroc([c["s"] for c, _ in claims], [y for _, y in claims]),
                            "c": auroc([c["c"] for c, _ in claims], [y for _, y in claims]),
                            "C": auroc([c["score_c"] for c, _ in claims], [y for _, y in claims])},
                  "judge_supported_rate": float(np.mean([y for _, y in claims]))}
    rej = holm([res[h]["p_value"] for h in HYPS], alpha=0.05)
    for h, r in zip(HYPS, rej):
        res[h]["holm_reject"] = bool(r)
        res[h]["SUPPORTED"] = bool(r and res[h]["diff_supported_per_answer"] > 0 and res[h]["guard_holds"])
    out = {"prereg": "docs/PREREG_C_CONFIRMATORY.md", "alpha": 0.05, "correction": "Holm over {H3, H4, H5}",
           "bootstrap": "paired, 10,000 resamples, seed 0, partition 0, calibration fraction 0.5", "target": 0.90,
           "judge_tag": JT, "git_commit": git_commit(), "results": res}
    p = Path(f"results/{split}/confirmatory_c.json" if split == "confirm" else f"results/{split}/confirmatory_c_codecheck.json")
    p.write_text(json.dumps(out, indent=1, default=float))
    for h in HYPS:
        r = res[h]
        print(f"{h} {r['generator']}: C-CF supported/answer {r['diff_supported_per_answer']:+.3f} "
              f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] p={r['p_value']:.4f} Holm={r['holm_reject']} | precision diff "
              f"{r['precision_diff']:+.3f} (lb {r['precision_lower_bound_95']:+.3f}, guard {r['guard_holds']}) -> SUPPORTED={r['SUPPORTED']} "
              f"| n_eval {r['n_eval']} | AUROC s {r['auroc']['s']:.3f} c {r['auroc']['c']:.3f} C {r['auroc']['C']:.3f}")
    print("wrote", p)


if __name__ == "__main__":
    main()
