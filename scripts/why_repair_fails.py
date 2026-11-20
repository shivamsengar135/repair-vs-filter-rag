"""Phase 5 diagnosis on DEV data only (EXPERIMENT_PLAN.md Phase 5: "if our method underperforms, analyse why"):
why does Triage-and-Repair (A) not beat filter-only (CF)?

For a judged long-form dev run of G1:
 1. Ranking quality of each claim score against judge labels (AUROC): s (CF), score_c (C), 1 − SelfCheckGPT
    contradiction, closed-book confidence c alone. Retention at a matched target depends on this ranking.
 2. The CF threshold at target 0.90 and how many claims sit above it.
 3. The revise branch: how many "contradicted" claims the judge actually labels unsupported; how often revision
    returned DELETE; the judge label of revised texts.
 4. The re-retrieve branch: change in support score, how many cross the CF threshold, judge labels.
Usage: python scripts/why_repair_fails.py --run results/dev/bio/longform__... --judge-tag phi-4-FP8-dynamic
"""
import argparse
import json
from pathlib import Path

import numpy as np

from hallu import conformal as C
from hallu.metrics import auroc
from hallu.store import read_jsonl, text_key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--judge-tag", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    rows = read_jsonl(run / "items.jsonl")
    lab = {r["id"]: r["label"] for r in read_jsonl(run / f"judge__{a.judge_tag}.jsonl")}
    sc = {r["id"]: r["claim_scores"] for r in read_jsonl(run / "selfcheck_nli.jsonl")} if (run / "selfcheck_nli.jsonl").exists() else {}
    cl = [(r["id"], c, (1 - sc[r["id"]][i]) if r["id"] in sc else None) for r in rows for i, c in enumerate(r["claims"])]
    y = [lab[text_key(i, c["text"])] for i, c, _ in cl]
    out = {"run": str(run), "n_claims": len(cl), "judge_supported_rate": float(np.mean(y)),
           "auroc_vs_judge": {"s (CF score: MiniCheck support)": auroc([c["s"] for _, c, _ in cl], y),
                              "score_c (C: two-signal)": auroc([c["score_c"] for _, c, _ in cl], y),
                              "c (closed-book confidence alone)": auroc([c["c"] for _, c, _ in cl], y),
                              **({"1 - SelfCheckGPT contradiction": auroc([x for _, _, x in cl], y)} if sc else {})}}
    resp = [[(c["s"], lab[text_key(r["id"], c["text"])]) for c in r["claims"]] for r in rows]
    q = C.calibrate_from_responses([[s for s, _ in x] for x in resp], [[l for _, l in x] for x in resp], 0.10)
    out["cf_threshold_at_0.90"] = q
    out["claims_above_cf_threshold"] = int(sum(c["s"] > q for _, c, _ in cl))
    s_arr = np.array([c["s"] for _, c, _ in cl])
    out["support_score_quantiles"] = {str(p): float(np.quantile(s_arr, p)) for p in (0.1, 0.25, 0.5, 0.75, 0.9)}
    rev = [(i, c) for i, c, _ in cl if c["action"] == "revise"]
    rer = [(i, c) for i, c, _ in cl if c["action"] == "reretrieve"]
    out["revise"] = {"n": len(rev), "original_judge_supported": int(sum(lab[text_key(i, c["text"])] for i, c in rev)),
                     "returned_DELETE": int(sum(c["repair"]["text"] is None for _, c in rev)),
                     "revised_text_judge_supported": int(sum(1 for i, c in rev if c["repair"]["text"] and lab.get(text_key(i, c["repair"]["text"]), False)))}
    out["reretrieve"] = {"n": len(rer), "original_judge_supported": int(sum(lab[text_key(i, c["text"])] for i, c in rer)),
                         "support_increased": int(sum(c["repair"]["s_after"] > c["s"] for _, c in rer)),
                         "crossed_cf_threshold": int(sum(c["repair"]["s_after"] > q for _, c in rer)),
                         "median_s_before": float(np.median([c["s"] for _, c in rer])) if rer else None,
                         "median_s_after": float(np.median([c["repair"]["s_after"] for _, c in rer])) if rer else None}
    low = [(i, c) for i, c, _ in cl if c["action"] == "low"]
    out["low"] = {"n": len(low), "judge_supported": int(sum(lab[text_key(i, c["text"])] for i, c in low))}
    keep = [(i, c) for i, c, _ in cl if c["action"] == "keep"]
    out["keep"] = {"n": len(keep), "judge_supported": int(sum(lab[text_key(i, c["text"])] for i, c in keep))}
    dst = run / f"why_repair_fails__{a.judge_tag}.json"
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
