"""Judged comparison of every long-form method's FINAL answers on one dataset/split (no filtering for baselines).

Rows: standard RAG (all claims of the long-form run), CoVe (both variants), RARR-style, FLARE, and — if present —
CF / C / A at each target from the long-form run's analysis file. Same 100 seeded evaluation halves for every row.
Usage: python scripts/compare_longform.py --dataset asqa --split dev --judge-tag phi-4-FP8-dynamic [--prompts v2]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from hallu import analysis as A
from hallu.store import read_jsonl, text_key


def latest(pattern: str, prompts: str) -> Path | None:
    cands = []
    for d in glob.glob(pattern):
        s = Path(d) / "summary.json"
        if s.exists():
            cfg = json.loads(s.read_text())["config"]
            # never an ablation / sensitivity run (Phase 6 options), whatever its mtime (fixed 2026-11-19; the existing
            # comparison files were written before any ablation run existed and reference the default runs)
            ablation = cfg.get("k", 5) != 5 or cfg.get("tau_s", 0.5) != 0.5 or cfg.get("retriever", "bm25") != "bm25" or \
                cfg.get("verifier", "MiniCheck-Flan-T5-Large") != "MiniCheck-Flan-T5-Large" or {"triage", "decompose", "ablation_prompts"} & set(cfg)
            if cfg.get("prompts") == prompts and "limit" not in cfg and not ablation:
                cands.append((s.stat().st_mtime, Path(d)))
    return max(cands)[1] if cands else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--judge-tag", required=True)
    ap.add_argument("--prompts", default="v2")
    ap.add_argument("--n-partitions", type=int, default=100)
    ap.add_argument("--model-tag", default="Qwen2.5-7B-Instruct", help="generator whose runs are compared (G1 by default)")
    a = ap.parse_args()
    base = f"results/{a.split}/{a.dataset}"
    t = a.model_tag
    runs = {"standard RAG (all claims)": latest(f"{base}/longform__{t}__*", a.prompts),
            "CoVe factor+revise (closed-book)": latest(f"{base}/cove-closed_book__{t}__*", a.prompts),
            "CoVe factor+revise (retrieval-verified)": latest(f"{base}/cove-rag_verify__{t}__*", a.prompts),
            "RARR-style revise-all": latest(f"{base}/rarr__{t}__*", a.prompts),
            "FLARE-direct": latest(f"{base}/flare__{t}__*", a.prompts)}
    out = {"dataset": a.dataset, "split": a.split, "judge_tag": a.judge_tag, "prompts": a.prompts, "rows": {}}
    cols = ["supported_claims_per_response", "claims_kept_per_response", "non_empty_rate", "factual_precision_micro", "all_retained_correct_rate"]
    print(f"{'method':42s} " + " ".join(f"{c[:22]:>24s}" for c in cols))
    for name, run in runs.items():
        if run is None or not (run / f"judge__{a.judge_tag}.jsonl").exists():
            print(f"{name:42s} (missing run or judge file)")
            continue
        lab = {r["id"]: r["label"] for r in read_jsonl(run / f"judge__{a.judge_tag}.jsonl")}
        resp = [[(1.0, lab[text_key(r["id"], c["text"])]) for c in r["claims"]] for r in read_jsonl(run / "items.jsonl")]
        e = A.evaluate(resp, None, a.n_partitions)
        out["rows"][name] = {"run": str(run), **{c: e[c] for c in cols}}
        print(f"{name:42s} " + " ".join(f"{e[c]['mean']:>17.3f}±{e[c]['std']:.3f}" for c in cols))
    lf = runs["standard RAG (all claims)"]
    if lf is not None and (lf / f"analysis__{a.judge_tag}.json").exists():
        an = json.loads((lf / f"analysis__{a.judge_tag}.json").read_text())
        for tg, by in an["targets"].items():  # was `t`, which overwrote the model tag in the output file name
            for name, e in by.items():
                if name.startswith(("CF", "C (", "A (", "SelfCheck")):
                    out["rows"][f"{name} @ target {tg}"] = {"run": str(lf), **{c: e[c] for c in cols}}
                    print(f"{name + ' @' + tg:42s} " + " ".join(f"{e[c]['mean']:>17.3f}±{e[c]['std']:.3f}" for c in cols))
    dst = Path(base) / f"comparison__{t}__{a.prompts}__{a.judge_tag}.json"
    dst.write_text(json.dumps(out, indent=1))
    print("wrote", dst)


if __name__ == "__main__":
    main()
