"""Claim-level discrimination on TEST (EXPLORATORY): AUROC of each score against the judge label, per generator x
long-form dataset, with a 95% bootstrap CI over responses (claims of one answer are resampled together, 2,000
resamples, seed 0). Scores: s (MiniCheck support), score_c (C), c (closed-book P(True)), 1 - con (NLI), and
SelfCheckGPT-NLI (G1 only, seed set 0). Writes results/<split>/claim_auroc.json + prints a table.
Usage: python scripts/claim_auroc.py [--split test]"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from hallu import metrics as M
from hallu.runutil import git_commit
from hallu.store import read_jsonl, text_key

JT = "phi-4-FP8-dynamic"
GENS = ["Qwen2.5-7B-Instruct", "Mistral-7B-Instruct-v0.3", "Falcon3-7B-Instruct"]


def is_default(cfg: dict) -> bool:
    """The frozen main configuration (no ablation / sensitivity option set)."""
    return (cfg.get("prompts") == "v2" and cfg.get("k") == 5 and cfg.get("tau_s") == 0.5 and cfg.get("retriever") == "bm25"
            and cfg.get("verifier") == "MiniCheck-Flan-T5-Large" and not ({"triage", "decompose", "ablation_prompts", "limit"} & set(cfg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    out = {"label": "EXPLORATORY", "git_commit": git_commit(), "judge_tag": JT, "results": {}}
    for ds in ("bio", "asqa"):
        for g in GENS:
            runs = [r for r in glob.glob(f"results/{a.split}/{ds}/longform__{g}__*/")
                    if (Path(r) / "summary.json").exists() and is_default(json.loads((Path(r) / "summary.json").read_text())["config"])]
            if len(runs) != 1:
                raise SystemExit(f"{ds}/{g}: expected one default run, got {runs}")
            run = Path(runs[0])
            rows = read_jsonl(run / "items.jsonl")
            labels = {r["id"]: r["label"] for r in read_jsonl(run / f"judge__{JT}.jsonl")}
            sc_path = run / "selfcheck_nli.jsonl"
            sc = {r["id"]: r["claim_scores"] for r in read_jsonl(sc_path)} if sc_path.exists() else None
            per_resp = []
            for r in rows:
                cl = r["claims"]
                y = [labels[text_key(r["id"], c["text"])] for c in cl]
                feats = {"s": [c["s"] for c in cl], "score_c (C)": [c["score_c"] for c in cl], "c": [c["c"] for c in cl],
                         "1 - con": [1 - c["con"] for c in cl]}
                if sc is not None:
                    feats["SelfCheckGPT-NLI"] = [1 - v for v in sc[r["id"]]]
                per_resp.append((y, feats))
            names = list(per_resp[0][1]) if per_resp else []
            res = {"run": str(run), "n_claims": int(sum(len(y) for y, _ in per_resp)),
                   "supported_rate": float(np.mean([v for y, _ in per_resp for v in y]))}
            rng = np.random.default_rng(0)
            idx = rng.integers(0, len(per_resp), size=(2000, len(per_resp)))
            for n in names:
                ys = np.array([v for y, _ in per_resp for v in y], bool)
                xs = np.array([v for _, f in per_resp for v in f[n]], float)
                point = M.auroc(xs, ys)
                lens = np.array([len(y) for y, _ in per_resp])
                starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
                boots = []
                for b in idx[:2000]:
                    sel = np.concatenate([np.arange(starts[i], starts[i] + lens[i]) for i in b]) if len(b) else np.array([], int)
                    boots.append(M.auroc(xs[sel], ys[sel]))
                lo, hi = np.nanquantile(boots, [0.025, 0.975])
                # claim-level calibration (plan §6): ECE, 10 equal-width bins, score read as P(supported)
                res[n] = {"auroc": point, "ci95": [float(lo), float(hi)], "ece": M.ece(xs, ys)}
            out["results"][f"{ds}/{g}"] = res
            print(f"{ds:4s} {g:26s} n={res['n_claims']:5d} sup={res['supported_rate']:.3f} | " +
                  " | ".join(f"{n} {res[n]['auroc']:.3f} [{res[n]['ci95'][0]:.3f},{res[n]['ci95'][1]:.3f}] ECE {res[n]['ece']:.3f}" for n in names), flush=True)
    Path(f"results/{a.split}/claim_auroc.json").write_text(json.dumps(out, indent=1, default=float))
    print("wrote", f"results/{a.split}/claim_auroc.json")


if __name__ == "__main__":
    main()
