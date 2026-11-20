"""Judge validation 1 (EXPERIMENT_PLAN.md §7): our judge vs FActScore's released human labels.

Data: data/labeled/{InstructGPT,ChatGPT,PerplexityAI}.jsonl — human-atomic-facts labelled S (supported),
NS (not supported) or IR (irrelevant). IR facts are excluded, as in FActScore's own evaluation of factual precision.
Each fact is judged with the same evidence and prompt as our biography judge (hallu.factscore_kb).
Reports per system: Cohen's kappa (bootstrap 95% CI), raw agreement, error rate of the estimated FActScore
(|judge FActScore - human FActScore|, the measure FActScore's paper uses), and confusion counts.

Usage: python scripts/validate_judge_factscore.py --judge-model RedHatAI/phi-4-FP8-dynamic
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hallu.factscore_kb import FactScoreKB, factscore_prompt
from hallu.llm import LLM, Request, binary_probability, user
from hallu.runutil import CACHE_DIR, DATA_ROOT, git_commit

SYSTEMS = ["InstructGPT", "ChatGPT", "PerplexityAI"]


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--concurrency", type=int, default=32)
    a = ap.parse_args()
    kb = FactScoreKB()
    llm = LLM(model=a.judge_model, cache_path=CACHE_DIR / f"llm__{a.judge_model.split('/')[-1]}.jsonl", concurrency=a.concurrency)
    out = {"judge_model": a.judge_model, "evidence": "FActScore KB, BM25 top-5 within the topic page", "git_commit": git_commit(), "systems": {}}
    t0 = time.perf_counter()
    pooled_h, pooled_j = [], []
    for sysname in SYSTEMS:
        rows = [json.loads(l) for l in (DATA_ROOT / f"factscore/data/labeled/{sysname}.jsonl").read_text().splitlines() if l.strip()]
        facts = []   # (topic, fact, human_supported, response index)
        for ri, r in enumerate(rows):
            for sent in (r.get("annotations") or []):
                for f in (sent.get("human-atomic-facts") or []):
                    if f["label"] in ("S", "NS"):
                        facts.append((r["topic"], f["text"], f["label"] == "S", ri))
        reqs = [Request(user(factscore_prompt(t, f, kb.top_k(t, f, k=5))), max_tokens=1, top_logprobs=20, tag="validate_factscore") for t, f, _, _ in facts]
        outs = llm.chat(reqs)
        probs = [binary_probability(o.get("first_token_top", []), ["true", "yes"], ["false", "no"]) for o in outs]
        j = np.array([p is not None and p >= 0.5 for p in probs], dtype=float)
        h = np.array([x[2] for x in facts], dtype=float)
        k = kappa(h, j)
        rng = np.random.default_rng(0)
        boot = [kappa(h[ix], j[ix]) for ix in rng.integers(0, len(h), size=(2000, len(h)))]
        # FActScore of a system = mean over responses (with >=1 fact) of the fraction of supported facts
        resp = sorted({x[3] for x in facts})
        by = {ri: [] for ri in resp}
        for (_, _, hs, ri), js in zip(facts, j):
            by[ri].append((hs, js))
        fs_h = float(np.mean([np.mean([hs for hs, _ in v]) for v in by.values()]))
        fs_j = float(np.mean([np.mean([js for _, js in v]) for v in by.values()]))
        out["systems"][sysname] = {"n_facts": len(facts), "n_unreadable": int(sum(p is None for p in probs)), "cohen_kappa": k,
                                   "kappa_95ci": [float(x) for x in np.nanquantile(boot, [0.025, 0.975])], "raw_agreement": float((h == j).mean()),
                                   "confusion": {"human_S_judge_S": int(((h == 1) & (j == 1)).sum()), "human_S_judge_NS": int(((h == 1) & (j == 0)).sum()),
                                                 "human_NS_judge_S": int(((h == 0) & (j == 1)).sum()), "human_NS_judge_NS": int(((h == 0) & (j == 0)).sum())},
                                   "factscore_human": fs_h, "factscore_judge": fs_j, "error_rate": abs(fs_j - fs_h)}
        print(sysname, json.dumps(out["systems"][sysname]), flush=True)
        pooled_h.append(h)
        pooled_j.append(j)
    H, J = np.concatenate(pooled_h), np.concatenate(pooled_j)
    rng = np.random.default_rng(0)
    boot = [kappa(H[ix], J[ix]) for ix in rng.integers(0, len(H), size=(2000, len(H)))]
    out["pooled"] = {"n_facts": int(len(H)), "cohen_kappa": kappa(H, J), "kappa_95ci": [float(x) for x in np.nanquantile(boot, [0.025, 0.975])],
                     "raw_agreement": float((H == J).mean())}
    print("pooled", json.dumps(out["pooled"]))
    out["seconds"] = round(time.perf_counter() - t0, 1)
    out["cost"] = llm.usage_summary()
    dst = Path("results/phase4/judge_validation_factscore.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print("wrote", dst)


if __name__ == "__main__":
    main()
