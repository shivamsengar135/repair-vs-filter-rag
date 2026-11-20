"""Budgeted repair B vs a random subset of the same size (EXPERIMENT_PLAN.md §5.3) — TEST, G1, EXPLORATORY.

For B in {1, 2, 4}: B(risk) applies the B attempted repairs with the lowest original support (hallu.pipeline);
B(random) applies B attempted repairs chosen uniformly at random per response (seeded 0..9, averaged). Both are
evaluated at target 0.90 over 100 partitions (supported claims / answer). Offline: repairs are already stored.
Writes results/<split>/budget_random.json. Usage: python scripts/budget_random.py [--split test]"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

from hallu import analysis as A
from hallu.pipeline import RERETRIEVE, REVISE, final_claims
from hallu.store import read_jsonl, text_key

JT = "phi-4-FP8-dynamic"


def default_run(split: str, ds: str) -> Path:
    for r in glob.glob(f"results/{split}/{ds}/longform__Qwen2.5-7B-Instruct__*/"):
        cfg = json.loads((Path(r) / "summary.json").read_text())["config"]
        if (cfg.get("k") == 5 and cfg.get("tau_s") == 0.5 and cfg.get("retriever") == "bm25" and cfg.get("verifier") == "MiniCheck-Flan-T5-Large"
                and not ({"triage", "decompose", "ablation_prompts", "limit"} & set(cfg))):
            return Path(r)
    raise SystemExit("no default run")


def random_budget(claims: list, b: int, rng) -> list:
    """Like final_claims(claims, b) but the b repairs are drawn uniformly among the attempted ones."""
    att = [i for i, c in enumerate(claims) if c["action"] in (REVISE, RERETRIEVE) and c.get("repair") is not None]
    keep = set(rng.choice(att, size=min(b, len(att)), replace=False).tolist()) if att else set()
    out = []
    for i, c in enumerate(claims):
        if i in keep:
            rep = c["repair"]
            if c["action"] == REVISE:
                if rep["text"] is None:
                    continue
                out.append({"text": rep["text"], "score": rep["s_after"]})
            else:
                out.append({"text": c["text"], "score": max(c["s"], rep["s_after"])})
        else:
            out.append({"text": c["text"], "score": c["s"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    out = {"label": "EXPLORATORY (§5.3)", "results": {}}
    for ds in ("bio", "asqa"):
        run = default_run(a.split, ds)
        rows = read_jsonl(run / "items.jsonl")
        lab = {r["id"]: r["label"] for r in read_jsonl(run / f"judge__{JT}.jsonl")}
        L = lambda iid, t: lab[text_key(iid, t)]  # noqa: E731
        res = {}
        for b in (1, 2, 4):
            risk = [[(f["score"], L(r["id"], f["text"])) for f in final_claims(r["claims"], b)] for r in rows]
            e_risk = A.evaluate(risk, 0.10, 100)["supported_claims_per_response"]
            means = []
            for seed in range(10):
                rng = np.random.default_rng(seed)
                rnd = [[(f["score"], L(r["id"], f["text"])) for f in random_budget(r["claims"], b, rng)] for r in rows]
                means.append(A.evaluate(rnd, 0.10, 100)["supported_claims_per_response"]["mean"])
            res[str(b)] = {"risk_ordered": e_risk, "random_mean_over_10_draws": float(np.mean(means)), "random_std_over_draws": float(np.std(means, ddof=1))}
            print(f"{ds} B={b}: risk-ordered {e_risk['mean']:.3f} ± {e_risk['std']:.3f} | random {np.mean(means):.3f} (sd over draws {np.std(means, ddof=1):.3f})", flush=True)
        out["results"][ds] = {"run": str(run), **res}
    Path(f"results/{a.split}/budget_random.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
