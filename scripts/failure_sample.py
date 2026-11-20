"""Failure-analysis sample (EXPERIMENT_PLAN.md §8): 100 seeded errors of A on TEST, G1, target 0.90, partition 0 —
50 wrong-but-kept (kept by the filter, judge says unsupported) and 50 correct-but-dropped (dropped, judge says
supported), 25 + 25 per long-form dataset. Writes labelling/failure_sample.csv with the evidence an annotator needs
and empty `category` / `notes` columns. Categories (plan §8): retrieval miss / decomposition error / verifier false
positive / verifier false negative / revision error / judge error.
Usage: python scripts/failure_sample.py [--seed 0]"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

from hallu import analysis as A
from hallu.pipeline import final_claims
from hallu.runutil import CachedBM25
from hallu.store import read_jsonl, text_key

JT = "phi-4-FP8-dynamic"
RUNS = {"bio": "results/test/bio/longform__Qwen2.5-7B-Instruct__5aa2ad1482e5",
        "asqa": "results/test/asqa/longform__Qwen2.5-7B-Instruct__ceeba04af770"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-cell", type=int, default=25)
    a = ap.parse_args()
    bm25 = CachedBM25()
    out_rows = []
    for ds, run in RUNS.items():
        run = Path(run)
        rows = read_jsonl(run / "items.jsonl")
        items = {r["id"]: r for r in read_jsonl(f"data/splits/{ds}_test.jsonl")}
        judge = {r["id"]: r for r in read_jsonl(run / f"judge__{JT}.jsonl")}
        finals = [final_claims(r["claims"]) for r in rows]
        resp = [[(f["score"], judge[text_key(r["id"], f["text"])]["label"]) for f in fs] for r, fs in zip(rows, finals)]
        ev = A.evaluate_once(resp, 0.10, seed=0)
        q = ev["q_hat"]
        cands = {"wrong_but_kept": [], "correct_but_dropped": []}
        for i in ev["eval_index"]:
            for f in finals[i]:
                lab = judge[text_key(rows[i]["id"], f["text"])]["label"]
                if f["score"] > q and not lab:
                    cands["wrong_but_kept"].append((i, f))
                elif f["score"] <= q and lab:
                    cands["correct_but_dropped"].append((i, f))
        rng = np.random.default_rng(a.seed)
        for kind, lst in cands.items():
            pick = sorted(rng.choice(len(lst), size=min(a.per_cell, len(lst)), replace=False).tolist())
            for j in pick:
                i, f = lst[j]
                r = rows[i]
                src = r["claims"][f["source_index"]]
                it = items[r["id"]]
                query = it["meta"]["topic"] if ds == "bio" else it["question"]
                hits = bm25.search([query], 5)[0]
                bp = src.get("best_passage")
                best = hits[bp] if bp is not None and bp < len(hits) else None
                out_rows.append({
                    "sample_id": f"{ds}-{kind}-{len(out_rows):03d}", "dataset": ds, "error_type": kind, "item_id": r["id"],
                    "question": r["question"], "claim_final": f["text"], "claim_original": src["text"], "origin": f["origin"],
                    "action": src["action"], "s_original": round(src["s"], 4), "filter_score": round(f["score"], 4),
                    "threshold_q_hat": round(q, 4), "c_closed_book": round(src["c"], 4), "con": round(src["con"], 4),
                    "judge_p_true": round(judge[text_key(r["id"], f["text"])]["p_true"], 4),
                    "best_passage": (f"{best['title']}: {best['text'][:700]}" if best else ""),
                    "retrieved_titles": " | ".join(h["title"] for h in hits),
                    "answer_excerpt": r["answer"][:600], "category": "", "notes": ""})
        print(ds, {k: len(v) for k, v in cands.items()}, "q_hat", round(q, 4))
    p = Path("labelling/failure_sample.csv")
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print("wrote", p, len(out_rows), "rows")


if __name__ == "__main__":
    main()
