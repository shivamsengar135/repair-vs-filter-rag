"""RARR-style "research and revise" baseline for the long-form datasets — our implementation (EXPERIMENT_PLAN.md §5.4).

Follows the structure of RARR (Gao et al., arXiv:2210.08726): generate verification questions for a finished text,
retrieve evidence for each question, check agreement, and edit the text where the evidence disagrees, one question
after another. Stated differences: our BM25 corpus replaces web search; one passage (the top hit) per question;
zero-shot prompts for an instruction-tuned model replace the paper's few-shot prompts; at most MAX_Q questions.
The text being revised is the model's closed-book draft (RARR post-edits the output of a generation model).

Output: results/<split>/<dataset>/rarr__<model>__<hash>/items.jsonl + summary.json (rows carry `claims`).
Usage: python scripts/run_rarr.py --dataset bio --split dev --model Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hallu.llm import LLM, Request, user
from hallu.parsing import parse_claims, parse_yes_no
from hallu.prompts import Prompts
from hallu.runutil import CACHE_DIR, CachedBM25, git_commit
from hallu.store import JsonlStore, config_hash, read_jsonl

MAX_Q = 8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["bio", "asqa"])
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--prompts", default="configs/prompts/v2.yaml")
    ap.add_argument("--baseline-prompts", default="configs/prompts/baselines_v1.yaml")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    P, B = Prompts(a.prompts), Prompts(a.baseline_prompts)
    cfg = {"pipeline": "rarr-style-v1", "dataset": a.dataset, "split": a.split, "model": a.model, "prompts": P.version,
           "baseline_prompts": B.version, "max_tokens": a.max_tokens, "temperature": 0.0, "seed": 0, "max_questions": MAX_Q,
           "retriever": "bm25", "passages_per_question": 1, **({"limit": a.limit} if a.limit else {})}
    tag = a.model.split("/")[-1]
    out_dir = Path("results") / a.split / a.dataset / f"rarr__{tag}__{config_hash(cfg)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    items = read_jsonl(f"data/splits/{a.dataset}_{a.split}.jsonl")
    if a.limit:
        items = items[:a.limit]
    llm = LLM(model=a.model, cache_path=CACHE_DIR / f"llm__{tag}.jsonl", concurrency=a.concurrency)
    bm25 = CachedBM25()
    t0 = time.perf_counter()

    drafts = llm.chat([Request(user(P.fill("long_closed_book", question=it["question"])), max_tokens=a.max_tokens, tag="1_draft") for it in items])
    qraw = llm.chat([Request(user(B.fill("rarr_questions", question=it["question"], text=d["text"])), max_tokens=512, tag="2_questions")
                     for it, d in zip(items, drafts)])
    questions = [parse_claims(q["text"], max_claims=MAX_Q) for q in qraw]   # same "- " list format as claims
    flat = [(i, j) for i, qs in enumerate(questions) for j in range(len(qs))]
    hits = bm25.search([questions[i][j] for i, j in flat], 1)
    evidence = {(i, j): (f"{h[0]['title']}. {h[0]['text']}" if h else None) for (i, j), h in zip(flat, hits)}

    texts = [d["text"] for d in drafts]
    n_edits = [0] * len(items)
    n_disagree = n_unparsed = 0
    for j in range(MAX_Q):   # sequential over questions (each edit sees earlier edits), parallel over responses
        active = [i for i in range(len(items)) if j < len(questions[i]) and evidence[(i, j)]]
        if not active:
            continue
        agree = llm.chat([Request(user(B.fill("rarr_agreement", passage=evidence[(i, j)], text=texts[i], vq=questions[i][j])), max_tokens=3, tag="3_agreement")
                          for i in active])
        to_edit = []
        for i, o in zip(active, agree):
            v = parse_yes_no(o["text"])
            n_unparsed += v is None
            if v is True:
                to_edit.append(i)
        n_disagree += len(to_edit)
        edits = llm.chat([Request(user(B.fill("rarr_edit", passage=evidence[(i, j)], text=texts[i], vq=questions[i][j])), max_tokens=a.max_tokens, tag="4_edit")
                          for i in to_edit])
        for i, e in zip(to_edit, edits):
            new = e["text"].strip()
            if new and new != texts[i]:
                texts[i] = new
                n_edits[i] += 1

    decs = llm.chat([Request(user(P.fill("decompose", question=it["question"], text=t)), max_tokens=1024, tag="5_decompose_final") for it, t in zip(items, texts)])
    claims = [parse_claims(d["text"]) for d in decs]
    store = JsonlStore(out_dir / "items.jsonl")
    done = store.ids()
    for it, d, qs, t, ne, cl in zip(items, drafts, questions, texts, n_edits, claims):
        if it["id"] not in done:
            store.append({"id": it["id"], "question": it["question"], "draft": d["text"], "questions": qs, "n_edits": ne, "answer": t,
                          "claims": [{"text": c, "action": "keep"} for c in cl]})
    summary = {"config": cfg, "config_hash": config_hash(cfg), "git_commit": git_commit(), "n": len(items),
               "questions_per_response": float(np.mean([len(q) for q in questions])), "agreement_checks": len(flat),
               "disagreements": n_disagree, "agreement_unparsed": int(n_unparsed), "edits_applied": int(sum(n_edits)),
               "responses_edited": int(sum(n > 0 for n in n_edits)), "final_claims_per_response": float(np.mean([len(c) for c in claims])),
               "cost": {"llm_by_stage": llm.usage_summary(), "retrieval_queries": bm25.calls, "seconds_total": round(time.perf_counter() - t0, 2),
                        "note": "seconds include cache hits"},
               "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("n", "questions_per_response", "agreement_checks", "disagreements", "agreement_unparsed",
                                              "edits_applied", "responses_edited", "final_claims_per_response")}))
    print("llm:", json.dumps(summary["cost"]["llm_by_stage"]))
    print("wrote", out_dir / "summary.json")


if __name__ == "__main__":
    main()
