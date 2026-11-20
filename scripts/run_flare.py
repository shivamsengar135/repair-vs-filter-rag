"""FLARE-direct baseline for the long-form datasets — our implementation for chat models (EXPERIMENT_PLAN.md §5.4).

Algorithm (Jiang et al., arXiv:2305.06983, §3 and Appendix A): retrieve with the user input and generate the first
sentence with those documents; then repeatedly (a) generate a tentative next sentence WITHOUT documents (64-token
look-ahead, first Punkt sentence), (b) if any token probability < theta, retrieve with the sentence's
high-confidence tokens (prob >= beta) and regenerate the sentence conditioned on only the newly retrieved
documents, else accept it. theta = 0.8, beta = 0.4 (paper's ASQA / 2Wiki setting).
Chat adaptation (stated): the partial answer is the final assistant message and is continued via vLLM's
continue_final_message; documents are placed in the user turn with the same `long_rag` prompt as standard RAG.
Stops at end of generation, at max_tokens of accepted text, or after MAX_STEPS sentences.

Usage: python scripts/run_flare.py --dataset bio --split dev --model Qwen/Qwen2.5-7B-Instruct [--limit N]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hallu.flare import first_sentence, masked_query, needs_retrieval, sentence_tokens
from hallu.llm import LLM, Request, user
from hallu.parsing import parse_claims
from hallu.prompts import Prompts
from hallu.retrieval import format_passages
from hallu.runutil import CACHE_DIR, DATA_ROOT, CachedBM25, git_commit
from hallu.store import JsonlStore, config_hash, read_jsonl

THETA, BETA, LOOKAHEAD, MAX_STEPS = 0.8, 0.4, 64, 20


def main() -> None:
    import nltk
    nltk.data.path.insert(0, str(DATA_ROOT / "nltk_data"))
    from nltk.tokenize import sent_tokenize

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["bio", "asqa"])
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--prompts", default="configs/prompts/v2.yaml")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    P = Prompts(a.prompts)
    cfg = {"pipeline": "flare-direct-v1", "dataset": a.dataset, "split": a.split, "model": a.model, "prompts": P.version, "k": a.k,
           "retriever": "bm25", "max_tokens": a.max_tokens, "theta": THETA, "beta": BETA, "lookahead": LOOKAHEAD, "max_steps": MAX_STEPS,
           "temperature": 0.0, "seed": 0, **({"limit": a.limit} if a.limit else {})}
    tag = a.model.split("/")[-1]
    out_dir = Path("results") / a.split / a.dataset / f"flare__{tag}__{config_hash(cfg)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    items = read_jsonl(f"data/splits/{a.dataset}_{a.split}.jsonl")
    if a.limit:
        items = items[:a.limit]
    llm = LLM(model=a.model, cache_path=CACHE_DIR / f"llm__{tag}.jsonl", concurrency=a.concurrency)
    bm25 = CachedBM25()
    t0 = time.perf_counter()
    q0 = [it["meta"]["topic"] if a.dataset == "bio" else it["question"] for it in items]

    def rag_msgs(it: dict, hits: list[dict], prefix: str) -> list[dict]:
        m = user(P.fill("long_rag", question=it["question"], passages=format_passages(hits)))
        return m + [{"role": "assistant", "content": prefix}] if prefix else m

    # step 1: first sentence with documents retrieved for the input
    hits0 = bm25.search(q0, a.k)
    first = llm.chat([Request(rag_msgs(it, h, ""), max_tokens=LOOKAHEAD, tag="1_first_sentence") for it, h in zip(items, hits0)])
    answer = [first_sentence(o["text"], sent_tokenize) for o in first]
    finished = [o["finish_reason"] == "stop" and first_sentence(o["text"], sent_tokenize).strip() == o["text"].strip() for o in first]
    n_steps, n_retr = [1] * len(items), [1] * len(items)
    gen_tokens = [0] * len(items)

    for step in range(1, MAX_STEPS):
        active = [i for i in range(len(items)) if not finished[i] and answer[i] and gen_tokens[i] < a.max_tokens]
        if not active:
            break
        look = llm.chat([Request(user(P.fill("long_closed_book", question=items[i]["question"])) + [{"role": "assistant", "content": answer[i]}],
                                 max_tokens=LOOKAHEAD, token_logprobs=True, continue_final=True, tag="2_lookahead") for i in active])
        retr_idx, retr_q = [], []
        for i, o in zip(active, look):
            s = first_sentence(o["text"], sent_tokenize)
            if not s:
                finished[i] = True
                continue
            toks = sentence_tokens(o.get("tokens", []), s)
            if needs_retrieval(toks, THETA):
                retr_idx.append(i)
                retr_q.append(masked_query(toks, BETA) or s)
            else:
                answer[i] = (answer[i] + " " + s).strip()
                gen_tokens[i] += len(toks)
                n_steps[i] += 1
                if o["finish_reason"] == "stop" and s.strip() == o["text"].strip():
                    finished[i] = True
        if retr_idx:
            hits = bm25.search(retr_q, a.k)
            regen = llm.chat([Request(rag_msgs(items[i], h, answer[i]), max_tokens=LOOKAHEAD, continue_final=True, tag="3_regenerate")
                              for i, h in zip(retr_idx, hits)])
            for i, o in zip(retr_idx, regen):
                s = first_sentence(o["text"], sent_tokenize)
                n_retr[i] += 1
                n_steps[i] += 1
                if not s:
                    finished[i] = True
                    continue
                answer[i] = (answer[i] + " " + s).strip()
                gen_tokens[i] += o["completion_tokens"]
                if o["finish_reason"] == "stop" and s.strip() == o["text"].strip():
                    finished[i] = True

    decs = llm.chat([Request(user(P.fill("decompose", question=it["question"], text=t)), max_tokens=1024, tag="4_decompose_final")
                     for it, t in zip(items, answer)])
    claims = [parse_claims(d["text"]) if t else [] for d, t in zip(decs, answer)]
    store = JsonlStore(out_dir / "items.jsonl")
    done = store.ids()
    for it, t, ns, nr, cl in zip(items, answer, n_steps, n_retr, claims):
        if it["id"] not in done:
            store.append({"id": it["id"], "question": it["question"], "answer": t, "n_sentences": ns, "n_retrievals": nr,
                          "claims": [{"text": c, "action": "keep"} for c in cl]})
    summary = {"config": cfg, "config_hash": config_hash(cfg), "git_commit": git_commit(), "n": len(items),
               "sentences_per_response": float(np.mean(n_steps)), "retrievals_per_response": float(np.mean(n_retr)),
               "retrieval_rate_after_first": float((sum(n_retr) - len(items)) / max(1, sum(n_steps) - len(items))),
               "hit_max_steps": int(sum(n >= MAX_STEPS for n in n_steps)), "final_claims_per_response": float(np.mean([len(c) for c in claims])),
               "cost": {"llm_by_stage": llm.usage_summary(), "retrieval_queries": bm25.calls, "seconds_total": round(time.perf_counter() - t0, 2),
                        "note": "seconds include cache hits"},
               "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("n", "sentences_per_response", "retrievals_per_response", "retrieval_rate_after_first",
                                              "hit_max_steps", "final_claims_per_response")}))
    print("llm:", json.dumps(summary["cost"]["llm_by_stage"]))
    print("wrote", out_dir / "summary.json")


if __name__ == "__main__":
    main()
