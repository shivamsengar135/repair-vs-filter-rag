"""Short-form QA runs for the two simplest methods: closed-book (no retrieval) and standard RAG.

Resumable: retrievals and generations are cached; the per-item output file is append-only.
Outputs under results/<split>/<dataset>/<method>__<model-tag>__<cfg-hash>/ : items.jsonl, summary.json

Usage: python scripts/run_qa.py --dataset popqa --split dev --method rag --model Qwen/Qwen2.5-7B-Instruct [--k 5] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from hallu import metrics as M
from hallu.llm import LLM, Request, user
from hallu.parsing import first_line, is_abstention
from hallu.prompts import Prompts
from hallu.retrieval import BM25Retriever, format_passages
from hallu.store import JsonlStore, config_hash, read_jsonl, text_key

DATA_ROOT = Path.home() / "halluc_data"


def git_commit() -> str:
    try:
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def retrieve_cached(queries: list[str], k: int, index_dir: Path, cache: JsonlStore) -> list[list[dict]]:
    have = cache.load()
    keys = [text_key("bm25", index_dir.name, str(k), q) for q in queries]
    missing = [(key, q) for key, q in dict(zip(keys, queries)).items() if key not in have]
    if missing:
        r = BM25Retriever(index_dir)
        for i in range(0, len(missing), 256):
            chunk = missing[i:i + 256]
            for (key, q), hits in zip(chunk, r.search([q for _, q in chunk], k=k)):
                rec = {"id": key, "query": q, "k": k, "hits": [{kk: h[kk] for kk in ("rank", "score", "pid", "title", "text")} for h in hits]}
                cache.append(rec)
                have[key] = rec
    return [have[key]["hits"] for key in keys]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--method", required=True, choices=["closed_book", "rag"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--prompts", default="configs/prompts/v1.yaml")
    ap.add_argument("--index", default=str(DATA_ROOT / "index" / "bm25_psgs_w100"))
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="debug only; limited runs are never reported")
    ap.add_argument("--cache-tag", default=None, help="repeat runs: separate LLM cache + output folder suffix (not part of "
                    "the config) to measure residual serving non-determinism (EXPERIMENT_PLAN.md §10)")
    a = ap.parse_args()

    prompts = Prompts(a.prompts)
    cfg = {"dataset": a.dataset, "split": a.split, "method": a.method, "model": a.model, "prompts": prompts.version,
           "max_tokens": a.max_tokens, "temperature": 0.0, "seed": 0,
           **({"k": a.k, "retriever": "bm25", "index": Path(a.index).name} if a.method == "rag" else {}),
           **({"limit": a.limit} if a.limit else {})}
    tag = a.model.split("/")[-1]
    out_dir = Path("results") / a.split / a.dataset / (f"{a.method}__{tag}__{config_hash(cfg)}" + (f"__{a.cache_tag}" if a.cache_tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    items = read_jsonl(f"data/splits/{a.dataset}_{a.split}.jsonl")
    if a.limit:
        items = items[:a.limit]
    cache_dir = DATA_ROOT / "cache"
    llm = LLM(model=a.model, cache_path=cache_dir / (f"llm__{tag}.jsonl" if not a.cache_tag else f"llm__{tag}__{a.cache_tag}.jsonl"),
              concurrency=a.concurrency)

    t0 = time.perf_counter()
    hits_all: list[list[dict]] = [[] for _ in items]
    t_retr = 0.0
    if a.method == "rag":
        tr = time.perf_counter()
        hits_all = retrieve_cached([it["question"] for it in items], a.k, Path(a.index), JsonlStore(cache_dir / f"retrieval__{Path(a.index).name}.jsonl"))
        t_retr = time.perf_counter() - tr
    reqs = []
    for it, hits in zip(items, hits_all):
        if a.method == "rag":
            p = prompts.fill("short_rag", question=it["question"], passages=format_passages(hits))
        else:
            p = prompts.fill("short_closed_book", question=it["question"])
        reqs.append(Request(user(p), max_tokens=a.max_tokens, tag="answer"))
    tg = time.perf_counter()
    outs = llm.chat(reqs)
    t_gen = time.perf_counter() - tg

    store = JsonlStore(out_dir / "items.jsonl")
    done = store.load()
    rows = []
    for it, hits, o in zip(items, hits_all, outs):
        pred = first_line(o["text"])
        row = {"id": it["id"], "question": it["question"], "answers": it["answers"], "prediction": pred, "raw": o["text"],
               "finish_reason": o["finish_reason"], "abstained": is_abstention(pred),
               "match": M.match_accuracy(pred, it["answers"]), "em": M.exact_match(pred, it["answers"]), "f1": M.token_f1(pred, it["answers"]),
               "prompt_tokens": o["prompt_tokens"], "completion_tokens": o["completion_tokens"],
               "retrieved_pids": [h["pid"] for h in hits],
               "gold_in_passages": float(any(M.match_accuracy(h["text"] + " " + h["title"], it["answers"]) for h in hits)) if hits else None}
        if it["id"] in done:
            assert done[it["id"]]["prediction"] == pred, f"non-deterministic output for {it['id']}"
        else:
            store.append(row)
        rows.append(row)

    n = len(rows)
    summary = {"config": cfg, "config_hash": config_hash(cfg), "git_commit": git_commit(), "n": n,
               "match_accuracy": float(np.mean([r["match"] for r in rows])), "em": float(np.mean([r["em"] for r in rows])),
               "f1": float(np.mean([r["f1"] for r in rows])), "abstention_rate": float(np.mean([r["abstained"] for r in rows])),
               "truncated_rate": float(np.mean([r["finish_reason"] == "length" for r in rows])),
               "retrieval_recall_at_k": (float(np.mean([r["gold_in_passages"] for r in rows])) if a.method == "rag" else None),
               "cost": {"llm": llm.usage_summary(), "prompt_tokens_per_q": float(np.mean([r["prompt_tokens"] for r in rows])),
                        "completion_tokens_per_q": float(np.mean([r["completion_tokens"] for r in rows])),
                        "retrieval_calls_per_q": 1.0 if a.method == "rag" else 0.0,
                        "wall_seconds": {"retrieval": round(t_retr, 2), "generation": round(t_gen, 2), "total": round(time.perf_counter() - t0, 2)},
                        "note": "wall-clock includes cache hits; use a cold cache for timing claims"},
               "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("n", "match_accuracy", "em", "f1", "abstention_rate", "truncated_rate", "retrieval_recall_at_k")}))
    print("cost:", json.dumps(summary["cost"]["llm"]), "wall:", summary["cost"]["wall_seconds"])
    print("wrote", out_dir / "summary.json")


if __name__ == "__main__":
    main()
