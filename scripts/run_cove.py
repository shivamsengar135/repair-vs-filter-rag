"""CoVe (factor + revise) baseline for the long-form datasets — our implementation from the paper's templates.

Variants:
  closed_book  as published: draft and verification answers come from the model alone
  rag_verify   labelled variant: verification questions are answered with BM25 passages (draft still closed-book)
Steps: draft -> plan (fact ||| question) -> execute each question in its own prompt (the draft is NOT in context)
       -> cross-check each fact -> splice the consistent facts -> decompose the final text into atomic claims
       (the same decomposition prompt as every other method, so claims are comparable).
Output: results/<split>/<dataset>/cove-<variant>__<model>__<hash>/items.jsonl + summary.json (rows carry `claims`
        in the format scripts/run_judge.py expects).

Usage: python scripts/run_cove.py --dataset bio --split dev --variant closed_book --model Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from hallu.llm import LLM, Request, user
from hallu.parsing import parse_claims
from hallu.prompts import Prompts
from hallu.retrieval import format_passages
from hallu.runutil import CACHE_DIR, CachedBM25, git_commit
from hallu.store import JsonlStore, config_hash, read_jsonl

_VERDICT = re.compile(r"^\W*(PARTIALLY CONSISTENT|INCONSISTENT|CONSISTENT)\b[.:\s]*(.*)$", re.I | re.S)


def parse_plan(text: str, max_pairs: int = 40) -> list[tuple[str, str]]:
    """Lines of the form '<fact> ||| <question>'; list markers are stripped; malformed lines are dropped."""
    out = []
    for line in text.splitlines():
        if "|||" not in line:
            continue
        fact, _, q = line.partition("|||")
        fact = re.sub(r"^\s*(?:[-*•]|\(?\d{1,3}[.)])\s+", "", fact).strip()
        q = q.strip()
        if len(fact) >= 3 and len(q) >= 3:
            out.append((fact, q))
    return out[:max_pairs]


def parse_crosscheck(text: str, fact: str) -> tuple[str, str | None]:
    """(verdict, kept text). CONSISTENT keeps the original fact; PARTIALLY keeps the stated part; else nothing.
    Unparseable output counts as INCONSISTENT (the fact is dropped) and is reported in the summary."""
    m = _VERDICT.match(text.strip())
    if not m:
        return "UNPARSED", None
    verdict, rest = m.group(1).upper(), m.group(2).strip().splitlines()[0].strip() if m.group(2).strip() else ""
    if verdict == "CONSISTENT":
        return verdict, fact
    if verdict == "PARTIALLY CONSISTENT":
        return verdict, (rest or None)
    return verdict, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["bio", "asqa"])
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--variant", required=True, choices=["closed_book", "rag_verify"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--prompts", default="configs/prompts/v2.yaml")
    ap.add_argument("--baseline-prompts", default="configs/prompts/baselines_v1.yaml")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    P, B = Prompts(a.prompts), Prompts(a.baseline_prompts)
    cfg = {"pipeline": "cove-factor-revise-v1", "variant": a.variant, "dataset": a.dataset, "split": a.split, "model": a.model,
           "prompts": P.version, "baseline_prompts": B.version, "max_tokens": a.max_tokens, "temperature": 0.0, "seed": 0,
           **({"k": a.k, "retriever": "bm25"} if a.variant == "rag_verify" else {}), **({"limit": a.limit} if a.limit else {})}
    tag = a.model.split("/")[-1]
    out_dir = Path("results") / a.split / a.dataset / f"cove-{a.variant}__{tag}__{config_hash(cfg)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    items = read_jsonl(f"data/splits/{a.dataset}_{a.split}.jsonl")
    if a.limit:
        items = items[:a.limit]
    llm = LLM(model=a.model, cache_path=CACHE_DIR / f"llm__{tag}.jsonl", concurrency=a.concurrency)
    bm25 = CachedBM25()
    t0 = time.perf_counter()

    drafts = llm.chat([Request(user(P.fill("long_closed_book", question=it["question"])), max_tokens=a.max_tokens, tag="1_draft") for it in items])
    plans_raw = llm.chat([Request(user(B.fill("cove_plan", question=it["question"], answer=d["text"])), max_tokens=1024, tag="2_plan")
                          for it, d in zip(items, drafts)])
    plans = [parse_plan(p["text"]) for p in plans_raw]
    flat = [(qi, fi) for qi, pl in enumerate(plans) for fi in range(len(pl))]
    if a.variant == "rag_verify":
        hits = bm25.search([plans[q][f][1] for q, f in flat], a.k)
        ex_reqs = [Request(user(B.fill("cove_execute_rag", vq=plans[q][f][1], passages=format_passages(h))), max_tokens=64, tag="3_execute")
                   for (q, f), h in zip(flat, hits)]
    else:
        ex_reqs = [Request(user(B.fill("cove_execute", vq=plans[q][f][1])), max_tokens=64, tag="3_execute") for q, f in flat]
    answers = llm.chat(ex_reqs)
    checks = llm.chat([Request(user(B.fill("cove_crosscheck", fact=plans[q][f][0], vq=plans[q][f][1], va=o["text"].strip().splitlines()[0] if o["text"].strip() else "")),
                               max_tokens=96, tag="4_crosscheck") for (q, f), o in zip(flat, answers)])
    kept: list[list[str]] = [[] for _ in items]
    verdicts: list[list[str]] = [[] for _ in items]
    for (q, f), c in zip(flat, checks):
        v, text = parse_crosscheck(c["text"], plans[q][f][0])
        verdicts[q].append(v)
        if text:
            kept[q].append(text)
    finals = [" ".join(t if t.endswith((".", "!", "?")) else t + "." for t in k) for k in kept]
    decs = llm.chat([Request(user(P.fill("decompose", question=it["question"], text=f)), max_tokens=1024, tag="5_decompose_final")
                     for it, f in zip(items, finals)])
    claims = [parse_claims(d["text"]) if f else [] for d, f in zip(decs, finals)]

    store = JsonlStore(out_dir / "items.jsonl")
    done = store.ids()
    for it, d, pl, v, f, cl in zip(items, drafts, plans, verdicts, finals, claims):
        if it["id"] not in done:
            store.append({"id": it["id"], "question": it["question"], "draft": d["text"], "n_planned_facts": len(pl), "verdicts": v,
                          "answer": f, "claims": [{"text": t, "action": "keep"} for t in cl]})
    all_v = [x for v in verdicts for x in v]
    summary = {"config": cfg, "config_hash": config_hash(cfg), "git_commit": git_commit(), "n": len(items),
               "planned_facts_per_response": float(np.mean([len(p) for p in plans])),
               "responses_with_no_parsable_plan": int(sum(len(p) == 0 for p in plans)),
               "verdicts": {k: all_v.count(k) for k in ("CONSISTENT", "PARTIALLY CONSISTENT", "INCONSISTENT", "UNPARSED")},
               "final_claims_per_response": float(np.mean([len(c) for c in claims])), "empty_final_answers": int(sum(not f for f in finals)),
               "cost": {"llm_by_stage": llm.usage_summary(), "retrieval_queries": bm25.calls, "seconds_total": round(time.perf_counter() - t0, 2),
                        "note": "seconds include cache hits"},
               "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("n", "planned_facts_per_response", "responses_with_no_parsable_plan", "verdicts",
                                              "final_claims_per_response", "empty_final_answers")}))
    print("llm:", json.dumps(summary["cost"]["llm_by_stage"]))
    print("wrote", out_dir / "summary.json")


if __name__ == "__main__":
    main()
