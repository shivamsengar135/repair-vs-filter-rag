"""Judge every candidate claim of a long-form run against reference evidence (EXPERIMENT_PLAN.md §7).

Claims judged: all original claims plus every repaired claim text, de-duplicated per item.
Evidence (never shown to any method):
  asqa: ALCE's oracle-reranked passages + the human-written reference long answers
  bio : passages of the entity's page in FActScore's Wikipedia knowledge base (requires enwiki-20230401.db)
Label = P("True") >= 0.5 from the judge's first-token probabilities; P is stored so thresholds can be audited.

Usage: python scripts/run_judge.py --run results/dev/asqa/longform__... --judge-model <served model name>
Output: <run>/judge__<judge-tag>.jsonl  and  <run>/judge__<judge-tag>__summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hallu.llm import LLM, Request, binary_probability, user
from hallu.prompts import Prompts
from hallu.runutil import CACHE_DIR, git_commit
from hallu.store import JsonlStore, read_jsonl, text_key


def asqa_evidence(item: dict) -> str:
    m = item["meta"]
    parts = [f"[{i}] ({d['title']}) {d['text']}" for i, d in enumerate(m["oracle_docs"], 1)]
    parts += [f"[reference answer {i}] {a}" for i, a in enumerate(m["gold_long_answers"], 1)]
    return "\n".join(parts)


def candidate_claims(row: dict) -> list[str]:
    seen, out = set(), []
    for c in row["claims"]:
        texts = [c["text"]]
        rep = c.get("repair")
        if rep and rep.get("text"):
            texts.append(rep["text"])
        for t in texts:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--prompts", default="configs/prompts/judge_v1.yaml")
    ap.add_argument("--concurrency", type=int, default=32)
    a = ap.parse_args()
    run = Path(a.run)
    cfg = json.loads((run / "summary.json").read_text())["config"]
    dataset, split = cfg["dataset"], cfg["split"]
    prompts = Prompts(a.prompts)
    items = {it["id"]: it for it in read_jsonl(f"data/splits/{dataset}_{split}.jsonl")}
    rows = read_jsonl(run / "items.jsonl")
    jtag = a.judge_model.split("/")[-1]
    llm = LLM(model=a.judge_model, cache_path=CACHE_DIR / f"llm__{jtag}.jsonl", concurrency=a.concurrency)

    todo = [(r["id"], t) for r in rows for t in candidate_claims(r)]
    n_no_kb = 0
    if dataset == "asqa":
        reqs = [Request(user(prompts.fill("judge_claim", evidence=asqa_evidence(items[i]), claim=t)), max_tokens=1, top_logprobs=20, tag="judge")
                for i, t in todo]
    else:  # bio: FActScore's knowledge source, evidence selection and prompt layout (EXPERIMENT_PLAN.md §14 C6)
        from hallu.factscore_kb import FactScoreKB, factscore_prompt
        kb = FactScoreKB()
        reqs = []
        for i, t in todo:
            topic = items[i]["meta"]["topic"]
            psgs = kb.top_k(topic, t, k=5)
            n_no_kb += not psgs
            reqs.append(Request(user(factscore_prompt(topic, t, psgs)), max_tokens=1, top_logprobs=20, tag="judge"))
    outs = llm.chat(reqs)
    store = JsonlStore(run / f"judge__{jtag}.jsonl")
    have = store.load()
    n_unreadable = n_true = 0
    for (i, t), o in zip(todo, outs):
        p = binary_probability(o.get("first_token_top", []), ["true", "yes"], ["false", "no"])
        if p is None:
            n_unreadable += 1
        label = bool(p is not None and p >= 0.5)   # unreadable verdicts count as NOT supported (conservative) and are reported
        n_true += label
        key = text_key(i, t)
        if key not in have:
            store.append({"id": key, "item_id": i, "claim": t, "p_true": p, "label": label})
    summary = {"run": str(run), "judge_model": a.judge_model,
               "judge_prompts": prompts.version if dataset == "asqa" else "factscore-layout (hallu.factscore_kb)",
               "claims_without_kb_page": n_no_kb, "n_claims": len(todo),
               "n_supported": n_true, "supported_rate": n_true / max(1, len(todo)), "n_unreadable": n_unreadable,
               "cost": llm.usage_summary(), "git_commit": git_commit()}
    (run / f"judge__{jtag}__summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
