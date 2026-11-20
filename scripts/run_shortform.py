"""Short-form version of the triage pipeline (EXPERIMENT_PLAN.md §5, Tier 1): standard RAG answer -> one claim ->
scores (s, con, c) -> triage -> repair by RE-ANSWERING -> final answer + score for selective answering.

A short answer is a single claim, so (stated in §14 C7):
* the claim is the question+answer rewritten as one sentence (prompt `qa_to_claim`);
* "revise" = re-answer the question from the passage that contradicts the claim;
  "re-retrieve" = retrieve top-k for (question + claim) and re-answer from those passages;
  the repaired answer is re-scored (support of its own claim against the passages it was answered from);
* CF / C / A abstain when the answer's score is below the calibrated threshold (single-claim conformal filter =
  selective answering). Standard RAG never abstains.
Outputs per item: rag answer + score s + score_c, A's answer + score, correctness of each (match / EM / F1).

Usage: python scripts/run_shortform.py --dataset popqa --split dev --model Qwen/Qwen2.5-7B-Instruct [--verifier-device cpu]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hallu import metrics as M
from hallu.llm import LLM, Request, binary_probability, user
from hallu.parsing import first_line, is_abstention
from hallu.pipeline import KEEP, LOW, RERETRIEVE, REVISE, Thresholds, score_two_signal, triage
from hallu.prompts import Prompts
from hallu.retrieval import format_passages
from hallu.runutil import CACHE_DIR, DEFAULT_INDEX, CachedBM25, git_commit
from hallu.store import JsonlStore, config_hash, read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["popqa", "nq", "triviaqa", "hotpotqa", "2wiki"])
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tau-s", type=float, default=0.5)
    ap.add_argument("--tau-c", type=float, default=0.5)
    ap.add_argument("--tau-con", type=float, default=0.5)
    ap.add_argument("--verifier-device", default=None)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    P, S = Prompts("configs/prompts/v1.yaml"), Prompts("configs/prompts/shortform_v1.yaml")
    th = Thresholds(a.tau_s, a.tau_c, a.tau_con)
    cfg = {"pipeline": "shortform-v1", "dataset": a.dataset, "split": a.split, "model": a.model, "prompts": [P.version, S.version],
           "k": a.k, "retriever": "bm25", "index": DEFAULT_INDEX.name, "max_tokens": 64, "temperature": 0.0, "seed": 0,
           "tau_s": a.tau_s, "tau_c": a.tau_c, "tau_con": a.tau_con, "verifier": "MiniCheck-Flan-T5-Large",
           "nli": "DeBERTa-v3-large-mnli-fever-anli-ling-wanli", **({"limit": a.limit} if a.limit else {})}
    tag = a.model.split("/")[-1]
    out_dir = Path("results") / a.split / a.dataset / f"shortform__{tag}__{config_hash(cfg)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    items = read_jsonl(f"data/splits/{a.dataset}_{a.split}.jsonl")
    if a.limit:
        items = items[:a.limit]
    llm = LLM(model=a.model, cache_path=CACHE_DIR / f"llm__{tag}.jsonl", concurrency=a.concurrency)
    bm25 = CachedBM25()
    t0 = time.perf_counter()

    def doc(h: dict) -> str:
        return f"{h['title']}. {h['text']}"

    def answer(prompt_name: str, it: dict, hits: list[dict], tagname: str) -> list:
        return [first_line(o["text"]) for o in llm.chat([Request(user(P.fill("short_rag", question=q["question"], passages=format_passages(h))
                                                                      if prompt_name == "short_rag" else
                                                                      S.fill("short_rag_with_passage", question=q["question"], passages=format_passages(h))),
                                                                 max_tokens=64, tag=tagname) for q, h in zip(it, hits)])]

    def to_claims(qs: list[dict], ans: list[str], tagname: str) -> list[str]:
        outs = llm.chat([Request(user(S.fill("qa_to_claim", question=q["question"], answer=x)), max_tokens=96, tag=tagname) for q, x in zip(qs, ans)])
        return [first_line(o["text"]) or f"{q['question']} {x}" for q, x, o in zip(qs, ans, outs)]

    # 1 retrieve + answer (identical prompt and retrieval to the standard-RAG baseline -> cache hits)
    hits = bm25.search([it["question"] for it in items], a.k)
    ans0 = answer("short_rag", items, hits, "1_answer")
    claims0 = to_claims(items, ans0, "2_qa_to_claim")
    # 2 score
    from hallu.verifier import NLI, MiniCheck
    mc, nli = MiniCheck(device=a.verifier_device), NLI(device=a.verifier_device)
    pairs = [(i, j) for i in range(len(items)) for j in range(len(hits[i]))]
    sup = mc.support_prob([doc(hits[i][j]) for i, j in pairs], [claims0[i] for i, _ in pairs])
    con = [x["contradiction"] for x in nli.probs([doc(hits[i][j]) for i, j in pairs], [claims0[i] for i, _ in pairs])]
    cb = llm.chat([Request(user(P.fill("closed_book_truth", claim=c)), max_tokens=1, top_logprobs=20, tag="3_closed_book_truth") for c in claims0])
    s0, con0, conj = [0.0] * len(items), [0.0] * len(items), [None] * len(items)
    for (i, j), sp, cn in zip(pairs, sup, con):
        s0[i] = max(s0[i], sp)
        if cn > con0[i]:
            con0[i], conj[i] = cn, j
    c0, n_unread = [], 0
    for o in cb:
        p = binary_probability(o.get("first_token_top", []), ["true", "yes"], ["false", "no"])
        n_unread += p is None
        c0.append(0.5 if p is None else p)
    acts = [triage(s, cn, c, th) if not is_abstention(x) else LOW for s, cn, c, x in zip(s0, con0, c0, ans0)]
    # 3 repair by re-answering
    rev = [i for i, x in enumerate(acts) if x == REVISE]
    rer = [i for i, x in enumerate(acts) if x == RERETRIEVE]
    rev_hits = [[hits[i][conj[i]]] for i in rev]
    rer_hits = bm25.search([f"{items[i]['question']} {claims0[i]}" for i in rer], a.k)
    rep_idx, rep_hits = rev + rer, rev_hits + rer_hits
    rep_ans = answer("short_rag_with_passage", [items[i] for i in rev], rev_hits, "4_reanswer_revise") + \
        answer("short_rag", [items[i] for i in rer], rer_hits, "4_reanswer_reretrieve")
    rep_claims = to_claims([items[i] for i in rep_idx], rep_ans, "5_qa_to_claim_repaired")
    rpairs = [(n, j) for n in range(len(rep_idx)) for j in range(len(rep_hits[n]))]
    rsup = mc.support_prob([doc(rep_hits[n][j]) for n, j in rpairs], [rep_claims[n] for n, _ in rpairs])
    rs = [0.0] * len(rep_idx)
    for (n, _), sp in zip(rpairs, rsup):
        rs[n] = max(rs[n], sp)

    store = JsonlStore(out_dir / "items.jsonl")
    done = store.ids()
    rows = []
    rep_map = {i: n for n, i in enumerate(rep_idx)}
    for i, it in enumerate(items):
        golds = it["answers"]
        a_ans, a_score, origin = ans0[i], s0[i], "original"
        if i in rep_map:
            n = rep_map[i]
            if acts[i] == REVISE or rs[n] > s0[i]:   # a revision replaces the answer; a re-retrieval only if it scores higher
                a_ans, a_score, origin = rep_ans[n], (rs[n] if acts[i] == REVISE else rs[n]), acts[i]
        row = {"id": it["id"], "question": it["question"], "answers": golds, "rag_answer": ans0[i], "claim": claims0[i],
               "s": s0[i], "con": con0[i], "c": c0[i], "score_c": score_two_signal(s0[i], c0[i]), "action": acts[i],
               "rag_abstained": is_abstention(ans0[i]),
               "rag_match": M.match_accuracy(ans0[i], golds), "rag_em": M.exact_match(ans0[i], golds), "rag_f1": M.token_f1(ans0[i], golds),
               "A_answer": a_ans, "A_score": a_score, "A_origin": origin,
               "A_match": M.match_accuracy(a_ans, golds), "A_em": M.exact_match(a_ans, golds), "A_f1": M.token_f1(a_ans, golds),
               **({"repair_answer": rep_ans[rep_map[i]], "repair_claim": rep_claims[rep_map[i]], "repair_s": rs[rep_map[i]]} if i in rep_map else {})}
        if it["id"] not in done:
            store.append(row)
        rows.append(row)
    act_counts = {k: acts.count(k) for k in (KEEP, REVISE, RERETRIEVE, LOW)}
    changed = [r for r in rows if r["A_origin"] != "original"]
    summary = {"config": cfg, "config_hash": config_hash(cfg), "git_commit": git_commit(), "n": len(rows),
               "rag_match": float(np.mean([r["rag_match"] for r in rows])), "A_match_no_abstention": float(np.mean([r["A_match"] for r in rows])),
               "actions": act_counts, "answers_changed_by_repair": len(changed),
               "repair_effect_on_match": {"fixed": sum(1 for r in changed if r["A_match"] > r["rag_match"]),
                                          "broken": sum(1 for r in changed if r["A_match"] < r["rag_match"])},
               "mean_s": float(np.mean(s0)), "closed_book_unreadable": n_unread,
               "auroc_note": "selective metrics computed by scripts/analyze_shortform.py",
               "cost": {"llm_by_stage": llm.usage_summary(), "retrieval_queries": bm25.calls,
                        "verifier_pairs": {"minicheck": len(pairs) + len(rpairs), "nli": len(pairs)},
                        "seconds_total": round(time.perf_counter() - t0, 2), "verifier_device": mc.device},
               "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("n", "rag_match", "A_match_no_abstention", "actions", "answers_changed_by_repair",
                                              "repair_effect_on_match", "mean_s", "closed_book_unreadable")}))
    print("wrote", out_dir / "summary.json")


if __name__ == "__main__":
    main()
