"""Long-form pipeline: standard RAG answer -> atomic claims -> scores -> triage -> repair (EXPERIMENT_PLAN.md §5).

One run produces everything needed, offline, for four methods on the same generations:
    standard RAG (all claims), CF filter-only (score s), Candidate C (two-signal score), Triage-and-Repair A
    (repaired claims and scores), and the budgeted variant B (subset of A's repairs).
The conformal threshold and the judge labels are applied later by the analysis scripts.

Stages (each cached through the LLM / retrieval caches, so the script is resumable by re-running it):
  1 retrieve top-k for the question          4 score claims: MiniCheck support s, NLI contradiction, closed-book P(True)
  2 generate the answer (RAG)                5 triage -> actions
  3 decompose the answer into atomic claims  6 repair: revise contradicted claims / re-retrieve for confident ones, re-score

Usage: python scripts/run_longform.py --dataset bio --split dev --model Qwen/Qwen2.5-7B-Instruct [--limit N] [--verifier-device cpu]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hallu.llm import LLM, Request, binary_probability, user
from hallu.parsing import parse_claims
from hallu.pipeline import KEEP, LOW, RERETRIEVE, REVISE, Thresholds, final_claims, score_two_signal, triage
from hallu.prompts import Prompts
from hallu.retrieval import format_passages
from hallu.runutil import CACHE_DIR, DATA_ROOT, DEFAULT_INDEX, DENSE_INDEX, git_commit, make_retriever
from hallu.store import JsonlStore, config_hash, read_jsonl


def retrieval_query(item: dict) -> str:
    """Biographies: the entity name. Other datasets: the question text."""
    return item["meta"]["topic"] if item["dataset"] == "bio" else item["question"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["bio", "asqa"])
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--tau-s", type=float, default=0.5)
    ap.add_argument("--tau-c", type=float, default=0.5)
    ap.add_argument("--tau-con", type=float, default=0.5)
    ap.add_argument("--prompts", default="configs/prompts/v2.yaml")
    ap.add_argument("--verifier-device", default=None, help="cpu while vLLM holds the GPU; default = cuda if free")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="debug only; limited runs are never reported")
    ap.add_argument("--stop-after", choices=["decompose", "score"], default=None,
                    help="scheduling only (not part of the config): 'decompose' needs the generator server; 'score' fills the "
                         "verifier score cache (run it with the GPU free and --verifier-device cuda); a later full run reuses both caches")
    # Phase 6 ablations / sensitivity (EXPERIMENT_PLAN.md §8, D6). Each enters the config only when not the default,
    # so the hashes (and output directories) of all earlier runs are unchanged.
    ap.add_argument("--retriever", choices=["bm25", "dense"], default="bm25")
    ap.add_argument("--verifier", choices=["minicheck", "alignscore", "nli", "generator"], default="minicheck",
                    help="support score s: MiniCheck (default) | AlignScore-large | DeBERTa NLI P(entailment) | the generator (P(True))")
    ap.add_argument("--triage", choices=["two_axis", "one_axis"], default="two_axis")
    ap.add_argument("--decompose", choices=["llm", "sentences"], default="llm")
    a = ap.parse_args()

    prompts = Prompts(a.prompts)
    th = Thresholds(a.tau_s, a.tau_c, a.tau_con)
    verifier_names = {"minicheck": "MiniCheck-Flan-T5-Large", "alignscore": "AlignScore-large (liuyanyi/AlignScore-large-hf)",
                      "nli": "DeBERTa-v3-large-mnli-fever-anli-ling-wanli P(entailment)", "generator": "generator P(True) | passage"}
    abl_prompts = Prompts("configs/prompts/ablation_v1.yaml") if a.verifier == "generator" else None
    index_name = DEFAULT_INDEX.name if a.retriever == "bm25" else DENSE_INDEX.name
    cfg = {"pipeline": "longform-v1", "dataset": a.dataset, "split": a.split, "model": a.model, "prompts": prompts.version,
           "k": a.k, "retriever": a.retriever, "index": index_name, "max_tokens": a.max_tokens, "temperature": 0.0, "seed": 0,
           "tau_s": a.tau_s, "tau_c": a.tau_c, "tau_con": a.tau_con,
           "verifier": verifier_names[a.verifier], "nli": "DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
           **({"ablation_prompts": abl_prompts.version} if abl_prompts else {}),
           **({"triage": a.triage} if a.triage != "two_axis" else {}),
           **({"decompose": a.decompose} if a.decompose != "llm" else {}),
           **({"limit": a.limit} if a.limit else {})}
    tag = a.model.split("/")[-1]
    out_dir = Path("results") / a.split / a.dataset / f"longform__{tag}__{config_hash(cfg)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    items = read_jsonl(f"data/splits/{a.dataset}_{a.split}.jsonl")
    if a.limit:
        items = items[:a.limit]
    llm = LLM(model=a.model, cache_path=CACHE_DIR / f"llm__{tag}.jsonl", concurrency=a.concurrency)
    bm25 = make_retriever(a.retriever)  # name kept for the BM25 default; dense is the §3 sensitivity retriever
    timing: dict[str, float] = {}

    def timed(name: str):
        class _T:
            def __enter__(s):
                s.t = time.perf_counter()
                print(f"[{time.strftime('%H:%M:%S')}] stage {name} ...", flush=True)

            def __exit__(s, *exc):
                timing[name] = timing.get(name, 0.0) + time.perf_counter() - s.t
                print(f"[{time.strftime('%H:%M:%S')}] stage {name} done in {time.perf_counter() - s.t:.0f}s", flush=True)
        return _T()

    # ---- 1 retrieve, 2 generate
    with timed("retrieve"):
        hits_all = bm25.search([retrieval_query(it) for it in items], a.k)
    with timed("generate"):
        gens = llm.chat([Request(user(prompts.fill("long_rag", question=it["question"], passages=format_passages(h))),
                                 max_tokens=a.max_tokens, tag="1_generate") for it, h in zip(items, hits_all)])
    # ---- 3 decompose
    with timed("decompose"):
        if a.decompose == "llm":
            decs = llm.chat([Request(user(prompts.fill("decompose", question=it["question"], text=g["text"])), max_tokens=1024, tag="2_decompose")
                             for it, g in zip(items, gens)])
            claims_all = [parse_claims(d["text"]) for d in decs]
        else:  # ablation §8.7: every sentence of the answer is a "claim" (NLTK Punkt, the splitter FLARE uses here)
            import nltk
            from nltk.tokenize import sent_tokenize
            nltk.data.path.insert(0, str(DATA_ROOT / "nltk_data"))
            decs = [{"text": "", "finish_reason": "n/a"} for _ in gens]
            claims_all = [[s.strip() for s in sent_tokenize(g["text"]) if s.strip()] for g in gens]
    if a.stop_after == "decompose":
        print(f"STOP_AFTER decompose: {len(items)} items, {sum(map(len, claims_all))} claims cached")
        return

    # ---- 4 score
    from hallu.verifier import NLI, MINICHECK_REV, NLI_REV, MiniCheck, ScoreCache, aggregate_claim_scores

    class _Lazy:  # verifier models are loaded only if some pair is not in the score cache
        def __init__(self, ctor):
            self.ctor, self.obj = ctor, None

        def __call__(self):
            if self.obj is None:
                with timed("load_verifier"):
                    self.obj = self.ctor()
            return self.obj

    mc_l, nli_l = _Lazy(lambda: MiniCheck(device=a.verifier_device)), _Lazy(lambda: NLI(device=a.verifier_device))
    mc_cache = ScoreCache(CACHE_DIR / "scores__minicheck.jsonl", f"minicheck@{MINICHECK_REV}")
    nli_cache = ScoreCache(CACHE_DIR / "scores__nli_contradiction.jsonl", f"nli-contradiction@{NLI_REV}")

    def mc_support(docs_: list[str], claims_: list[str]) -> list[float]:
        return mc_cache.get(lambda d, c: mc_l().support_prob(d, c), docs_, claims_)

    gen_verify_unreadable = [0]
    if a.verifier == "minicheck":
        support = mc_support
    elif a.verifier == "alignscore":
        from hallu.verifier import ALIGNSCORE_REV, AlignScore
        al_l = _Lazy(lambda: AlignScore(device=a.verifier_device))
        al_cache = ScoreCache(CACHE_DIR / "scores__alignscore.jsonl", f"alignscore@{ALIGNSCORE_REV}")

        def support(docs_: list[str], claims_: list[str]) -> list[float]:
            return al_cache.get(lambda d, c: al_l().support_prob(d, c), docs_, claims_)
    elif a.verifier == "nli":
        ent_cache = ScoreCache(CACHE_DIR / "scores__nli_entailment.jsonl", f"nli-entailment@{NLI_REV}")

        def support(docs_: list[str], claims_: list[str]) -> list[float]:
            return ent_cache.get(lambda d, c: [x["entailment"] for x in nli_l().probs(d, c)], docs_, claims_)
    else:  # generator as verifier: P(True) from the first token's top-20 logprobs (same reading as closed_book_truth)
        def support(docs_: list[str], claims_: list[str]) -> list[float]:
            outs = llm.chat([Request(user(abl_prompts.fill("generator_verify", passage=d, claim=c)), max_tokens=1, top_logprobs=20,
                                     tag="3b_generator_verify") for d, c in zip(docs_, claims_)])
            vals = []
            for o in outs:
                p = binary_probability(o.get("first_token_top", []), ["true", "yes"], ["false", "no"])
                if p is None:
                    gen_verify_unreadable[0] += 1
                    p = 0.0  # unreadable verdict = no evidence of support
                vals.append(p)
            return vals

    pairs = [(qi, ci, pi) for qi, cl in enumerate(claims_all) for ci in range(len(cl)) for pi in range(len(hits_all[qi]))]
    docs = [f"{hits_all[q][p]['title']}. {hits_all[q][p]['text']}" for q, _, p in pairs]
    hyps = [claims_all[q][c] for q, c, _ in pairs]
    with timed("support_" + a.verifier):
        sup = support(docs, hyps)
    with timed("nli"):
        con = nli_cache.get(lambda d, c: [x["contradiction"] for x in nli_l().probs(d, c)], docs, hyps)
    if a.stop_after == "score":
        print(f"STOP_AFTER score: {len(pairs)} pairs scored (minicheck misses {mc_cache.misses}, nli misses {nli_cache.misses})")
        return
    with timed("closed_book_truth"):
        flat_claims = [(qi, ci) for qi, cl in enumerate(claims_all) for ci in range(len(cl))]
        cb = llm.chat([Request(user(prompts.fill("closed_book_truth", claim=claims_all[q][c])), max_tokens=1, top_logprobs=20, tag="3_closed_book_truth")
                       for q, c in flat_claims])
    cb_prob = {}
    n_cb_unreadable = 0
    for (q, c), o in zip(flat_claims, cb):
        p = binary_probability(o.get("first_token_top", []), ["true", "yes"], ["false", "no"])
        if p is None:
            n_cb_unreadable += 1
            p = 0.5
        cb_prob[(q, c)] = p

    per_claim: dict[tuple[int, int], dict] = {}
    for (q, c, p), s, cn in zip(pairs, sup, con):
        d = per_claim.setdefault((q, c), {"s_list": [], "con_list": []})
        d["s_list"].append(s)
        d["con_list"].append(cn)
    records: list[list[dict]] = []
    for qi, cl in enumerate(claims_all):
        recs = []
        for ci, text in enumerate(cl):
            d = per_claim.get((qi, ci), {"s_list": [], "con_list": []})
            agg = aggregate_claim_scores([d["s_list"]])[0]
            con_max = max(d["con_list"]) if d["con_list"] else 0.0
            con_idx = int(np.argmax(d["con_list"])) if d["con_list"] else None
            c = cb_prob[(qi, ci)]
            recs.append({"text": text, "s": agg["support"], "best_passage": agg["best"], "con": con_max, "con_passage": con_idx,
                         "c": c, "score_c": score_two_signal(agg["support"], c),
                         "action": triage(agg["support"], con_max, c, th, one_axis=(a.triage == "one_axis"))})
        records.append(recs)

    # ---- 6 repair
    rev = [(qi, ci) for qi, recs in enumerate(records) for ci, r in enumerate(recs) if r["action"] == REVISE]
    rer = [(qi, ci) for qi, recs in enumerate(records) for ci, r in enumerate(recs) if r["action"] == RERETRIEVE]
    with timed("revise"):
        rev_out = llm.chat([Request(user(prompts.fill("revise_claim", claim=records[q][c]["text"],
                                                       passage=f"{hits_all[q][records[q][c]['con_passage']]['title']}. {hits_all[q][records[q][c]['con_passage']]['text']}")),
                                    max_tokens=96, tag="4_revise") for q, c in rev])
    new_texts = []
    for o in rev_out:
        t = o["text"].strip().splitlines()[0].strip() if o["text"].strip() else ""
        new_texts.append(None if (not t or t.upper().startswith("DELETE")) else t)
    with timed("rescore_revised"):
        rpairs = [(i, p) for i, ((q, c), t) in enumerate(zip(rev, new_texts)) if t is not None for p in range(len(hits_all[q]))]
        rs = support([f"{hits_all[rev[i][0]][p]['title']}. {hits_all[rev[i][0]][p]['text']}" for i, p in rpairs], [new_texts[i] for i, _ in rpairs])
    best_after: dict[int, float] = {}
    for (i, _), s in zip(rpairs, rs):
        best_after[i] = max(best_after.get(i, 0.0), s)
    for i, ((q, c), t) in enumerate(zip(rev, new_texts)):
        records[q][c]["repair"] = {"text": t, "s_after": best_after.get(i, 0.0), "prompt_tokens": rev_out[i]["prompt_tokens"],
                                   "completion_tokens": rev_out[i]["completion_tokens"]}
    with timed("reretrieve"):
        new_hits = bm25.search([f"{retrieval_query(items[q])} {records[q][c]['text']}" for q, c in rer], a.k)
    with timed("rescore_reretrieved"):
        qpairs = [(i, p) for i in range(len(rer)) for p in range(len(new_hits[i]))]
        qs = support([f"{new_hits[i][p]['title']}. {new_hits[i][p]['text']}" for i, p in qpairs], [records[rer[i][0]][rer[i][1]]["text"] for i, _ in qpairs])
    best_new: dict[int, tuple[float, int]] = {}
    for (i, p), s in zip(qpairs, qs):
        if s > best_new.get(i, (-1.0, -1))[0]:
            best_new[i] = (s, p)
    for i, (q, c) in enumerate(rer):
        s_new, p_new = best_new.get(i, (0.0, None))
        records[q][c]["repair"] = {"text": records[q][c]["text"], "s_after": s_new,
                                   "new_passage": ({"pid": new_hits[i][p_new]["pid"], "title": new_hits[i][p_new]["title"]} if p_new is not None else None)}

    # ---- write
    store = JsonlStore(out_dir / "items.jsonl")
    done = store.load()
    rows = []
    for it, hits, g, d, recs in zip(items, hits_all, gens, decs, records):
        row = {"id": it["id"], "question": it["question"], "answer": g["text"], "answer_finish_reason": g["finish_reason"],
               "passages": [{"pid": h["pid"], "title": h["title"]} for h in hits], "decomposition_raw": d["text"], "claims": recs,
               "final_claims_A": final_claims(recs)}
        if it["id"] not in done:
            store.append(row)
        rows.append(row)

    n_claims = [len(r["claims"]) for r in rows]
    acts = [c["action"] for r in rows for c in r["claims"]]
    repaired = [c for r in rows for c in r["claims"] if c.get("repair")]
    summary = {
        "config": cfg, "config_hash": config_hash(cfg), "git_commit": git_commit(), "n": len(rows),
        "claims_per_response_mean": float(np.mean(n_claims)), "responses_with_zero_claims": int(sum(n == 0 for n in n_claims)),
        "answers_truncated": int(sum(r["answer_finish_reason"] == "length" for r in rows)),
        "decompositions_truncated": int(sum(d["finish_reason"] == "length" for d in decs)),
        "actions": {k: acts.count(k) for k in (KEEP, REVISE, RERETRIEVE, LOW)},
        "closed_book_unreadable": n_cb_unreadable,
        **({"generator_verify_unreadable": gen_verify_unreadable[0]} if a.verifier == "generator" else {}),
        "repair": {"revise_attempted": len(rev), "revise_deleted": int(sum(t is None for t in new_texts)),
                   "revise_now_supported": int(sum(1 for c in repaired if c["action"] == REVISE and c["repair"]["text"] and c["repair"]["s_after"] >= a.tau_s)),
                   "reretrieve_attempted": len(rer),
                   "reretrieve_now_supported": int(sum(1 for c in repaired if c["action"] == RERETRIEVE and c["repair"]["s_after"] >= a.tau_s))},
        "mean_support_all_claims": float(np.mean([c["s"] for r in rows for c in r["claims"]])) if acts else None,
        "cost": {"llm_by_stage": llm.usage_summary(), "retrieval_queries": bm25.calls, "verifier_pairs": {"minicheck": len(pairs) + len(rpairs) + len(qpairs), "nli": len(pairs)},
                 "verifier_cache": {"minicheck_hits": mc_cache.hits, "minicheck_misses": mc_cache.misses,
                                    "nli_hits": nli_cache.hits, "nli_misses": nli_cache.misses},
                 "seconds_by_stage": {k: round(v, 2) for k, v in timing.items()}, "verifier_device": a.verifier_device or "auto",
                 "note": "stage seconds include cache hits; use a cold cache for timing claims"},
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("n", "claims_per_response_mean", "responses_with_zero_claims", "answers_truncated",
                                              "decompositions_truncated", "actions", "closed_book_unreadable", "repair", "mean_support_all_claims")}))
    print("seconds:", summary["cost"]["seconds_by_stage"])
    print("llm:", json.dumps(summary["cost"]["llm_by_stage"]))
    print("wrote", out_dir / "summary.json")


if __name__ == "__main__":
    main()
