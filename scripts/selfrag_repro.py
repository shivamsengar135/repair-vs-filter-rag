"""Phase 4 sanity reproduction (EXPERIMENT_PLAN.md §11): Self-RAG 7B on PopQA long-tail, reported 54.9 (match).

Port of the official retrieval_lm/run_short_form.py, function call_model_rerank_w_scores_batch, mode
adaptive_retrieval, with the README's PopQA command: --max_new_tokens 100 --threshold 0.2 --ndocs 10
--use_groundness --use_utility --use_seqscore --dtype half (w_rel = w_sup = w_use = 1.0 from argparse defaults).
Official code saved at refs/raw/code/selfrag_*.py (fetched 2026-11-14).

Faithfully kept, including quirks:
* prompt "### Instruction:\\n{question}\\n\\n### Response:\\n"; evidence appended as
  "[Retrieval]<paragraph>{title}\\n{text}</paragraph>"; the first 10 of the 25 shipped Contriever passages.
* retrieve iff  lp[Retrieval] / (lp[Retrieval] + lp[No Retrieval]) > 0.2 using RAW LOG-probabilities at the first
  generated position (official lines 74-81).
* scores: relevance from position 0; groundedness from the first position whose token is a grounding token;
  utility from the first position whose token is a utility token; final = exp(mean token logprob) + rel + grd + ut.
* answer = highest-scoring evidence path (closed=False for PopQA); control tokens, "</s>" and newlines stripped;
  a leading "#" or ":" removed; metric = official `match` (case-sensitive raw substring of any gold answer).
Necessary differences (stated in the ledger):
* reflection-token log-probabilities come from vLLM's `logprob_token_ids` (exact values) instead of the
  top-32016 / top-5000 lists of vLLM 0.2.x; the original sets tokens outside the top 5000 to -100. Only tokens with
  probability below 1/5000 can differ.
* the official script as published passes max_depth to a function that does not accept it (TypeError); max_depth is
  unused by the decoding, so it is simply not passed.
* served by vLLM 0.33.0 (OpenAI completions endpoint) in float16, greedy, seed 0.

Usage: python scripts/selfrag_repro.py [--limit N]   (needs vLLM serving selfrag/selfrag_llama2_7b on :8000)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import numpy as np

from hallu.runutil import CACHE_DIR, DATA_ROOT, git_commit
from hallu.store import JsonlStore, text_key

MODEL = "selfrag/selfrag_llama2_7b"
DATA = DATA_ROOT / "selfrag_eval/a/eval_data/popqa_longtail_w_gs.jsonl"
CONTROL = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]",
           "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]
RET = ["[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]"]
REL = ["[Irrelevant]", "[Relevant]"]
GRD = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]"]
UT = [f"[Utility:{i}]" for i in range(1, 6)]
THRESHOLD, NDOCS, MAX_NEW, W_REL, W_SUP, W_USE = 0.2, 10, 100, 1.0, 1.0, 1.0


def official_match(prediction: str, ground_truth: list[str]) -> int:
    for gt in ground_truth:
        if gt in prediction:
            return 1
    return 0


def postprocess(answer: str) -> str:
    for t in CONTROL:
        answer = answer.replace(t, "")
    return answer.replace("</s>", "").replace("\n", "").replace("<|endoftext|>", "")


def parse(choice: dict) -> dict:
    """tokens (ids), sampled-token logprobs, and per-position {token_id: logprob} for the requested ids."""
    lp = choice["logprobs"]
    ids = [int(t.split(":", 1)[1]) for t in lp["tokens"]]
    top = [{int(k.split(":", 1)[1]): v for k, v in (d or {}).items()} for d in lp["top_logprobs"]]
    return {"text": choice["text"], "ids": ids, "token_logprobs": lp["token_logprobs"], "top": top}


class Client:
    def __init__(self, base_url: str, cache: Path, concurrency: int = 32):
        self.base_url = base_url
        self.client = None   # created inside each asyncio.run: its connection pool is bound to one event loop
        self.store = JsonlStore(cache, on_duplicate="skip")
        self.mem = self.store.load()
        self.concurrency = concurrency
        self.sem: asyncio.Semaphore | None = None   # created inside each asyncio.run (a semaphore is bound to one loop)
        self.calls = self.prompt_tokens = self.completion_tokens = 0

    async def one(self, prompt: str, ids: list[int]) -> dict:
        key = text_key("selfrag-repro-v1", MODEL, prompt, json.dumps(ids))
        if key not in self.mem:
            async with self.sem:
                r = await self.client.completions.create(
                    model=MODEL, prompt=prompt, max_tokens=MAX_NEW, temperature=0.0, top_p=1.0, seed=0, logprobs=1,
                    extra_body={"logprob_token_ids": ids, "return_tokens_as_token_ids": True})
            ch = r.choices[0].model_dump()
            rec = {"id": key, **parse(ch), "prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens}
            if key not in self.mem:
                self.mem[key] = rec
                self.store.append(rec)
        rec = self.mem[key]
        self.calls += 1
        self.prompt_tokens += rec["prompt_tokens"]
        self.completion_tokens += rec["completion_tokens"]
        return rec

    def run(self, jobs: list[tuple[str, list[int]]]) -> list[dict]:
        async def go():
            from openai import AsyncOpenAI
            self.sem = asyncio.Semaphore(self.concurrency)
            async with AsyncOpenAI(base_url=self.base_url, api_key="EMPTY", timeout=600) as client:
                self.client = client
                return await asyncio.gather(*(self.one(p, i) for p, i in jobs))
        return asyncio.run(go())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--dataset", default="repro", help="'repro' = official PopQA long-tail file + its Contriever passages; "
                    "otherwise one of our short-form datasets with OUR BM25 top-10 passages (baseline use)")
    ap.add_argument("--split", default="dev")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tid = {t: tok.convert_tokens_to_ids(t) for t in RET + REL + GRD + UT}
    assert all(isinstance(v, int) and v >= 32000 for v in tid.values()), tid  # added tokens sit above the Llama-2 vocab
    if a.dataset == "repro":
        rows = [json.loads(l) for l in DATA.read_text().splitlines()]
    else:
        from hallu.runutil import CachedBM25
        from hallu.store import read_jsonl
        rows = read_jsonl(f"data/splits/{a.dataset}_{a.split}.jsonl")
        hits = CachedBM25().search([r["question"] for r in rows], NDOCS)
        for r, h in zip(rows, hits):
            r["ctxs"] = [{"title": x["title"], "text": x["text"]} for x in h]
    if a.limit:
        rows = rows[:a.limit]
    cli = Client(a.base_url, CACHE_DIR / "selfrag_repro.jsonl")
    t0 = time.perf_counter()

    prompts = [f"### Instruction:\n{r['question']}\n\n### Response:\n" for r in rows]
    first = cli.run([(p, [tid[t] for t in RET]) for p in prompts])
    do_ret = []
    for f in first:
        sd = {t: f["top"][0].get(tid[t], -100) for t in RET} if f["top"] else {t: -100 for t in RET}
        do_ret.append(sd["[Retrieval]"] / (sd["[Retrieval]"] + sd["[No Retrieval]"]) > THRESHOLD)

    ev_jobs, ev_owner = [], []
    for i, (r, p) in enumerate(zip(rows, prompts)):
        if do_ret[i]:
            for para in r["ctxs"][:NDOCS]:
                ev_jobs.append((p + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(para["title"], para["text"]), [tid[t] for t in REL + GRD + UT]))
                ev_owner.append(i)
    nr_jobs = [(p + "[No Retrieval]", [tid["[Retrieval]"]]) for i, p in enumerate(prompts) if not do_ret[i]]
    ev_out = cli.run(ev_jobs)
    nr_out = cli.run(nr_jobs)

    best: dict[int, tuple[float, str]] = {}
    for i, o in zip(ev_owner, ev_out):
        top = o["top"]
        rel = {t: math.exp(top[0].get(tid[t], -100)) for t in REL} if top else {t: math.exp(-100) for t in REL}
        relevance = rel["[Relevant]"] / sum(rel.values())
        g_idx = next((k for k, x in enumerate(o["ids"]) if x in {tid[t] for t in GRD}), None)
        if g_idx is not None:
            g = {t: math.exp(top[g_idx].get(tid[t], -100)) for t in GRD}
            ground = g["[Fully supported]"] / sum(g.values()) + 0.5 * g["[Partially supported]"] / sum(g.values())
        else:
            ground = 0.0
        u_idx = next((k for k, x in enumerate(o["ids"]) if x in {tid[t] for t in UT}), None)
        if u_idx is not None:
            u = {t: math.exp(top[u_idx].get(tid[t], -100)) for t in UT}
            utility = sum(w * u[f"[Utility:{k + 1}]"] / sum(u.values()) for k, w in enumerate([-1, -0.5, 0, 0.5, 1]))
        else:
            utility = 0.0
        seq = sum(x for x in o["token_logprobs"] if x is not None) / max(len(o["ids"]), 1)
        score = math.exp(seq) + W_REL * relevance + W_SUP * ground + W_USE * utility
        if i not in best or score > best[i][0]:   # official: sorted(..., reverse=True)[0] -> first max wins on ties
            best[i] = (score, o["text"])

    preds, nr_iter = [], iter(nr_out)
    for i in range(len(rows)):
        pred = postprocess(best[i][1] if do_ret[i] else next(nr_iter)["text"])
        if pred and pred[0] in "#:":
            pred = pred[1:]
        preds.append(pred)
    from hallu import metrics as M
    scores = [official_match(p, r["answers"]) for p, r in zip(preds, rows)]
    ours = [M.match_accuracy(p, r["answers"]) for p, r in zip(preds, rows)]
    out = {"n": len(rows), "dataset": a.dataset, "split": a.split if a.dataset != "repro" else "official PopQA long-tail",
           "match_official": float(np.mean(scores)), "match_ours_normalised": float(np.mean(ours)), "retrieval_frequency": float(np.mean(do_ret)),
           **({"reported_in_paper": 0.549, "data_sha256": "3b9ae00ff8ad42995bf5b0f7c5b920e369c8be2c2f2772d3a8cf728d4795efc3"} if a.dataset == "repro" else
              {"passages": f"our BM25 top-{NDOCS}"}),
           "model": MODEL,
           "settings": {"threshold": THRESHOLD, "ndocs": NDOCS, "max_new_tokens": MAX_NEW, "w_rel": W_REL, "w_sup": W_SUP, "w_use": W_USE, "dtype": "float16"},
           "cost": {"calls": cli.calls, "prompt_tokens": cli.prompt_tokens, "completion_tokens": cli.completion_tokens,
                    "seconds": round(time.perf_counter() - t0, 1)}, "git_commit": git_commit(), **({"limit": a.limit} if a.limit else {})}
    od = Path("results/phase4/selfrag_repro") if a.dataset == "repro" else Path("results") / a.split / a.dataset / "selfrag__selfrag_llama2_7b"
    od.mkdir(parents=True, exist_ok=True)
    name = "summary.json" if not a.limit else f"summary_limit{a.limit}.json"
    (od / name).write_text(json.dumps(out, indent=1))
    with (od / ("predictions.jsonl" if not a.limit else f"predictions_limit{a.limit}.jsonl")).open("w") as f:
        for r, p, s, o, d in zip(rows, preds, scores, ours, do_ret):
            f.write(json.dumps({"id": r.get("id"), "question": r["question"], "answers": r["answers"], "pred": p, "match_official": s,
                                "match": o, "retrieved": bool(d)}) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
