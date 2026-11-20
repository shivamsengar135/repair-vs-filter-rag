"""SelfCheckGPT-NLI scores for the claims of an existing long-form run (baseline; EXPERIMENT_PLAN.md §5.4, deviation D2).

Scoring replicates the official implementation (potsawee/selfcheckgpt, SelfCheckNLI.predict, read 2026-11-14):
model `potsawee/deberta-v3-large-mnli` (two classes; index 1 = contradiction), input pair = (text to check, sample),
score = mean over samples of P(contradiction). Differences from the paper, stated in D2: the unit checked is an
atomic claim of the answer (not a sentence), and the N samples are stochastic re-generations of the SAME RAG
prompt, so the baseline filters exactly the claims that CF / C / A filter.

Stage "sample" needs the generator on vLLM; stage "score" needs the NLI model (GPU if free, else CPU — slow).
Output: <run>/selfcheck_nli.jsonl  rows {id, claim_scores: [mean P(contradiction) per claim], n_samples}

Usage: python scripts/run_selfcheck.py --run results/dev/bio/longform__... --stage sample|score [--device cuda]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hallu.llm import LLM, Request, user
from hallu.prompts import Prompts
from hallu.retrieval import format_passages
from hallu.runutil import CACHE_DIR, CachedBM25
from hallu.store import JsonlStore, read_jsonl

NLI_MODEL = "potsawee/deberta-v3-large-mnli"
# Pinned to the authors' main revision, which ships only pytorch_model.bin; without use_safetensors=False transformers
# fetches a bot-converted model.safetensors from an unmerged PR (refs/pr/2) — observed 2026-11-16.
NLI_REV = "83f4359862aba52c07506cd23d4454db77945b36"
N_SAMPLES = 5
TEMPERATURE = 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stage", required=True, choices=["sample", "score"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed-set", type=int, default=0, help="k -> sample seeds 5k+1..5k+5; output selfcheck_nli.jsonl (k=0) "
                    "or selfcheck_nli_seedset{k}.jsonl (EXPERIMENT_PLAN.md §10: >= 3 seeds for sampled methods)")
    a = ap.parse_args()
    seeds = range(1 + N_SAMPLES * a.seed_set, 1 + N_SAMPLES * (a.seed_set + 1))
    out_name = "selfcheck_nli.jsonl" if a.seed_set == 0 else f"selfcheck_nli_seedset{a.seed_set}.jsonl"
    run = Path(a.run)
    cfg = json.loads((run / "summary.json").read_text())["config"]
    prompts = Prompts(f"configs/prompts/{cfg['prompts']}.yaml")
    items = {it["id"]: it for it in read_jsonl(f"data/splits/{cfg['dataset']}_{cfg['split']}.jsonl")}
    rows = read_jsonl(run / "items.jsonl")
    tag = cfg["model"].split("/")[-1]
    llm = LLM(model=cfg["model"], cache_path=CACHE_DIR / f"llm__{tag}.jsonl")
    bm25 = CachedBM25()
    queries = [items[r["id"]]["meta"]["topic"] if cfg["dataset"] == "bio" else r["question"] for r in rows]
    hits = bm25.search(queries, cfg["k"])
    for r, h in zip(rows, hits):  # must be the passages the original answer saw
        assert [x["pid"] for x in h] == [p["pid"] for p in r["passages"]], f"retrieval mismatch for {r['id']}"
    reqs = [Request(user(prompts.fill("long_rag", question=r["question"], passages=format_passages(h))), max_tokens=cfg["max_tokens"],
                    temperature=TEMPERATURE, seed=s, tag="selfcheck_samples")
            for r, h in zip(rows, hits) for s in seeds]
    outs = llm.chat(reqs)   # in stage "score" these are all cache hits (and are still counted as cost)
    samples = [[outs[i * N_SAMPLES + j]["text"] for j in range(N_SAMPLES)] for i in range(len(rows))]
    print("samples:", json.dumps(llm.usage_summary()))
    if a.stage == "sample":
        return

    import torch
    from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = DebertaV2Tokenizer.from_pretrained(NLI_MODEL, revision=NLI_REV)
    model = DebertaV2ForSequenceClassification.from_pretrained(NLI_MODEL, revision=NLI_REV, use_safetensors=False).to(device).eval()
    assert model.config.num_labels == 2, model.config.num_labels
    pairs = [(ri, ci, si) for ri, r in enumerate(rows) for ci in range(len(r["claims"])) for si in range(N_SAMPLES)]
    probs = np.zeros(len(pairs), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(pairs), a.batch_size):
            chunk = pairs[i:i + a.batch_size]
            # official code: batch_encode_plus(pairs, padding="longest", truncation=True, ...); transformers 5 removed
            # batch_encode_plus, the direct call with text / text_pair has the same semantics
            enc = tok(text=[rows[r]["claims"][c]["text"] for r, c, _ in chunk], text_pair=[samples[r][s] for r, _, s in chunk],
                      add_special_tokens=True, padding="longest", truncation=True, return_tensors="pt",
                      return_token_type_ids=True, return_attention_mask=True).to(device)
            probs[i:i + len(chunk)] = torch.softmax(model(**enc).logits.float(), dim=-1)[:, 1].cpu().numpy()
    acc: dict[tuple[int, int], list[float]] = {}
    for (r, c, _), p in zip(pairs, probs):
        acc.setdefault((r, c), []).append(float(p))
    store = JsonlStore(run / out_name)
    done = store.ids()
    for ri, r in enumerate(rows):
        if r["id"] not in done:
            store.append({"id": r["id"], "n_samples": N_SAMPLES, "temperature": TEMPERATURE,
                          "claim_scores": [float(np.mean(acc[(ri, ci)])) for ci in range(len(r["claims"]))]})
    allp = [float(np.mean(v)) for v in acc.values()]
    print(json.dumps({"n_claims": len(allp), "mean_contradiction": float(np.mean(allp)) if allp else None, "device": device, "nli_pairs": len(pairs)}))


if __name__ == "__main__":
    main()
