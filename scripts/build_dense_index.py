"""Encode all DPR psgs_w100 passages with BAAI/bge-small-en-v1.5 (EXPERIMENT_PLAN.md §3 secondary retriever, D6).

Passage text = "title text" (no instruction, per the model card); CLS pooling; L2 normalisation; fp16 storage.
Output: $DATA_ROOT/index/dense_bge_small/emb_{shard:03d}.npy (SHARD passages each, row i = passage index
shard*SHARD + i, same order as the BM25 PassageStore) + build_meta.json. Resumable: finished shards are skipped.
Usage: python scripts/build_dense_index.py [--batch 512] [--max-len 256]"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from hallu.retrieval import PassageStore
from hallu.runutil import DATA_ROOT, DEFAULT_INDEX, git_commit

MODEL, REV = "BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
SHARD = 1_000_000


def main() -> None:
    from transformers import AutoModel, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=256)
    a = ap.parse_args()
    out = DATA_ROOT / "index" / "dense_bge_small"
    out.mkdir(parents=True, exist_ok=True)
    store = PassageStore(DEFAULT_INDEX)
    n = len(store)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    model = AutoModel.from_pretrained(MODEL, revision=REV).to("cuda").half().eval()
    n_shards = (n + SHARD - 1) // SHARD
    t0 = time.perf_counter()
    truncated = 0
    for s in range(n_shards):
        f = out / f"emb_{s:03d}.npy"
        if f.exists():
            continue
        lo, hi = s * SHARD, min(n, (s + 1) * SHARD)
        emb = np.empty((hi - lo, 384), dtype=np.float16)
        ts = time.perf_counter()
        for i in range(lo, hi, a.batch):
            recs = [store.get(j) for j in range(i, min(hi, i + a.batch))]
            texts = [f"{r['title']} {r['text']}" for r in recs]
            enc = tok(texts, max_length=a.max_len, truncation=True, padding=True, return_tensors="pt")
            truncated += int((enc["attention_mask"].sum(1) == a.max_len).sum())
            with torch.no_grad():
                h = model(**{k: v.to("cuda") for k, v in enc.items()}).last_hidden_state[:, 0]
                h = torch.nn.functional.normalize(h.float(), p=2, dim=1)
            emb[i - lo:i - lo + len(recs)] = h.cpu().numpy().astype(np.float16)
        np.save(out / f"emb_{s:03d}.tmp.npy", emb)
        (out / f"emb_{s:03d}.tmp.npy").rename(f)
        el = time.perf_counter() - ts
        print(f"shard {s + 1}/{n_shards} [{lo}, {hi}) {el:.0f}s ({(hi - lo) / el:.0f} passages/s), "
              f"total {time.perf_counter() - t0:.0f}s, truncated so far {truncated}", flush=True)
    meta = {"model": MODEL, "revision": REV, "n_passages": n, "shard_size": SHARD, "n_shards": n_shards, "dim": 384,
            "dtype": "float16", "pooling": "cls", "normalized": True, "passage_text": "title + ' ' + text",
            "max_len": a.max_len, "query_instruction": "Represent this sentence for searching relevant passages: ",
            "git_commit": git_commit()}
    (out / "build_meta.json").write_text(json.dumps(meta, indent=1))
    print("DENSE_INDEX_DONE", json.dumps(meta))


if __name__ == "__main__":
    main()
