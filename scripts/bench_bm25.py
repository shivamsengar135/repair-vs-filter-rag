"""Compare bm25s search configurations on 200 dev questions: speed, and that the top-5 ids are identical."""
import json
import time
from pathlib import Path

import bm25s

from hallu.retrieval import make_tokenizer
from hallu.store import read_jsonl

D = Path.home() / "halluc_data/index/bm25_psgs_w100/bm25s"
qs = [r["question"] for r in read_jsonl("data/splits/hotpotqa_dev.jsonl")[:200]]
tok = make_tokenizer()
tok.load_vocab(str(D))
toks = tok.tokenize(qs, update_vocab=False, return_as="tuple", show_progress=False)
ref = None
for name, mmap, numba in (("mmap+numpy", True, False), ("ram+numpy", False, False), ("ram+numba", False, True)):
    t0 = time.perf_counter()
    bm = bm25s.BM25.load(str(D), mmap=mmap)
    if numba:
        try:
            bm.activate_numba_scorer()
            bm.retrieve(tok.tokenize(["warm up query"], update_vocab=False, return_as="tuple", show_progress=False), k=5, show_progress=False)
        except Exception as e:  # noqa: BLE001
            print(f"{name}: numba unavailable: {type(e).__name__}: {str(e)[:160]}")
            continue
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    idx, sc = bm.retrieve(toks, k=5, show_progress=False, n_threads=0 if not numba else -1)
    dt = time.perf_counter() - t0
    same = None if ref is None else bool((idx == ref).all())
    if ref is None:
        ref = idx.copy()
    print(json.dumps({"config": name, "load_s": round(t_load, 1), "search_s_per_query": round(dt / len(qs), 4), "top5_identical_to_first": same}), flush=True)
    del bm
