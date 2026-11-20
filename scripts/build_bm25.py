"""Phase 4: build the BM25 index over psgs_w100 (resumable in two stages).

Stage 1 (tokenise): one pass over the gzipped TSV -> passages.bin / passage_offsets.npy / passage_ids.npy,
          token_ids.npy (flat int32) / token_offsets.npy, tokenizer vocabulary.  Marker: stage1.done
Stage 2 (index):   bm25s index from the token ids -> <out>/bm25s/ + build_meta.json.            Marker: stage2.done

Usage: python scripts/build_bm25.py --tsv ~/halluc_data/corpus/psgs_w100.tsv.gz --out ~/halluc_data/index/bm25_psgs_w100 [--limit N]
"""
from __future__ import annotations

import argparse
import array
import csv
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

from hallu.retrieval import TOKENIZER_SPEC, BM25Retriever, TokenIdCorpus, make_tokenizer

BATCH = 50_000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stage1(tsv: Path, out: Path, limit: int | None) -> None:
    tok = make_tokenizer()
    csv.field_size_limit(10_000_000)
    flat = array.array("i")
    tok_off = array.array("q", [0])
    p_off = array.array("q", [0])
    pids = array.array("q")
    n = 0
    t0 = time.time()
    with gzip.open(tsv, "rt", encoding="utf-8", newline="") as f, open(out / "passages.bin", "wb") as blob:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        assert header == ["id", "text", "title"], header
        batch_texts: list[str] = []

        def flush() -> None:
            if not batch_texts:
                return
            ids = tok.tokenize(batch_texts, update_vocab=True, return_as="ids", show_progress=False)
            for doc in ids:
                flat.extend(doc)
                tok_off.append(len(flat))
            batch_texts.clear()

        pos = 0
        for row in reader:
            if len(row) != 3:
                raise ValueError(f"bad row at passage {n}: {row[:1]}")
            pid, text, title = row
            rec = f"{title}\t{text}".encode("utf-8")
            blob.write(rec)
            pos += len(rec)
            p_off.append(pos)
            pids.append(int(pid))
            batch_texts.append(f"{title} {text}")
            n += 1
            if len(batch_texts) >= BATCH:
                flush()
                if n % 1_000_000 == 0:
                    log(f"stage1: {n:,} passages, {len(flat):,} tokens, {time.time() - t0:.0f}s")
            if limit and n >= limit:
                break
        flush()
    assert len(tok_off) == n + 1 == len(p_off)
    np.save(out / "token_ids.npy", np.frombuffer(flat, dtype=np.int32))
    np.save(out / "token_offsets.npy", np.frombuffer(tok_off, dtype=np.int64))
    np.save(out / "passage_offsets.npy", np.frombuffer(p_off, dtype=np.int64))
    np.save(out / "passage_ids.npy", np.frombuffer(pids, dtype=np.int64))
    (out / "bm25s").mkdir(exist_ok=True)
    tok.save_vocab(str(out / "bm25s"))
    vocab = tok.get_vocab_dict()
    (out / "stage1.done").write_text(json.dumps({"n_passages": n, "n_tokens": len(flat), "vocab_size": len(vocab),
                                                 "seconds": round(time.time() - t0)}))
    log(f"stage1 done: {n:,} passages, {len(flat):,} tokens, vocab {len(vocab):,}")


def stage2(out: Path, tsv: Path, limit: int | None) -> None:
    import bm25s

    t0 = time.time()
    tok = make_tokenizer()
    tok.load_vocab(str(out / "bm25s"))
    vocab = tok.get_vocab_dict()
    flat = np.load(out / "token_ids.npy", mmap_mode="r")
    off = np.load(out / "token_offsets.npy")
    corpus = TokenIdCorpus(flat, off)
    log(f"stage2: indexing {len(corpus):,} docs, vocab {len(vocab):,}")
    bm = bm25s.BM25()  # library defaults: recorded below
    # bm25s accepts any object exposing `.ids` (per-document token ids) and `.vocab`; our ids are a lazy view
    from types import SimpleNamespace
    bm.index(SimpleNamespace(ids=corpus, vocab=vocab), show_progress=False)
    bm.save(str(out / "bm25s"))
    tok.save_vocab(str(out / "bm25s"))
    meta = {"bm25s_version": bm25s.__version__, "bm25_params": {"k1": bm.k1, "b": bm.b, "delta": bm.delta, "method": bm.method},
            "tokenizer": TOKENIZER_SPEC, "n_passages": len(corpus), "source": str(tsv), "limit": limit,
            "index_seconds": round(time.time() - t0), "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "build_meta.json").write_text(json.dumps(meta, indent=1))
    (out / "stage2.done").write_text("ok")
    log(f"stage2 done in {time.time() - t0:.0f}s: {json.dumps(meta['bm25_params'])}")


def selfcheck(out: Path) -> None:
    r = BM25Retriever(out)
    # third query: tokens that cannot occur in the corpus (a first attempt used "xxyyzz", which IS in Wikipedia)
    qs = ["Who is Aaron, the brother of Moses, in the Abrahamic religions?", "", "qzjxkvwqzjxkvwqzjxkvw wqkzxjvwqkzxjvwqkzxjv"]
    t0 = time.perf_counter()  # monotonic: the WSL wall clock was observed jumping backwards (2026-11-13)
    res = r.search(qs, k=5)
    log(f"selfcheck: search took {time.perf_counter() - t0:.2f}s")
    for q, hits in zip(qs, res):
        print(f"  Q={q!r}: " + " | ".join(f"{h['pid']}:{h['title']}:{h['score']:.1f}" for h in hits))
    assert res[0] and res[0][0]["title"] == "Aaron", "expected an 'Aaron' passage at rank 1"
    assert res[1] == [] and res[2] == [], "empty / out-of-vocabulary queries must return no hits"
    p0 = r.passages.get(0)
    assert p0["pid"] == 1 and p0["title"] == "Aaron" and p0["text"].startswith("Aaron"), p0
    print("SELFCHECK_OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    tsv, out = Path(a.tsv).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "stage1.done").exists():
        stage1(tsv, out, a.limit)
    else:
        log("stage1 already done: " + (out / "stage1.done").read_text())
    if not (out / "stage2.done").exists():
        stage2(out, tsv, a.limit)
    else:
        log("stage2 already done")
    selfcheck(out)
    print("BUILD_BM25_OK")


if __name__ == "__main__":
    sys.exit(main())
