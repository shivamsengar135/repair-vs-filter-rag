"""BM25 retrieval over the DPR Wikipedia passages (EXPERIMENT_PLAN.md §3).

Design constraints: 21M passages must be indexed and served inside ~60 GB RAM without Java or a C compiler.
* Passage text lives in one binary file + an offsets array and is memory-mapped (random access by passage index).
* Token ids live in one flat int32 array + offsets; `TokenIdCorpus` presents them to bm25s as a lazy sequence of
  lists, so the corpus never exists as Python lists of ints.
* Tokenisation = bm25s Tokenizer, lower-cased, English stop-words, Snowball English stemmer (PyStemmer).
  Indexed text = title + " " + passage text.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

TOKENIZER_SPEC = {"lower": True, "stopwords": "en", "stemmer": "snowball-english", "indexed_text": "title + ' ' + text"}


def make_tokenizer():
    import bm25s
    import Stemmer

    return bm25s.tokenization.Tokenizer(lower=True, stopwords="en", stemmer=Stemmer.Stemmer("english"))


class TokenIdCorpus:
    """Lazy sequence of token-id lists backed by a flat int32 array and an offsets array (len = n_docs + 1)."""

    def __init__(self, flat: np.ndarray, offsets: np.ndarray):
        self.flat, self.offsets = flat, offsets

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, i: int) -> list[int]:
        if i < 0:
            i += len(self)
        return self.flat[self.offsets[i]:self.offsets[i + 1]].tolist()

    def __iter__(self) -> Iterator[list[int]]:
        flat, off = self.flat, self.offsets
        for i in range(len(off) - 1):
            yield flat[off[i]:off[i + 1]].tolist()


class PassageStore:
    """Memory-mapped passage texts: passages.bin holds UTF-8 "title\\ttext" records back to back."""

    def __init__(self, directory: str | Path):
        d = Path(directory)
        self.offsets = np.load(d / "passage_offsets.npy", mmap_mode="r")
        self.ids = np.load(d / "passage_ids.npy", mmap_mode="r")
        self.blob = np.memmap(d / "passages.bin", dtype=np.uint8, mode="r")

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def get(self, idx: int) -> dict:
        raw = bytes(self.blob[self.offsets[idx]:self.offsets[idx + 1]]).decode("utf-8")
        title, _, text = raw.partition("\t")
        return {"pid": int(self.ids[idx]), "title": title, "text": text}


class BM25Retriever:
    def __init__(self, index_dir: str | Path, load_passages: bool = True):
        import bm25s

        d = Path(index_dir)
        self.meta = json.loads((d / "build_meta.json").read_text())
        # In-RAM index + numba scorer: 0.11 s/query vs 0.47 s with mmap+numpy, identical top-5
        # (scripts/bench_bm25.py, logs/phase4_bench_bm25.log). Falls back to numpy if numba is unavailable.
        self.bm25 = bm25s.BM25.load(str(d / "bm25s"), mmap=False)
        self.n_threads = 0
        try:
            self.bm25.activate_numba_scorer()
            self.n_threads = -1
        except Exception:  # noqa: BLE001
            pass
        self.tokenizer = make_tokenizer()
        self.tokenizer.load_vocab(str(d / "bm25s"))
        self.passages = PassageStore(d) if load_passages else None

    def search(self, queries: Sequence[str], k: int = 5) -> list[list[dict]]:
        """Top-k passages per query: [{rank, score, pid, title, text}]. Empty queries return []."""
        toks = self.tokenizer.tokenize(list(queries), update_vocab=False, return_as="tuple", show_progress=False)
        idx, scores = self.bm25.retrieve(toks, k=k, show_progress=False, n_threads=self.n_threads)
        out = []
        for qi, q in enumerate(queries):
            hits = []
            if q.strip():
                for r in range(idx.shape[1]):
                    s = float(scores[qi, r])
                    if s <= 0.0:
                        continue  # bm25s pads with zero-score documents when fewer than k match
                    rec = {"rank": len(hits) + 1, "score": s, "doc_index": int(idx[qi, r])}
                    if self.passages is not None:
                        rec.update(self.passages.get(int(idx[qi, r])))
                    hits.append(rec)
            out.append(hits)
        return out


class DenseRetriever:
    """Exact inner-product search over the bge-small passage embeddings (scripts/build_dense_index.py).
    Embeddings are held in RAM as fp16 (~16 GB) and scored on the CPU in fp32 blocks, so search works while vLLM
    holds the GPU. Queries get the model card's retrieval instruction; CLS pooling; L2 normalisation."""

    def __init__(self, index_dir: str | Path, passage_dir: str | Path, device: str = "cpu", block: int = 500_000):
        from transformers import AutoModel, AutoTokenizer

        d = Path(index_dir)
        self.meta = json.loads((d / "build_meta.json").read_text())
        self.emb = np.concatenate([np.load(d / f"emb_{s:03d}.npy") for s in range(self.meta["n_shards"])])
        assert self.emb.shape == (self.meta["n_passages"], self.meta["dim"]), self.emb.shape
        self.tok = AutoTokenizer.from_pretrained(self.meta["model"], revision=self.meta["revision"])
        self.model = AutoModel.from_pretrained(self.meta["model"], revision=self.meta["revision"]).to(device).eval()
        self.device, self.block = device, block
        self.passages = PassageStore(passage_dir)

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        import torch

        texts = [self.meta["query_instruction"] + q for q in queries]
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                enc = self.tok(texts[i:i + 64], max_length=512, truncation=True, padding=True, return_tensors="pt").to(self.device)
                h = self.model(**enc).last_hidden_state[:, 0]
                out.append(torch.nn.functional.normalize(h.float(), p=2, dim=1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def search(self, queries: Sequence[str], k: int = 5) -> list[list[dict]]:
        """Top-k passages per query: [{rank, score, pid, title, text}]. Empty queries return []."""
        live = [i for i, q in enumerate(queries) if q.strip()]
        out: list[list[dict]] = [[] for _ in queries]
        if not live:
            return out
        q = self.encode_queries([queries[i] for i in live])
        best_s = np.full((len(live), k), -np.inf, dtype=np.float32)
        best_i = np.zeros((len(live), k), dtype=np.int64)
        for lo in range(0, len(self.emb), self.block):
            s = q @ self.emb[lo:lo + self.block].astype(np.float32).T
            top = np.argpartition(-s, k - 1, axis=1)[:, :k] if s.shape[1] > k else np.argsort(-s, axis=1)[:, :k]
            cand_s = np.concatenate([best_s, np.take_along_axis(s, top, axis=1)], axis=1)
            cand_i = np.concatenate([best_i, top + lo], axis=1)
            o = np.argsort(-cand_s, axis=1, kind="stable")[:, :k]
            best_s, best_i = np.take_along_axis(cand_s, o, axis=1), np.take_along_axis(cand_i, o, axis=1)
        for row, qi in enumerate(live):
            for r in range(k):
                rec = {"rank": r + 1, "score": float(best_s[row, r]), "doc_index": int(best_i[row, r])}
                rec.update(self.passages.get(int(best_i[row, r])))
                out[qi].append(rec)
        return out


def format_passages(hits: Sequence[dict], max_words_each: int | None = None) -> str:
    """Numbered evidence block used in prompts: "[1] (Title) text"."""
    lines = []
    for i, h in enumerate(hits, 1):
        text = h["text"] if max_words_each is None else " ".join(h["text"].split()[:max_words_each])
        lines.append(f"[{i}] ({h['title']}) {text}")
    return "\n".join(lines)
