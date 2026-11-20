"""Shared helpers for experiment entry points: provenance, cached retrieval, paths."""
from __future__ import annotations

import subprocess
from pathlib import Path

from hallu.store import JsonlStore, text_key

DATA_ROOT = Path.home() / "halluc_data"
CACHE_DIR = DATA_ROOT / "cache"
DEFAULT_INDEX = DATA_ROOT / "index" / "bm25_psgs_w100"

_HIT_FIELDS = ("rank", "score", "pid", "title", "text")


def git_commit() -> str:
    try:
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


class CachedBM25:
    """BM25 search with a disk cache. The 28 GB index is loaded lazily, only if some query is not cached."""

    def __init__(self, index_dir: Path = DEFAULT_INDEX):
        self.index_dir = Path(index_dir)
        self.cache = JsonlStore(CACHE_DIR / f"retrieval__{self.index_dir.name}.jsonl", on_duplicate="skip")
        self._have = self.cache.load()
        self._retriever = None
        self.calls = 0  # number of queries requested (cache hits included): this is the method's retrieval cost

    def search(self, queries: list[str], k: int) -> list[list[dict]]:
        self.calls += len(queries)
        keys = [text_key("bm25", self.index_dir.name, str(k), q) for q in queries]
        missing = [(key, q) for key, q in dict(zip(keys, queries)).items() if key not in self._have]
        if missing:
            if self._retriever is None:
                from hallu.retrieval import BM25Retriever
                self._retriever = BM25Retriever(self.index_dir)
            for i in range(0, len(missing), 256):
                chunk = missing[i:i + 256]
                for (key, q), hits in zip(chunk, self._retriever.search([q for _, q in chunk], k=k)):
                    rec = {"id": key, "query": q, "k": k, "hits": [{f: h[f] for f in _HIT_FIELDS} for h in hits]}
                    self.cache.append(rec)
                    self._have[key] = rec
        return [self._have[key]["hits"] for key in keys]


DENSE_INDEX = DATA_ROOT / "index" / "dense_bge_small"


class CachedDense(CachedBM25):
    """Same interface and caching as CachedBM25, over the bge-small dense index (EXPERIMENT_PLAN.md D6).
    The 16 GB embedding matrix is loaded lazily, only if some query is not cached."""

    def __init__(self, index_dir: Path = DENSE_INDEX, passage_dir: Path = DEFAULT_INDEX):
        self.index_dir, self.passage_dir = Path(index_dir), Path(passage_dir)
        self.cache = JsonlStore(CACHE_DIR / f"retrieval__{self.index_dir.name}.jsonl", on_duplicate="skip")
        self._have = self.cache.load()
        self._retriever = None
        self.calls = 0

    def search(self, queries: list[str], k: int) -> list[list[dict]]:
        self.calls += len(queries)
        keys = [text_key("dense", self.index_dir.name, str(k), q) for q in queries]
        missing = [(key, q) for key, q in dict(zip(keys, queries)).items() if key not in self._have]
        if missing:
            if self._retriever is None:
                from hallu.retrieval import DenseRetriever
                self._retriever = DenseRetriever(self.index_dir, self.passage_dir)
            for i in range(0, len(missing), 1024):
                chunk = missing[i:i + 1024]
                for (key, q), hits in zip(chunk, self._retriever.search([q for _, q in chunk], k=k)):
                    rec = {"id": key, "query": q, "k": k, "hits": [{f: h[f] for f in _HIT_FIELDS} for h in hits]}
                    self.cache.append(rec)
                    self._have[key] = rec
        return [self._have[key]["hits"] for key in keys]


def make_retriever(kind: str = "bm25"):
    """'bm25' (primary, §3) or 'dense' (bge-small sensitivity, D6)."""
    if kind == "bm25":
        return CachedBM25()
    if kind == "dense":
        return CachedDense()
    raise ValueError(kind)
