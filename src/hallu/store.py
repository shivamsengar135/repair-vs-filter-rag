"""Append-only JSONL stores for resumable stages and for caching generations / retrievals.

A stage writes one record per (item id); on restart it loads the ids already present and skips them.
Each store file carries a config hash in its name so results from different configurations never mix.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator


def config_hash(cfg: Any, n: int = 12) -> str:
    """Stable short hash of a JSON-serialisable configuration."""
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:n]


def text_key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


class JsonlStore:
    """One JSON object per line, each with a unique string field `id`. Tolerates a truncated last line.

    on_duplicate="raise" (result files): appending an existing id is an error.
    on_duplicate="skip" (caches shared by concurrent processes): an existing id is silently not re-written.
    Concurrent processes can still both write the same id when neither has seen the other's line yet; readers
    then see duplicate lines, which is harmless for caches because their content is deterministic (load() keeps one).
    """

    def __init__(self, path: str | os.PathLike, on_duplicate: str = "raise"):
        if on_duplicate not in ("raise", "skip"):
            raise ValueError(on_duplicate)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ids: set[str] | None = None
        self.on_duplicate = on_duplicate

    def __iter__(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # partial line from an interrupted write; the item will simply be redone

    def ids(self) -> set[str]:
        if self._ids is None:
            self._ids = {r["id"] for r in self if "id" in r}
        return self._ids

    def append(self, rec: dict) -> bool:
        """Write rec; returns False (nothing written) for a duplicate id in a skip-mode store."""
        if "id" not in rec:
            raise ValueError("record needs an 'id'")
        if rec["id"] in self.ids():
            if self.on_duplicate == "skip":
                return False
            raise ValueError(f"duplicate id {rec['id']!r} in {self.path}")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
        self.ids().add(rec["id"])
        return True

    def todo(self, items: Iterable[dict]) -> list[dict]:
        done = self.ids()
        return [it for it in items if it["id"] not in done]

    def load(self) -> dict[str, dict]:
        return {r["id"]: r for r in self}


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    return list(JsonlStore(path))


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> str:
    """Write rows atomically and return the file's SHA-256 (used for the frozen dev/test splits)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(p)
    return hashlib.sha256(p.read_bytes()).hexdigest()
