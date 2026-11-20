"""FActScore knowledge source and evidence selection, replicating factscore/retrieval.py and factscorer.py
(shmsw25/FActScore, saved in refs/raw/code/, read 2026-11-16) with their BM25 option:

* DB: documents(title PRIMARY KEY, text); a page's passages are joined by SPECIAL_SEPARATOR.
* retrieval query = topic + " " + claim.strip(); BM25Okapi (rank_bm25) over the page's passages, each with "<s>" and
  "</s>" removed and whitespace-split; top-k by argsort(-scores) (all passages if the page has fewer than k).
* prompt context = the k passages in REVERSED order, each "Title: {title}\\nText: {text}\\n\\n" (tags removed),
  preceded by "Answer the question about {topic} based on the given context.\\n\\n"; a final "." is added if the
  context does not end with punctuation; then "\\n\\nInput: {claim} True or False?\\nOutput:".
"""
from __future__ import annotations

import sqlite3
import string
from pathlib import Path

import numpy as np

SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"
DEFAULT_DB = Path.home() / "halluc_data/factscore/enwiki-20230401.db"


class FactScoreKB:
    def __init__(self, db_path: str | Path = DEFAULT_DB):
        self.con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        self._pages: dict[str, list[dict] | None] = {}
        self._bm25: dict[str, object] = {}

    def passages(self, topic: str) -> list[dict] | None:
        """The page's passages, or None if the title is not in the DB (FActScore asserts; we report instead)."""
        if topic not in self._pages:
            rows = self.con.execute("SELECT text FROM documents WHERE title = ?", (topic,)).fetchall()
            self._pages[topic] = [{"title": topic, "text": p} for p in rows[0][0].split(SPECIAL_SEPARATOR)] if len(rows) == 1 else None
        return self._pages[topic]

    def top_k(self, topic: str, claim: str, k: int = 5) -> list[dict]:
        psgs = self.passages(topic)
        if not psgs:
            return []
        if topic not in self._bm25:
            from rank_bm25 import BM25Okapi
            self._bm25[topic] = BM25Okapi([p["text"].replace("<s>", "").replace("</s>", "").split() for p in psgs])
        scores = self._bm25[topic].get_scores((topic + " " + claim.strip()).split())
        return [psgs[i] for i in np.argsort(-scores)[:k]]


def factscore_prompt(topic: str, claim: str, passages: list[dict]) -> str:
    definition = "Answer the question about {} based on the given context.\n\n".format(topic)
    context = ""
    for psg in reversed(passages):
        context += "Title: {}\nText: {}\n\n".format(psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
    definition += context.strip()
    if not definition[-1] in string.punctuation:
        definition += "."
    return "{}\n\nInput: {} True or False?\nOutput:".format(definition.strip(), claim.strip())
