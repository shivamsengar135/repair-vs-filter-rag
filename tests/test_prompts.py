from pathlib import Path

import pytest

from hallu.prompts import Prompts
from hallu.retrieval import TokenIdCorpus, format_passages

import numpy as np

P = Prompts(Path(__file__).resolve().parents[1] / "configs" / "prompts" / "v1.yaml")


def test_all_templates_fill_with_declared_fields():
    assert P.version == "v1"
    for name in P.templates:
        text = P.fill(name, **{f: f"<{f}>" for f in P.fields(name)})
        assert "{" not in text and "}" not in text
        for f in P.fields(name):
            assert f"<{f}>" in text


def test_fill_rejects_missing_and_extra_fields():
    with pytest.raises(KeyError):
        P.fill("short_rag", question="q")
    with pytest.raises(KeyError):
        P.fill("short_closed_book", question="q", passages="p")


def test_expected_fields():
    assert P.fields("short_rag") == {"question", "passages"}
    assert P.fields("decompose") == {"question", "text"}
    assert P.fields("revise_claim") == {"passage", "claim"}


def test_format_passages():
    hits = [{"title": "A", "text": "one two three"}, {"title": "B", "text": "four"}]
    assert format_passages(hits) == "[1] (A) one two three\n[2] (B) four"
    assert format_passages(hits, max_words_each=2) == "[1] (A) one two\n[2] (B) four"
    assert format_passages([]) == ""


def test_token_id_corpus_is_a_lazy_sequence():
    flat = np.array([5, 6, 7, 8, 9], dtype=np.int32)
    off = np.array([0, 2, 2, 5], dtype=np.int64)
    c = TokenIdCorpus(flat, off)
    assert len(c) == 3
    assert list(c) == [[5, 6], [], [7, 8, 9]]
    assert c[0] == [5, 6] and c[-1] == [7, 8, 9]
    assert all(isinstance(x, int) for doc in c for x in doc)
