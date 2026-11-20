import math

from hallu.flare import first_sentence, masked_query, needs_retrieval, sentence_tokens


def simple_split(t):
    parts, cur = [], ""
    for ch in t:
        cur += ch
        if ch in ".!?":
            parts.append(cur.strip())
            cur = ""
    if cur.strip():
        parts.append(cur.strip())
    return parts


def T(tok, p):
    return {"token": tok, "logprob": math.log(p)}


TOKS = [T(" Joe", 0.99), T(" Biden", 0.95), T(" attended", 0.9), T(" the", 0.97), T(" University", 0.6),
        T(" of", 0.98), T(" Pennsylvania", 0.3), T(".", 0.99), T(" He", 0.9), T(" was", 0.9)]


def test_first_sentence():
    assert first_sentence("  Joe Biden attended Penn. He was born.", simple_split) == "Joe Biden attended Penn."
    assert first_sentence("", simple_split) == ""
    assert first_sentence("no stop", simple_split) == "no stop"


def test_sentence_tokens_covers_exactly_the_sentence():
    s = "Joe Biden attended the University of Pennsylvania."
    toks = sentence_tokens(TOKS, s)
    assert len(toks) == 8 and toks[-1]["token"] == "."
    assert sentence_tokens(TOKS, "") == TOKS[:1]  # degenerate case: one token
    assert sentence_tokens([], "x") == []


def test_needs_retrieval_and_mask():
    toks = TOKS[:8]
    assert needs_retrieval(toks, 0.8)            # 0.6 and 0.3 are below 0.8
    assert not needs_retrieval(toks, 0.2)
    assert not needs_retrieval([], 0.8)
    assert masked_query(toks, 0.4) == "Joe Biden attended the University of."   # only "Pennsylvania" (0.3) masked
    assert masked_query(toks, 0.7) == "Joe Biden attended the of."
    assert masked_query(toks, 0.0) == "Joe Biden attended the University of Pennsylvania."
