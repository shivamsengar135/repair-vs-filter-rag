"""Pure helpers for FLARE-direct (Jiang et al., arXiv:2305.06983) — unit-tested, no model calls.

Paper settings used (Appendix A/C, read 2026-11-14): look-ahead of 64 tokens; first sentence by NLTK Punkt;
retrieval triggered if ANY token of the tentative sentence has probability < theta; implicit query = the sentence
with tokens of probability < beta masked out; theta = 0.8, beta = 0.4 (their 2WikiMultihopQA / ASQA setting).
"""
from __future__ import annotations

import math
from typing import Sequence


def first_sentence(text: str, sent_tokenize) -> str:
    """First sentence of `text` (keeps leading whitespace out); '' for empty text."""
    t = text.lstrip()
    if not t:
        return ""
    sents = sent_tokenize(t)
    return sents[0] if sents else t


def sentence_tokens(tokens: Sequence[dict], sentence: str) -> list[dict]:
    """Prefix of the generated tokens that covers `sentence` (tokens carry leading spaces; matched on stripped text).

    Returns tokens until their concatenation, with leading whitespace removed, reaches len(sentence).
    """
    out, acc = [], ""
    target = len(sentence)
    for t in tokens:
        out.append(t)
        acc += t["token"]
        if len(acc.lstrip()) >= target:
            break
    return out


def needs_retrieval(toks: Sequence[dict], theta: float) -> bool:
    """True if any token probability is below theta (empty token list -> False)."""
    return any(math.exp(t["logprob"]) < theta for t in toks)


def masked_query(toks: Sequence[dict], beta: float) -> str:
    """Implicit query: the sentence with tokens of probability < beta removed; whitespace normalised."""
    kept = "".join(t["token"] for t in toks if math.exp(t["logprob"]) >= beta)
    return " ".join(kept.split())
