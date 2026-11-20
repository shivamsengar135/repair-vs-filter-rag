"""Parsing of LLM outputs: claim lists, yes/no verdicts, short answers. Kept strict and unit-tested because
silent parsing errors would corrupt every downstream number."""
from __future__ import annotations

import json
import re

_BULLET = re.compile(r"^\s*(?:[-*•]|\(?\d{1,3}[.)])\s+(.*\S)\s*$")
_NO_CLAIMS = re.compile(r"^\s*(no (verifiable |factual )?claims?\.?|none\.?|n/a)\s*$", re.I)


def parse_claims(text: str, max_claims: int = 60) -> list[str]:
    """Extract atomic claims from a decomposition output.

    Accepts a JSON list of strings, or one claim per bulleted / numbered line. Lines that are not list items are
    ignored (models often add a preamble). Duplicates are removed preserving order. Returns [] for an explicit
    "no claims" answer or when nothing parses.
    """
    t = text.strip()
    if not t or _NO_CLAIMS.match(t):
        return []
    claims: list[str] = []
    m = re.search(r"\[.*\]", t, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                claims = [x.strip() for x in data]
        except json.JSONDecodeError:
            pass
    if not claims:
        for line in t.splitlines():
            b = _BULLET.match(line)
            if b:
                claims.append(b.group(1).strip())
    out, seen = [], set()
    for c in claims:
        key = " ".join(c.lower().split())
        if len(c) >= 3 and key not in seen and not _NO_CLAIMS.match(c):
            seen.add(key)
            out.append(c)
    return out[:max_claims]


_YES = re.compile(r"^\W*(yes|true|supported|correct)\b", re.I)
_NO = re.compile(r"^\W*(no|false|unsupported|not supported|incorrect|refuted|contradicted)\b", re.I)


def parse_yes_no(text: str) -> bool | None:
    """True / False from the start of a verdict; None if the output starts with neither (caller must count these)."""
    t = text.strip()
    if _NO.match(t):  # check "no"/"not supported" first: "not supported" must not match "supported"
        return False
    if _YES.match(t):
        return True
    return None


_ABSTAIN = re.compile(r"\b(i (do not|don't|cannot|can't) (know|answer|determine|find)|unanswerable|no answer|"
                      r"not enough information|insufficient information|unknown)\b", re.I)


def is_abstention(text: str) -> bool:
    """Heuristic detector of an explicit abstention in a short answer; empty output counts as abstention."""
    t = text.strip()
    return (not t) or bool(_ABSTAIN.search(t))


def first_line(text: str) -> str:
    """Short-form answers: keep the first non-empty line, strip a leading 'Answer:' label."""
    for line in text.splitlines():
        s = line.strip()
        if s:
            return re.sub(r"^(final\s+)?answer\s*[:\-]\s*", "", s, flags=re.I).strip()
    return ""
