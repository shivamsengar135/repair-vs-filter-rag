"""Decision logic of Triage-and-Repair (EXPERIMENT_PLAN.md §5.1-5.3) as pure functions.

Model calls live in scripts/run_longform.py; everything here is deterministic and unit-tested.

Per claim we have:
    s      MiniCheck support probability, max over the passages shown to the generator
    con    NLI contradiction probability, max over the same passages
    c      generator's closed-book P("True") for the claim (0.5 when it could not be read; such cases are counted)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

KEEP, REVISE, RERETRIEVE, LOW = "keep", "revise", "reretrieve", "low"


@dataclass(frozen=True)
class Thresholds:
    tau_s: float = 0.5    # support needed to keep a claim as-is
    tau_c: float = 0.5    # parametric confidence needed to spend a re-retrieval
    tau_con: float = 0.5  # contradiction probability that triggers a revision


def triage(s: float, con: float, c: float, th: Thresholds, one_axis: bool = False) -> str:
    """Action for one claim. Order of tests follows the plan: supported -> contradicted -> confident -> low.
    one_axis=True is ablation §8.4 (no parametric-confidence axis): c is ignored, so every claim that is neither
    supported nor contradicted is re-retrieved (the LOW bucket disappears)."""
    if s >= th.tau_s:
        return KEEP
    if con >= th.tau_con:
        return REVISE
    if one_axis or c >= th.tau_c:
        return RERETRIEVE
    return LOW


def score_two_signal(s: float, c: float) -> float:
    """Candidate C filter score: P(faithful) * 1 + (1 - P(faithful)) * P(true | unfaithful) with
    P(faithful) = s and P(true | unfaithful) = c (EXPERIMENT_PLAN.md §14, clarification C3)."""
    return s + (1.0 - s) * c


def final_claims(claims: Sequence[dict], budget: int | None = None,
                 branches: tuple[str, ...] = (REVISE, RERETRIEVE)) -> list[dict]:
    """Claims of method A's final answer with their filter scores, under a repair budget.

    Each input claim dict has: text, s, action, and for repaired claims `repair` = {text | None, s_after}.
    budget=None applies every attempted repair; budget=B applies only the B repairs with the lowest original
    support s (highest risk first, ties by position). A repair outside the budget leaves the claim as it was.
    A revision that returned DELETE (repair.text is None) removes the claim.
    `branches` restricts which repair branches are applied (ablations §8.2 re-retrieval only, §8.3 revision only);
    the two branches act on disjoint claims and are computed independently, so this equals re-running with one off.
    Returns [{text, score, origin, source_index}] with origin in {original, revised, reretrieved}.
    """
    attempted = [i for i, cl in enumerate(claims) if cl["action"] in branches and cl.get("repair") is not None]
    order = sorted(attempted, key=lambda i: (claims[i]["s"], i))
    allowed = set(order if budget is None else order[:max(0, budget)])
    out = []
    for i, cl in enumerate(claims):
        if i in allowed:
            rep = cl["repair"]
            if cl["action"] == REVISE:
                if rep["text"] is None:
                    continue
                out.append({"text": rep["text"], "score": rep["s_after"], "origin": "revised", "source_index": i})
            else:  # re-retrieval never changes the text; the claim keeps the better of its two support scores
                out.append({"text": cl["text"], "score": max(cl["s"], rep["s_after"]), "origin": "reretrieved", "source_index": i})
        else:
            out.append({"text": cl["text"], "score": cl["s"], "origin": "original", "source_index": i})
    return out


def repairs_applied(claims: Sequence[dict], budget: int | None = None) -> int:
    attempted = [cl for cl in claims if cl["action"] in (REVISE, RERETRIEVE) and cl.get("repair") is not None]
    return len(attempted) if budget is None else min(len(attempted), max(0, budget))
