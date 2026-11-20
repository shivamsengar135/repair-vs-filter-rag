"""Client for the vLLM OpenAI-compatible server with disk caching and cost accounting.

* Every request is cached on disk under a key derived from (model, messages, sampling parameters), so stages are
  resumable and repeated configurations cost nothing to re-run.
* Cost accounting counts the tokens of every request a method makes, INCLUDING cache hits (the stored usage is
  re-counted), so a method's reported cost does not depend on what happened to be cached.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from hallu.store import JsonlStore, text_key


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, rec: dict, cached: bool) -> None:
        self.calls += 1
        self.cache_hits += int(cached)
        self.prompt_tokens += rec["prompt_tokens"]
        self.completion_tokens += rec["completion_tokens"]

    def as_dict(self) -> dict:
        return {"calls": self.calls, "cache_hits": self.cache_hits, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens}


@dataclass
class Request:
    messages: list[dict]
    max_tokens: int = 256
    temperature: float = 0.0
    seed: int = 0
    top_logprobs: int = 0          # > 0 requests first-token alternatives (for yes/no probabilities)
    stop: list[str] | None = None
    tag: str = ""                  # stage label for per-stage cost accounting
    token_logprobs: bool = False   # store the logprob of EVERY generated token (FLARE)
    continue_final: bool = False   # last message is a partial assistant turn to be continued (vLLM extension)

    def key(self, model: str) -> str:
        import json
        parts = [model, json.dumps(self.messages, sort_keys=True, ensure_ascii=False), str(self.max_tokens),
                 repr(float(self.temperature)), str(self.seed), str(self.top_logprobs), json.dumps(self.stop)]
        # new options are appended only when used, so keys of earlier (cached) requests are unchanged
        if self.token_logprobs:
            parts.append("token_logprobs")
        if self.continue_final:
            parts.append("continue_final")
        return text_key(*parts)


def binary_probability(top: Sequence[dict], positive: Sequence[str], negative: Sequence[str]) -> float | None:
    """P(positive) / (P(positive) + P(negative)) from first-token alternatives [{token, logprob}].

    Tokens are compared after stripping whitespace/punctuation and lower-casing. Returns None when neither class
    appears among the alternatives (the caller must count and report these cases, never silently default).
    """
    pos = {p.lower() for p in positive}
    neg = {n.lower() for n in negative}
    p_pos = p_neg = 0.0
    for alt in top:
        t = alt["token"].strip().strip(".,:;!\"'()[]").lower()
        if t in pos:
            p_pos += math.exp(alt["logprob"])
        elif t in neg:
            p_neg += math.exp(alt["logprob"])
    if p_pos + p_neg <= 0.0:
        return None
    return p_pos / (p_pos + p_neg)


@dataclass
class LLM:
    model: str
    cache_path: str | Path
    base_url: str = "http://127.0.0.1:8000/v1"
    concurrency: int = 32
    max_retries: int = 4
    usage: dict[str, Usage] = field(default_factory=dict)
    client: Any = None             # injectable for tests; must provide chat.completions.create (async)

    def __post_init__(self) -> None:
        self.store = JsonlStore(self.cache_path, on_duplicate="skip")  # shared by concurrent processes
        self._mem: dict[str, dict] = self.store.load()
        # Without an injected client, a fresh AsyncOpenAI client is opened inside every asyncio.run (see _run): an
        # async client's connection pool is bound to one event loop, and reusing it after that loop closed fails
        # with "Event loop is closed" (observed 2026-11-16 in the Self-RAG script).
        self._own_client = self.client is None

    async def _one(self, req: Request, sem: asyncio.Semaphore) -> dict:
        key = req.key(self.model)
        if key in self._mem:
            return {**self._mem[key], "cached": True}
        kwargs: dict[str, Any] = dict(model=self.model, messages=req.messages, max_tokens=req.max_tokens,
                                      temperature=req.temperature, seed=req.seed)
        if req.stop:
            kwargs["stop"] = req.stop
        if req.top_logprobs:
            kwargs.update(logprobs=True, top_logprobs=req.top_logprobs)
        if req.token_logprobs:
            kwargs["logprobs"] = True
        if req.continue_final:
            if req.messages[-1]["role"] != "assistant":
                raise ValueError("continue_final needs a final assistant message")
            kwargs["extra_body"] = {"continue_final_message": True, "add_generation_prompt": False}
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                async with sem:
                    r = await self.client.chat.completions.create(**kwargs)
                break
            except Exception as e:  # noqa: BLE001  (network / server errors: retry with back-off)
                last = e
                await asyncio.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"LLM request failed after {self.max_retries} attempts: {last}")
        ch = r.choices[0]
        rec = {"id": key, "text": ch.message.content or "", "finish_reason": ch.finish_reason,
               "prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens}
        if req.top_logprobs and ch.logprobs and ch.logprobs.content:
            rec["first_token_top"] = [{"token": a.token, "logprob": a.logprob} for a in ch.logprobs.content[0].top_logprobs]
        if req.token_logprobs and ch.logprobs and ch.logprobs.content:
            rec["tokens"] = [{"token": t.token, "logprob": t.logprob} for t in ch.logprobs.content]
        if key not in self._mem:  # two identical concurrent requests may both miss the cache; store once
            self._mem[key] = rec
            self.store.append(rec)
        return {**rec, "cached": False}

    async def _run(self, reqs: Sequence[Request]) -> list[dict]:
        sem = asyncio.Semaphore(self.concurrency)
        if not self._own_client:
            return list(await asyncio.gather(*(self._one(r, sem) for r in reqs)))
        if all(r.key(self.model) in self._mem for r in reqs):   # fully cached: no connection needed
            return list(await asyncio.gather(*(self._one(r, sem) for r in reqs)))
        from openai import AsyncOpenAI
        async with AsyncOpenAI(base_url=self.base_url, api_key="EMPTY", timeout=600) as client:
            self.client = client
            try:
                return list(await asyncio.gather(*(self._one(r, sem) for r in reqs)))
            finally:
                self.client = None

    def chat(self, reqs: Sequence[Request]) -> list[dict]:
        """Run requests concurrently; results are in input order. Updates per-tag usage counters."""
        out = asyncio.run(self._run(reqs))
        for req, rec in zip(reqs, out):
            self.usage.setdefault(req.tag or "untagged", Usage()).add(rec, rec["cached"])
        return out

    def usage_summary(self) -> dict:
        return {tag: u.as_dict() for tag, u in sorted(self.usage.items())}


def user(content: str) -> list[dict]:
    return [{"role": "user", "content": content}]
