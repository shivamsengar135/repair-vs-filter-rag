import math
from types import SimpleNamespace

import pytest

from hallu.llm import LLM, Request, binary_probability, user


def test_binary_probability_normalises_and_merges_variants():
    top = [{"token": " True", "logprob": math.log(0.6)}, {"token": "true", "logprob": math.log(0.1)},
           {"token": "False", "logprob": math.log(0.2)}, {"token": "The", "logprob": math.log(0.1)}]
    assert binary_probability(top, ["true", "yes"], ["false", "no"]) == pytest.approx(0.7 / 0.9)
    assert binary_probability([{"token": "Yes.", "logprob": 0.0}], ["yes"], ["no"]) == 1.0
    assert binary_probability([{"token": "The", "logprob": 0.0}], ["yes"], ["no"]) is None
    assert binary_probability([], ["yes"], ["no"]) is None


class FakeCompletions:
    def __init__(self, fail_first: int = 0):
        self.calls = 0
        self.fail_first = fail_first

    async def create(self, **kw):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise ConnectionError("boom")
        text = "echo:" + kw["messages"][-1]["content"]
        lp = None
        if kw.get("logprobs"):
            lp = SimpleNamespace(content=[SimpleNamespace(top_logprobs=[SimpleNamespace(token="Yes", logprob=-0.1)])])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop", logprobs=lp)],
                               usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3))


def make(tmp_path, **kw):
    comp = FakeCompletions(**kw)
    client = SimpleNamespace(chat=SimpleNamespace(completions=comp))
    return LLM(model="m", cache_path=tmp_path / "cache.jsonl", client=client, max_retries=3), comp


def test_chat_order_cache_and_usage(tmp_path, monkeypatch):
    llm, comp = make(tmp_path)
    reqs = [Request(user(f"q{i}"), tag="gen") for i in range(5)]
    out = llm.chat(reqs)
    assert [o["text"] for o in out] == [f"echo:q{i}" for i in range(5)]
    assert comp.calls == 5 and not any(o["cached"] for o in out)
    # second run: all cache hits, no new server calls, but usage is still counted (cost is cache-independent)
    out2 = llm.chat(reqs)
    assert comp.calls == 5 and all(o["cached"] for o in out2)
    u = llm.usage_summary()["gen"]
    assert u == {"calls": 10, "cache_hits": 5, "prompt_tokens": 70, "completion_tokens": 30}
    # a fresh object reads the cache from disk
    llm2, comp2 = make(tmp_path)
    assert llm2.chat(reqs[:2])[0]["cached"] and comp2.calls == 0


def test_cache_key_depends_on_parameters(tmp_path):
    llm, comp = make(tmp_path)
    llm.chat([Request(user("q"), max_tokens=10)])
    llm.chat([Request(user("q"), max_tokens=20)])
    llm.chat([Request(user("q"), max_tokens=10, temperature=1.0)])
    llm.chat([Request(user("q"), max_tokens=10, temperature=1.0, seed=1)])
    assert comp.calls == 4


def test_retry_then_success_and_logprobs(tmp_path, monkeypatch):
    async def no_sleep(_):
        return None
    monkeypatch.setattr("asyncio.sleep", no_sleep)
    llm, comp = make(tmp_path, fail_first=2)
    out = llm.chat([Request(user("q"), top_logprobs=5)])
    assert comp.calls == 3 and out[0]["first_token_top"] == [{"token": "Yes", "logprob": -0.1}]


def test_new_options_do_not_change_old_keys():
    r = Request(user("q"))
    assert r.key("m") == Request(user("q"), token_logprobs=False, continue_final=False).key("m")
    assert r.key("m") != Request(user("q"), token_logprobs=True).key("m")
    msgs = user("q") + [{"role": "assistant", "content": "partial"}]
    assert Request(msgs).key("m") != Request(msgs, continue_final=True).key("m")
    # the formula used before these options existed (guards the on-disk cache against silent invalidation)
    import json
    from hallu.store import text_key
    msgs = user("hello")
    old = text_key("m", json.dumps(msgs, sort_keys=True, ensure_ascii=False), "8", repr(0.0), "0", "0", json.dumps(None))
    assert Request(msgs, max_tokens=8).key("m") == old


def test_continue_final_and_token_logprobs(tmp_path):
    seen = {}

    class C:
        async def create(self, **kw):
            seen.update(kw)
            toks = [SimpleNamespace(token="Paris", logprob=-0.2, top_logprobs=[]), SimpleNamespace(token=".", logprob=-0.01, top_logprobs=[])]
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Paris."), finish_reason="stop",
                                                            logprobs=SimpleNamespace(content=toks))],
                                   usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2))

    llm = LLM(model="m", cache_path=tmp_path / "c.jsonl", client=SimpleNamespace(chat=SimpleNamespace(completions=C())))
    msgs = user("capital of France?") + [{"role": "assistant", "content": "It is"}]
    out = llm.chat([Request(msgs, token_logprobs=True, continue_final=True)])[0]
    assert seen["extra_body"] == {"continue_final_message": True, "add_generation_prompt": False} and seen["logprobs"] is True
    assert out["tokens"] == [{"token": "Paris", "logprob": -0.2}, {"token": ".", "logprob": -0.01}]
    with pytest.raises(ValueError):
        llm.chat([Request(user("x"), continue_final=True)])


def test_gives_up_after_max_retries(tmp_path, monkeypatch):
    async def no_sleep(_):
        return None
    monkeypatch.setattr("asyncio.sleep", no_sleep)
    llm, _ = make(tmp_path, fail_first=99)
    with pytest.raises(RuntimeError):
        llm.chat([Request(user("q"))])
