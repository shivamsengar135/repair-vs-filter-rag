import json

import pytest

from hallu import parsing as P
from hallu.store import JsonlStore, config_hash, read_jsonl, text_key, write_jsonl


def test_parse_claims_numbered_and_bulleted():
    out = P.parse_claims("Here are the claims:\n1. Paris is in France.\n2) It has 2 million people.\n- It is old.\n\nDone")
    assert out == ["Paris is in France.", "It has 2 million people.", "It is old."]


def test_parse_claims_json_list():
    assert P.parse_claims('Sure: ["A is B.", "C is D."]') == ["A is B.", "C is D."]
    # invalid JSON falls back to bullet parsing
    assert P.parse_claims('["A is B.", \n- C is D.') == ["C is D."]


def test_parse_claims_dedup_empty_and_cap():
    assert P.parse_claims("- A is B.\n- a  is b.\n- C.") == ["A is B."]  # duplicate removed, 2-char item dropped
    assert P.parse_claims("") == []
    assert P.parse_claims("No claims.") == []
    assert P.parse_claims("- None") == []
    assert P.parse_claims("just prose without list markers") == []
    assert len(P.parse_claims("\n".join(f"- claim number {i}" for i in range(100)), max_claims=10)) == 10


def test_parse_claims_ignores_decimal_numbers_in_text():
    # "3.5 million" must not be read as list item "3."
    assert P.parse_claims("- The city has 3.5 million people.") == ["The city has 3.5 million people."]
    assert P.parse_claims("3.5 million people live there.") == []


def test_parse_yes_no():
    assert P.parse_yes_no("Yes, it is.") is True
    assert P.parse_yes_no(" TRUE") is True
    assert P.parse_yes_no("Supported.") is True
    assert P.parse_yes_no("No.") is False
    assert P.parse_yes_no("Not supported by the passage") is False
    assert P.parse_yes_no("Unsupported") is False
    assert P.parse_yes_no("The passage says yes") is None
    assert P.parse_yes_no("") is None
    assert P.parse_yes_no("Nothing here") is None  # "No" must be a whole word
    assert P.parse_yes_no("Yesterday") is None


def test_is_abstention_and_first_line():
    assert P.is_abstention("I don't know.")
    assert P.is_abstention("")
    assert P.is_abstention("There is not enough information to answer.")
    assert not P.is_abstention("Paris")
    assert P.first_line("\n\nAnswer: Paris\nBecause...") == "Paris"
    assert P.first_line("Final answer - 42") == "42"
    assert P.first_line("") == ""


def test_config_hash_is_order_independent_and_sensitive():
    assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})
    assert text_key("a", "bc") != text_key("ab", "c")


def test_jsonl_store_resume_and_duplicates(tmp_path):
    p = tmp_path / "s.jsonl"
    s = JsonlStore(p)
    s.append({"id": "1", "x": 1})
    s.append({"id": "2", "x": 2})
    with pytest.raises(ValueError):
        s.append({"id": "1", "x": 9})
    with pytest.raises(ValueError):
        s.append({"x": 9})
    # simulate a crash mid-write: truncated last line must be ignored and the item redone
    with p.open("a") as f:
        f.write('{"id": "3", "x": ')
    s2 = JsonlStore(p)
    assert s2.ids() == {"1", "2"}
    assert [it["id"] for it in s2.todo([{"id": "1"}, {"id": "3"}])] == ["3"]
    assert s2.load()["2"]["x"] == 2


def test_skip_mode_tolerates_concurrent_writers(tmp_path):
    p = tmp_path / "cache.jsonl"
    a, b = JsonlStore(p, on_duplicate="skip"), JsonlStore(p, on_duplicate="skip")
    assert a.append({"id": "k", "v": 1}) is True
    # b loads its id set only now (after a's write): the duplicate is skipped instead of raising
    assert b.append({"id": "k", "v": 1}) is False
    # two writers that never saw each other: both lines land on disk; readers still get one record per id
    c, d = JsonlStore(p, on_duplicate="skip"), JsonlStore(p, on_duplicate="skip")
    c.ids(); d.ids()
    c.append({"id": "z", "v": 2}); d.append({"id": "z", "v": 2})
    assert sum(1 for _ in JsonlStore(p)) == 3 and JsonlStore(p).load()["z"]["v"] == 2
    with pytest.raises(ValueError):
        JsonlStore(p, on_duplicate="bad")


def test_write_jsonl_is_deterministic(tmp_path):
    rows = [{"id": "a", "q": "x"}, {"id": "b", "q": "y"}]
    h1 = write_jsonl(tmp_path / "a.jsonl", rows)
    h2 = write_jsonl(tmp_path / "b.jsonl", [dict(reversed(list(r.items()))) for r in rows])
    assert h1 == h2 and len(h1) == 64
    assert read_jsonl(tmp_path / "a.jsonl") == rows
    assert json.loads((tmp_path / "a.jsonl").read_text().splitlines()[0]) == rows[0]
