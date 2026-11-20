from hallu.verifier import ScoreCache


def test_score_cache_calls_only_for_missing(tmp_path):
    calls = []

    def fn(docs, claims):
        calls.append(len(docs))
        return [len(d) + len(c) for d, c in zip(docs, claims)]

    c = ScoreCache(tmp_path / "s.jsonl", "m")
    assert c.get(fn, ["ab", "c"], ["x", "yz"]) == [3, 3]
    assert c.get(fn, ["ab", "new"], ["x", "q"]) == [3, 4]
    assert calls == [2, 1] and (c.hits, c.misses) == (1, 3)
    c2 = ScoreCache(tmp_path / "s.jsonl", "m")              # reloads from disk
    assert c2.get(fn, ["c", "ab"], ["yz", "x"]) == [3, 3] and calls == [2, 1]
    c3 = ScoreCache(tmp_path / "s.jsonl", "other-model")    # name is part of the key
    c3.get(fn, ["ab"], ["x"])
    assert calls == [2, 1, 1]
    assert ScoreCache(tmp_path / "s.jsonl", "m").get(fn, [], []) == []
