import numpy as np

from hallu.retrieval import DenseRetriever


class _Store:
    def get(self, i):
        return {"pid": i, "title": f"t{i}", "text": f"x{i}"}


def _retriever(emb, block, qvecs):
    r = DenseRetriever.__new__(DenseRetriever)  # skip model loading
    r.emb, r.block, r.passages = emb, block, _Store()
    r.encode_queries = lambda qs: qvecs[: len(qs)]
    return r


def test_blockwise_topk_matches_bruteforce():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((1037, 16)).astype(np.float16)
    q = rng.standard_normal((5, 16)).astype(np.float32)
    exact = np.argsort(-(q @ emb.astype(np.float32).T), axis=1, kind="stable")[:, :7]
    for block in (100, 333, 5000):
        hits = _retriever(emb, block, q).search([f"q{i}" for i in range(5)], k=7)
        got = np.array([[h["doc_index"] for h in hs] for hs in hits])
        assert (got == exact).all(), block
        assert all(hs[0]["rank"] == 1 and hs[-1]["rank"] == 7 for hs in hits)


def test_empty_query_returns_no_hits():
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((50, 8)).astype(np.float16)
    q = rng.standard_normal((1, 8)).astype(np.float32)
    hits = _retriever(emb, 20, q).search(["  ", "real"], k=3)
    assert hits[0] == [] and len(hits[1]) == 3
