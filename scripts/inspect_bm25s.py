"""Print bm25s internals relevant to indexing a large corpus from pre-tokenised ids."""
import inspect

import bm25s

print("bm25s", bm25s.__version__)
print(inspect.getsource(bm25s.BM25._infer_corpus_object))
src = inspect.getsource(bm25s.BM25.index)
i = src.find("inferred_corpus_obj = ")
print(src[i:i + 2600])
