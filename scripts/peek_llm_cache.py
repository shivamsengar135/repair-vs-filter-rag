"""Show a few cached LLM outputs for one model and stage tag. Usage: python scripts/peek_llm_cache.py MODEL_TAG TAG [N]"""
import json
import sys
from collections import Counter

from hallu.runutil import CACHE_DIR

tag_model, stage, n = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4
shown, kinds = 0, Counter()
with open(CACHE_DIR / f"llm__{tag_model}.jsonl", encoding="utf-8") as fh:
    for line in fh:
        r = json.loads(line)
        if stage not in json.dumps(r)[:4000]:
            continue
        text = r.get("text") or r.get("response", {}).get("text") or ""
        kinds[text.strip()[:30]] += 1
        if shown < n:
            print("KEYS", list(r)[:12])
            print("TEXT", repr(text[:500]))
            shown += 1
print("most common output prefixes:", kinds.most_common(8))
