"""Rough diagnostic of a generator's long-form behaviour from its LLM cache (no tags are stored, so outputs are
classified by text): decompositions that say "No claims." and long answers that decline for lack of information.
Usage: python scripts/count_refusals.py MODEL_TAG"""
import json
import re
import sys

from hallu.runutil import CACHE_DIR

decl = re.compile(r"(does not|doesn't|do not|don't) (contain|provide|include|mention)|no (relevant )?information|not (enough|sufficient) information|isn't a direct|there is no", re.I)
n = no_claims = bullets = declines = long_answers = 0
with open(CACHE_DIR / f"llm__{sys.argv[1]}.jsonl", encoding="utf-8") as fh:
    for line in fh:
        t = (json.loads(line).get("text") or "").strip()
        n += 1
        if t.lower().startswith("no claims"):
            no_claims += 1
        elif t.startswith("- "):
            bullets += 1
        elif len(t) > 200:
            long_answers += 1
            declines += bool(decl.search(t[:400]))
print(json.dumps({"records": n, "decomposition_no_claims": no_claims, "decomposition_bulleted": bullets,
                  "long_texts": long_answers, "long_texts_declining_in_first_400_chars": declines}))
