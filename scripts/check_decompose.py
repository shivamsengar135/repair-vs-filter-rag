"""Compare decomposition prompt versions on the zero-claim responses of a dev run (bring-up check, dev only).
Usage: python scripts/check_decompose.py ITEMS.jsonl MODEL [old_version new_version]"""
import sys

from hallu.llm import LLM, Request, user
from hallu.parsing import parse_claims
from hallu.prompts import Prompts
from hallu.runutil import CACHE_DIR
from hallu.store import read_jsonl

rows = read_jsonl(sys.argv[1])
model = sys.argv[2]
old, new = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else ("v1", "v2")
llm = LLM(model=model, cache_path=CACHE_DIR / f"llm__{model.split('/')[-1]}.jsonl")
counts = {}
for ver in (old, new):
    P = Prompts(f"configs/prompts/{ver}.yaml")
    outs = llm.chat([Request(user(P.fill("decompose", question=r["question"], text=r["answer"])), max_tokens=1024, tag=f"decompose_{ver}") for r in rows])
    counts[ver] = [len(parse_claims(o["text"])) for o in outs]
z_old = [i for i, n in enumerate(counts[old]) if n == 0]
print(f"responses: {len(rows)} | zero claims {old}: {len(z_old)}, {new}: {sum(n == 0 for n in counts[new])} | "
      f"mean claims {old}: {sum(counts[old]) / len(rows):.2f}, {new}: {sum(counts[new]) / len(rows):.2f}")
print(f"of the {len(z_old)} zero-claim responses under {old}, {sum(counts[new][i] > 0 for i in z_old)} get claims under {new}")
changed = [i for i in range(len(rows)) if counts[old][i] != counts[new][i] and counts[old][i] > 0]
print(f"responses that had claims under {old} and changed count under {new}: {len(changed)}")
P = Prompts(f"configs/prompts/{new}.yaml")
for i in z_old[:4]:
    o = llm.chat([Request(user(P.fill("decompose", question=rows[i]["question"], text=rows[i]["answer"])), max_tokens=1024, tag=f"decompose_{new}")])[0]
    print(f"\n- {rows[i]['question'][:60]} | answer: {rows[i]['answer'][:160]!r}\n  {new} claims: {parse_claims(o['text'])[:4]}")
