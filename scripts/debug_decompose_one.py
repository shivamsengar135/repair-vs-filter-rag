"""Show the raw decomposition output (v2) for zero-claim responses of a run. Usage: debug_decompose_one.py ITEMS MODEL [n]"""
import sys

from hallu.llm import LLM, Request, user
from hallu.parsing import parse_claims
from hallu.prompts import Prompts
from hallu.runutil import CACHE_DIR
from hallu.store import read_jsonl

rows = [r for r in read_jsonl(sys.argv[1]) if not r["claims"]]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
llm = LLM(model=sys.argv[2], cache_path=CACHE_DIR / f"llm__{sys.argv[2].split('/')[-1]}.jsonl")
P = Prompts("configs/prompts/v2.yaml")
for r in rows[:n]:
    o = llm.chat([Request(user(P.fill("decompose", question=r["question"], text=r["answer"])), max_tokens=1024, tag="debug")])[0]
    print(f"### {r['question'][:60]} | answer {len(r['answer'])} chars, starts {r['answer'][:120]!r}")
    print(f"raw ({o['finish_reason']}): {o['text'][:500]!r}")
    print("parsed:", parse_claims(o["text"])[:3], "\n")
