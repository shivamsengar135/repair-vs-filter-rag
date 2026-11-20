"""Preflight (no method output involved): does the FActScore KB have a page for every bio_confirm topic?"""
import json
from pathlib import Path

from hallu.factscore_kb import FactScoreKB

kb = FactScoreKB()
topics = [json.loads(l)["meta"]["topic"] for l in Path("data/splits/bio_confirm.jsonl").read_text(encoding="utf-8").splitlines()]
missing = [t for t in topics if not kb.passages(t)]
print(f"{len(topics)} topics, {len(topics) - len(missing)} with a KB page; missing: {missing[:10]}")
for s in ("test",):
    t2 = [json.loads(l)["meta"]["topic"] for l in Path(f"data/splits/bio_{s}.jsonl").read_text(encoding="utf-8").splitlines()]
    print(f"for comparison bio_{s}: {sum(1 for t in t2 if kb.passages(t))}/{len(t2)} with a KB page")
