"""Phase 4: FIXED dev/test splits for the long-form datasets (EXPERIMENT_PLAN.md §4).

* bio  : FActScore's 500 unlabelled prompt entities -> 30% dev (150) / 70% test (350). Prompt as in the FActScore
         README: "Question: Tell me a bio of <entity>."  The 183 human-labelled entities are NOT included (they
         are reserved for judge validation).
* asqa : the 948 ASQA items distributed by ALCE -> 200 dev / 748 test.
Same conventions as scripts/build_splits.py (seeded, sorted, refuses to overwrite with different content).
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

from hallu.store import write_jsonl

BASE_SEED = 20261113
OUT = Path("data/splits")
ROOT = Path.home() / "halluc_data"


def bio() -> dict[str, list[dict]]:
    ents = [l.strip() for l in (ROOT / "factscore/data/unlabeled/prompt_entities.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    labeled = {l.strip() for l in (ROOT / "factscore/data/labeled/prompt_entities.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
    assert len(ents) == len(set(ents)) == 500, len(ents)
    overlap = sorted(set(ents) & labeled)
    rows = [{"id": "bio-" + hashlib.sha1(e.encode()).hexdigest()[:12], "dataset": "bio", "question": f"Question: Tell me a bio of {e}.",
             "answers": [], "meta": {"topic": e}} for e in sorted(ents) if e not in labeled]
    print(f"bio: {len(ents)} unlabelled entities, {len(overlap)} also in the labelled set (excluded): {overlap[:5]}", file=sys.stderr)
    rng = random.Random(f"{BASE_SEED}-bio")
    rng.shuffle(rows)
    n_dev = round(0.30 * len(rows))
    return {"dev": sorted(rows[:n_dev], key=lambda r: r["id"]), "test": sorted(rows[n_dev:], key=lambda r: r["id"])}


def asqa() -> dict[str, list[dict]]:
    full = json.load(open(ROOT / "alce/ALCE-data/asqa_eval_gtr_top100.json"))
    oracle = {x["sample_id"]: x for x in json.load(open(ROOT / "alce/ALCE-data/asqa_eval_gtr_top100_reranked_oracle.json"))}
    rows = []
    for x in full:
        rows.append({"id": f"asqa-{x['sample_id']}", "dataset": "asqa", "question": x["question"],
                     "answers": [],
                     "meta": {"qa_pairs": [{"question": q["question"], "short_answers": q["short_answers"]} for q in x["qa_pairs"]],
                              "gold_long_answers": [a["long_answer"] for a in x["annotations"]],
                              # judge evidence only (never shown to a method): ALCE's oracle-reranked passages
                              "oracle_docs": [{"id": d["id"], "title": d["title"], "text": d["text"]} for d in oracle[x["sample_id"]]["docs"]]}})
    rows.sort(key=lambda r: r["id"])
    rng = random.Random(f"{BASE_SEED}-asqa")
    rng.shuffle(rows)
    return {"dev": sorted(rows[:200], key=lambda r: r["id"]), "test": sorted(rows[200:], key=lambda r: r["id"])}


def main() -> None:
    man_path = OUT / "MANIFEST.json"
    manifest = json.loads(man_path.read_text())
    for name, fn in (("bio", bio), ("asqa", asqa)):
        parts = fn()
        assert not ({r["id"] for r in parts["dev"]} & {r["id"] for r in parts["test"]})
        for split, rows in parts.items():
            dst = OUT / f"{name}_{split}.jsonl"
            blob = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode()
            sha = hashlib.sha256(blob).hexdigest()
            if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() != sha:
                sys.exit(f"REFUSING to overwrite {dst}: content would change (frozen split).")
            assert write_jsonl(dst, rows) == sha
            manifest[f"{name}_{split}"] = {"n": len(rows), "sha256": sha, "seed": f"{BASE_SEED}-{name}"}
            print(f"{name:5s} {split:4s} n={len(rows):4d} sha256={sha[:16]}  e.g. {rows[0]['question'][:70]!r}")
    man_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
