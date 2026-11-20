"""Draw the 150-claim human-validation sample (EXPERIMENT_PLAN.md §7) from a judged long-form DEV run.

Sampling (seed 20261113): 50 claims from each source — "rag" (original claims, i.e. what standard RAG / CF see),
"repaired" (texts produced by method A's repair step), "kept_by_A" (claims in A's final candidate set) —
each stratified half / half by the judge's label where possible, then shuffled together.
Writes labelling/claims_to_label.csv (what the user sees: no source, no judge label) and
labelling/.key_DO_NOT_OPEN.json (row -> source, judge label, ids). See docs/LABELLING_GUIDE.md.

Usage: python scripts/make_labelling_sample.py --run results/dev/asqa/longform__... --judge-tag phi-4-FP8-dynamic
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from hallu.pipeline import final_claims
from hallu.store import read_jsonl, text_key

SEED, PER_SOURCE = 20261113, 50


def evidence_text(item: dict) -> str:
    m = item["meta"]
    parts = [f"[{i}] ({d['title']}) {d['text']}" for i, d in enumerate(m["oracle_docs"], 1)]
    parts += [f"[reference answer {i}] {a}" for i, a in enumerate(m["gold_long_answers"], 1)]
    return "\n\n".join(parts)


def stratified(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    pos = [x for x in pool if x["judge_label"]]
    neg = [x for x in pool if not x["judge_label"]]
    rng.shuffle(pos)
    rng.shuffle(neg)
    take_neg = min(len(neg), n // 2)
    take_pos = min(len(pos), n - take_neg)
    take_neg = min(len(neg), n - take_pos)
    return pos[:take_pos] + neg[:take_neg]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--judge-tag", required=True)
    ap.add_argument("--out", default="labelling")
    a = ap.parse_args()
    run = Path(a.run)
    cfg = json.loads((run / "summary.json").read_text())["config"]
    if cfg["split"] != "dev":
        raise SystemExit("labelling sample must come from DEV outputs")
    items = {it["id"]: it for it in read_jsonl(f"data/splits/{cfg['dataset']}_{cfg['split']}.jsonl")}
    judged = {r["id"]: r for r in read_jsonl(run / f"judge__{a.judge_tag}.jsonl")}
    pools: dict[str, list[dict]] = {"rag": [], "repaired": [], "kept_by_A": []}
    seen: set[tuple[str, str]] = set()
    for r in read_jsonl(run / "items.jsonl"):
        for c in r["claims"]:
            pools["rag"].append({"item_id": r["id"], "claim": c["text"]})
            rep = c.get("repair")
            if rep and rep.get("text") and rep["text"] != c["text"]:
                pools["repaired"].append({"item_id": r["id"], "claim": rep["text"]})
        for fc in final_claims(r["claims"]):
            pools["kept_by_A"].append({"item_id": r["id"], "claim": fc["text"]})
    rng = random.Random(SEED)
    rows = []
    order = ("repaired", "kept_by_A", "rag")       # rarest pool first so duplicates are taken from where they are scarce
    shortfall = 0                                   # quota a small pool could not fill moves to the next pools
    for n_src, src in enumerate(order):
        pool = []
        for x in pools[src]:
            k = (x["item_id"], x["claim"])
            if k in seen:
                continue
            j = judged[text_key(x["item_id"], x["claim"])]
            pool.append({**x, "source": src, "judge_label": j["label"], "judge_p_true": j["p_true"]})
        remaining_sources = len(order) - n_src
        quota = PER_SOURCE + (shortfall + remaining_sources - 1) // remaining_sources if shortfall else PER_SOURCE
        picked = stratified(pool, quota, rng)
        shortfall = max(0, shortfall + PER_SOURCE - len(picked))
        seen |= {(x["item_id"], x["claim"]) for x in picked}
        rows += picked
        print(f"{src:10s}: pool {len(pool):5d} -> picked {len(picked)} (judge-supported {sum(x['judge_label'] for x in picked)})")
    rng.shuffle(rows)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    with (out / "claims_to_label.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "topic", "claim", "evidence", "label", "note"])
        for i, x in enumerate(rows, 1):
            w.writerow([i, items[x["item_id"]]["question"], x["claim"], evidence_text(items[x["item_id"]]), "", ""])
    (out / ".key_DO_NOT_OPEN.json").write_text(json.dumps({"run": str(run), "judge_tag": a.judge_tag, "seed": SEED,
                                                            "rows": [{"row": i, **x} for i, x in enumerate(rows, 1)]}, indent=1))
    print(f"wrote {out / 'claims_to_label.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
