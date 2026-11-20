"""Phase 4: build the FIXED dev/test subsets of the QA datasets (EXPERIMENT_PLAN.md §4).

Sampled once with base seed 20261113, written to data/splits/<dataset>_<dev|test>.jsonl with a manifest that
records sizes, source split, and SHA-256. Re-running reproduces identical files (the script refuses to overwrite
a file whose content would differ). Unified record: {id, dataset, question, answers, meta}.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset

from hallu.store import write_jsonl

BASE_SEED = 20261113
N_DEV, N_TEST = 500, 1000
OUT = Path("data/splits")


def popqa():
    d = load_dataset("akariasai/PopQA")["test"]
    rows = [{"id": f"popqa-{r['id']}", "question": r["question"], "answers": json.loads(r["possible_answers"]),
             "meta": {"prop": r["prop"], "s_pop": r["s_pop"], "subj": r["subj"]}} for r in d]
    return {"dev": ("test", rows), "test": ("test", rows)}  # single split: disjoint samples of it


def nq():
    ds = load_dataset("google-research-datasets/nq_open", "nq_open")
    out = {}
    for ours, theirs in (("dev", "train"), ("test", "validation")):
        out[ours] = (theirs, [{"id": f"nq-{theirs}-{i}", "question": r["question"], "answers": list(r["answer"]), "meta": {}}
                              for i, r in enumerate(ds[theirs])])
    return out


def triviaqa():
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")
    out = {}
    for ours, theirs in (("dev", "train"), ("test", "validation")):
        rows = []
        for r in ds[theirs]:
            a = r["answer"]
            answers = list(dict.fromkeys([a["value"], *a["aliases"]]))
            rows.append({"id": f"tqa-{r['question_id']}", "question": r["question"], "answers": answers, "meta": {}})
        out[ours] = (theirs, rows)
    return out


def hotpotqa():
    ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki")
    out = {}
    for ours, theirs in (("dev", "train"), ("test", "validation")):
        out[ours] = (theirs, [{"id": f"hotpot-{r['id']}", "question": r["question"], "answers": [r["answer"]],
                               "meta": {"type": r["type"], "level": r["level"]}} for r in ds[theirs]])
    return out


def twowiki():
    base = "hf://datasets/xanhho/2WikiMultihopQA/"
    ds = load_dataset("parquet", data_files={"train": base + "train.parquet", "dev": base + "dev.parquet"})
    cols = ds["dev"].column_names
    print("  2wiki columns:", cols, file=sys.stderr)
    idc = "_id" if "_id" in cols else "id"
    out = {}
    for ours, theirs in (("dev", "train"), ("test", "dev")):
        out[ours] = (theirs, [{"id": f"2wiki-{r[idc]}", "question": r["question"], "answers": [r["answer"]],
                               "meta": {"type": r.get("type")}} for r in ds[theirs]])
    return out


LOADERS = {"popqa": popqa, "nq": nq, "triviaqa": triviaqa, "hotpotqa": hotpotqa, "2wiki": twowiki}


def sample(rows: list[dict], n: int, seed_tag: str, exclude: set[str]) -> list[dict]:
    """Deterministic sample: de-duplicate ids, drop items with no usable answer, sort by id, then seeded sample."""
    seen, pool = set(), []
    for r in rows:
        answers = [a for a in r["answers"] if isinstance(a, str) and a.strip()]
        if r["id"] in seen or r["id"] in exclude or not answers or not r["question"].strip():
            continue
        seen.add(r["id"])
        pool.append({**r, "answers": answers})
    pool.sort(key=lambda r: r["id"])
    rng = random.Random(f"{BASE_SEED}-{seed_tag}")
    picked = rng.sample(pool, min(n, len(pool)))
    return sorted(picked, key=lambda r: r["id"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    man_path = OUT / "MANIFEST.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    for name in (sys.argv[1:] or LOADERS):
        parts = LOADERS[name]()
        used: set[str] = set()
        for ours, n in (("dev", N_DEV), ("test", N_TEST)):
            src_split, rows = parts[ours]
            picked = sample(rows, n, f"{name}-{ours}", used if name == "popqa" else set())
            used |= {r["id"] for r in picked}
            for r in picked:
                r["dataset"] = name
            dst = OUT / f"{name}_{ours}.jsonl"
            blob = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in picked).encode()
            sha = hashlib.sha256(blob).hexdigest()
            if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() != sha:
                sys.exit(f"REFUSING to overwrite {dst}: content would change (frozen split). Investigate.")
            assert write_jsonl(dst, picked) == sha
            manifest[f"{name}_{ours}"] = {"n": len(picked), "requested": n, "source_split": src_split,
                                          "source_rows": len(rows), "sha256": sha, "seed": f"{BASE_SEED}-{name}-{ours}"}
            print(f"{name:9s} {ours:4s} n={len(picked):4d} from {src_split} ({len(rows)} rows) sha256={sha[:16]}  e.g. {picked[0]['question'][:60]!r} -> {picked[0]['answers'][:2]}")
        if name == "popqa":
            a = {json.loads(l)["id"] for l in (OUT / "popqa_dev.jsonl").read_text().splitlines()}
            b = {json.loads(l)["id"] for l in (OUT / "popqa_test.jsonl").read_text().splitlines()}
            assert not (a & b), "PopQA dev/test overlap"
    man_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print("wrote", man_path)


if __name__ == "__main__":
    main()
