"""Extract the ALCE archive and print the ASQA file schemas (read-only inspection)."""
import json
import tarfile
from pathlib import Path

root = Path.home() / "halluc_data" / "alce"
if not (root / "ALCE-data").exists():
    with tarfile.open(root / "ALCE-data.tar") as t:
        t.extractall(root, filter="data")
for f in sorted((root / "ALCE-data").glob("*.json")):
    print(f"{f.stat().st_size / 1e6:8.1f} MB  {f.name}")
for name in ("asqa_eval_gtr_top100.json", "asqa_eval_gtr_top100_reranked_oracle.json"):
    d = json.load(open(root / "ALCE-data" / name))
    ex = d[0]
    print(f"\n## {name}: {len(d)} items; keys: {list(ex.keys())}")
    for k, v in ex.items():
        if isinstance(v, list):
            print(f"  {k}: list[{len(v)}] of {type(v[0]).__name__ if v else '-'}; first: {json.dumps(v[0])[:330] if v else ''}")
        else:
            print(f"  {k}: {json.dumps(v)[:300]}")
    ids = [x.get("sample_id") or x.get("id") for x in d]
    print("  unique ids:", len(set(ids)), "| docs per item:", sorted({len(x["docs"]) for x in d})[:5],
          "| qa_pairs per item (min/mean/max):", min(len(x["qa_pairs"]) for x in d),
          round(sum(len(x["qa_pairs"]) for x in d) / len(d), 2), max(len(x["qa_pairs"]) for x in d))
