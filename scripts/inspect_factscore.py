"""Extract the FActScore data archive and print its structure (entity lists, human-annotation schema)."""
import json
import zipfile
from collections import Counter
from pathlib import Path

root = Path.home() / "halluc_data" / "factscore"
z = zipfile.ZipFile(root / "data.zip")
for i in z.infolist():
    if not i.is_dir():
        print(f"{i.file_size:>10}  {i.filename}")
z.extractall(root)  # trusted location, plain data files (jsonl/txt)

for sub in ("labeled", "unlabeled"):
    p = root / "data" / sub / "prompt_entities.txt"
    if p.exists():
        ents = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"\n{sub}/prompt_entities.txt: {len(ents)} entities, {len(set(ents))} unique; first: {ents[:3]}")

for f in sorted((root / "data" / "labeled").glob("*.jsonl")):
    rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"\n{f.name}: {len(rows)} rows; keys: {list(rows[0].keys())}")
    labels, n_ann = Counter(), 0
    for r in rows:
        for sent in (r.get("annotations") or []):
            for a in (sent.get("human-atomic-facts") or []):
                labels[a.get("label")] += 1
        n_ann += bool(r.get("annotations"))
    print(f"  rows with annotations: {n_ann}; atomic-fact labels: {dict(labels)}")
    ex = next(r for r in rows if r.get("annotations"))
    s0 = ex["annotations"][0]
    print("  example:", json.dumps({"topic": ex.get("topic"), "input": ex.get("input"), "output": (ex.get("output") or "")[:120],
                                    "annotation[0] keys": list(s0.keys()), "facts[0:2]": (s0.get("human-atomic-facts") or [])[:2]})[:700])
