"""Confirmatory split for H3-H5 (docs/PREREG_C_CONFIRMATORY.md): the 183 FActScore entities with human labels.

They were never used for any method run (build_longform_splits.py excluded them; they served only judge validation 1,
which judged FActScore's released outputs of other systems). Same row format and prompt as bio_dev / bio_test.
Writes data/splits/bio_confirm.jsonl and its MANIFEST entry; refuses to overwrite different content.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from hallu.store import write_jsonl

ROOT = Path.home() / "halluc_data"
OUT = Path("data/splits")

ents = [l.strip() for l in (ROOT / "factscore/data/labeled/prompt_entities.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
assert len(ents) == len(set(ents)) == 183, len(ents)
used = set()
for s in ("dev", "test"):
    used |= {json.loads(l)["meta"]["topic"] for l in (OUT / f"bio_{s}.jsonl").read_text(encoding="utf-8").splitlines()}
overlap = sorted(set(ents) & used)
if overlap:
    sys.exit(f"overlap with bio dev/test: {overlap}")
rows = sorted(({"id": "bio-" + hashlib.sha1(e.encode()).hexdigest()[:12], "dataset": "bio",
                "question": f"Question: Tell me a bio of {e}.", "answers": [], "meta": {"topic": e}} for e in ents),
              key=lambda r: r["id"])
dst = OUT / "bio_confirm.jsonl"
blob = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode()
sha = hashlib.sha256(blob).hexdigest()
if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() != sha:
    sys.exit(f"REFUSING to overwrite {dst}: content would change (frozen split)")
assert write_jsonl(dst, rows) == sha
man = json.loads((OUT / "MANIFEST.json").read_text())
man["bio_confirm"] = {"n": len(rows), "sha256": sha, "source": "factscore/data/labeled/prompt_entities.txt (all 183)"}
(OUT / "MANIFEST.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
print(f"bio_confirm n={len(rows)} sha256={sha}  overlap with dev/test: 0  e.g. {rows[0]['question']!r}")
