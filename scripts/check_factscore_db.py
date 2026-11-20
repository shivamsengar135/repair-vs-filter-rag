"""Integrity + schema check of FActScore's knowledge base (read-only). Schema per factscore/retrieval.py:
documents(title PRIMARY KEY, text) with passages joined by SPECIAL_SEPARATOR."""
import json
import sqlite3
from pathlib import Path

from hallu.store import read_jsonl

SEP = "####SPECIAL####SEPARATOR####"
db = Path.home() / "halluc_data/factscore/enwiki-20230401.db"
con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
print("tables:", con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
print("schema:", con.execute("SELECT sql FROM sqlite_master WHERE name='documents'").fetchone())
print("quick_check:", con.execute("PRAGMA quick_check").fetchone())
n = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
print("documents:", n)
topics = [r["meta"]["topic"] for f in ("bio_dev", "bio_test") for r in read_jsonl(f"data/splits/{f}.jsonl")]
labeled = [l.strip() for l in (Path.home() / "halluc_data/factscore/data/labeled/prompt_entities.txt").read_text().splitlines() if l.strip()]
for name, ts in (("bio dev+test", topics), ("labeled", labeled)):
    miss = [t for t in ts if con.execute("SELECT 1 FROM documents WHERE title = ?", (t,)).fetchone() is None]
    print(f"{name}: {len(ts)} topics, {len(miss)} missing from KB {miss[:5]}")
row = con.execute("SELECT text FROM documents WHERE title = ?", (topics[0],)).fetchone()
psgs = row[0].split(SEP)
print(json.dumps({"topic": topics[0], "n_passages": len(psgs), "first": psgs[0][:300]}))
