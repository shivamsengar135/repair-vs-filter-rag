"""Print abstracts from refs/raw/meta.json for given keys. Usage: meta_abs.py MAXCHARS key [key ...]"""
import json
import sys

recs = {r["key"]: r for r in json.load(open("refs/raw/meta.json"))["records"]}
mx = int(sys.argv[1])
for k in sys.argv[2:]:
    a = recs[k]["arxiv"]
    print(f"## {k} | {a['arxiv_id_versioned']} | journal_ref={a['journal_ref']} | comment={(a['comment'] or '')[:90]}\n{a['abstract'][:mx]}\n")
