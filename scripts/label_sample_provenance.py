"""Provenance of the human-validation sample (run, judge tag, seed, source x judge-label counts). Run only after
labelling is finished: it reads the key file."""
import json
from collections import Counter

d = json.load(open("labelling/.key_DO_NOT_OPEN.json"))
print("run:", d["run"], "| judge:", d["judge_tag"], "| seed:", d["seed"])
print(Counter((r["source"], r["judge_label"]) for r in d["rows"]))
