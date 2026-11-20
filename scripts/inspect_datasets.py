"""Phase 4: download the QA datasets and print their splits, sizes, features and one example (schema check)."""
import json

from datasets import load_dataset

SPECS = [
    ("akariasai/PopQA", None),
    ("google-research-datasets/nq_open", "nq_open"),
    ("mandarjoshi/trivia_qa", "rc.nocontext"),
    ("hotpotqa/hotpot_qa", "fullwiki"),
    ("xanhho/2WikiMultihopQA", None),
]
for name, cfg in SPECS:
    print(f"\n##### {name} [{cfg}]")
    try:
        ds = load_dataset(name, cfg) if cfg else load_dataset(name)
    except Exception as e:  # noqa: BLE001
        print("  LOAD FAILED:", type(e).__name__, str(e)[:300])
        continue
    for split, d in ds.items():
        print(f"  split {split}: {len(d)} rows; features: {list(d.features)}")
    first_split = list(ds)[0]
    ex = ds[first_split][0]
    print("  example:", json.dumps({k: (v if len(str(v)) < 260 else str(v)[:260] + '…') for k, v in ex.items()}, default=str)[:1400])
