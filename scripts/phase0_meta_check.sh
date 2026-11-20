#!/usr/bin/env bash
# Phase 0: read-only metadata checks (git identity, caches, HF model card, PyPI vLLM version).
echo "--- line endings"; file "$(dirname "$0")"/*.sh 2>/dev/null
echo "--- git identity"; git config --global user.name; git config --global user.email
echo "--- caches"; ls ~/.cache ~/.local/bin 2>&1; ls ~/.cache/huggingface 2>&1
echo "--- HF model metadata"
curl -sS -m 30 "https://huggingface.co/api/models/${1:-Qwen/Qwen2.5-7B-Instruct}?blobs=true" | python3 -c '
import sys, json
d = json.load(sys.stdin)
print({k: d.get(k) for k in ["id", "gated", "private", "sha", "lastModified"]})
print("license:", (d.get("cardData") or {}).get("license"))
tot = 0
for s in d.get("siblings", []):
    tot += s.get("size") or 0
    print("  ", s["rfilename"], s.get("size"))
print("total GB: %.2f" % (tot / 1e9))
'
echo "--- PyPI vllm"
curl -sS -m 30 https://pypi.org/pypi/vllm/json | python3 -c '
import sys, json
i = json.load(sys.stdin)["info"]
print("vllm latest:", i["version"], "| requires_python:", i["requires_python"])
print([r for r in i["requires_dist"] if r.split("[")[0].split("=")[0].split(">")[0].split("<")[0].strip() in ("torch", "transformers")])
'
