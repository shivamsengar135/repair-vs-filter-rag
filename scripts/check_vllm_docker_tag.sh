#!/usr/bin/env bash
# Read-only: does the official vLLM image have a tag matching our pinned version, and how big is it?
TAG="${1:-v0.33.0}"
curl -sS -m 30 "https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags/$TAG" | python3 -c '
import sys, json
d = json.load(sys.stdin)
if "name" not in d:
    print("TAG NOT FOUND:", d); sys.exit(1)
print("tag:", d["name"], "| last_updated:", d.get("last_updated"))
for im in d.get("images", []):
    print("  %s/%s compressed %.2f GB digest %s" % (im.get("os"), im.get("architecture"), (im.get("size") or 0) / 1e9, (im.get("digest") or "")[:19]))
'
