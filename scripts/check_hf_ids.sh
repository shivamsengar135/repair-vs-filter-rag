#!/usr/bin/env bash
# Verify that Hugging Face model ids exist (primary source: HF API). Usage: bash scripts/check_hf_ids.sh id [id ...]
cd "$(dirname "$0")/.."
for id in "$@"; do
  curl -s --max-time 60 "https://huggingface.co/api/models/$id?blobs=true" | python3 scripts/hf_model_info.py "$id"
done
