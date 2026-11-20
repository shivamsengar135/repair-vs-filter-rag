#!/usr/bin/env bash
# grep a cached HF model's README and list its files. Usage: bash scripts/grep_model_card.sh org/name 'regex'
d=$(ls -d "$HOME/.cache/huggingface/hub/models--${1/\//--}/snapshots/"*/ | head -n 1)
grep -n -i -E "$2" "$d/README.md" | cut -c1-220 | head -n 30
for f in "$d"/1_Pooling/config.json "$d"/config_sentence_transformers.json; do [ -f "$f" ] && { echo "--- $f"; cat "$f"; echo; }; done
ls "$d"
