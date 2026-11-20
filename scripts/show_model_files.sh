#!/usr/bin/env bash
# Show a cached HF model's files, config.json and README head. Usage: bash scripts/show_model_files.sh org/name
d=$(ls -d "$HOME/.cache/huggingface/hub/models--${1/\//--}/snapshots/"*/ | head -n 1)
ls -la "$d"
echo "--- config.json"; head -c 3000 "$d/config.json"; echo
echo "--- README.md"; head -c "${2:-3000}" "$d/README.md"; echo
for f in "$d"/*.py; do [ -f "$f" ] && { echo "--- $f"; head -n 60 "$f"; }; done
