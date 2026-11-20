#!/usr/bin/env bash
# Phase 0: create/refresh the pinned Python env from pyproject.toml + uv.lock.
set -euo pipefail
source "$(dirname "$0")/wsl_env.sh"
uv --version
uv sync --python 3.12
echo "--- installed key packages"
uv pip list --python "$UV_PROJECT_ENVIRONMENT/bin/python" | grep -iE "^(vllm|torch|transformers|tokenizers|huggingface.hub|openai|numpy|xformers|pytest) " || true
echo "SETUP_ENV_OK"
