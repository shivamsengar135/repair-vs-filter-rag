#!/usr/bin/env bash
# Run a Python script in the pinned env with src/ on the path. Usage: bash scripts/py.sh script.py [args]
set -euo pipefail
source "$(dirname "$0")/wsl_env.sh"
export PYTHONPATH="$PROJECT_ROOT/src" HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
exec "$UV_PROJECT_ENVIRONMENT/bin/python" "$@"
