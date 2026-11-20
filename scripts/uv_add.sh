#!/usr/bin/env bash
# Add dependencies to the pinned project env (never ad-hoc pip). Usage: bash scripts/uv_add.sh pkg [pkg ...]
set -euo pipefail
source "$(dirname "$0")/wsl_env.sh"
uv add "$@" 2>&1 | tail -n 25
echo "--- now installed:"
for p in "$@"; do uv pip list --python "$UV_PROJECT_ENVIRONMENT/bin/python" 2>/dev/null | grep -iE "^${p%%[<>=]*} " || true; done
echo UV_ADD_OK
