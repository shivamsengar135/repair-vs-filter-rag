#!/usr/bin/env bash
# Run the unit tests in the pinned env. Usage: bash scripts/run_tests.sh [pytest args]
set -uo pipefail
source "$(dirname "$0")/wsl_env.sh"
export PYTHONPATH="$PROJECT_ROOT/src"
# -p no:cacheprovider: do not write .pytest_cache onto the slow Windows mount
"$UV_PROJECT_ENVIRONMENT/bin/python" -m pytest -p no:cacheprovider -q tests "$@" 2>&1 | tee logs/pytest_latest.log
rc="${PIPESTATUS[0]}"
# marker line so callers that pipe the output (and lose the exit code) can still check the result
[ "$rc" = 0 ] && echo "TESTS_OK" || echo "TESTS_FAILED rc=$rc"
exit "$rc"
