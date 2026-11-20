#!/usr/bin/env bash
# After PHASE6_RUNS_END: regenerate every derived table / figure that includes the new runs. Log: logs/phase6_finalize.log
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/py.sh scripts/ablation_table.py | grep -E 'dense|wrote'
bash scripts/py.sh scripts/figures.py --split test | tail -n 1
bash scripts/py.sh scripts/ledger_tables.py
bash scripts/build_paper.sh
grep -A3 -i 'dense' results/test/tables_ablations.md | head -n 8
