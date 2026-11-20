#!/usr/bin/env bash
# Smoke checks before Phase 5 (no server needed; uses cached dev data).
cd "$(dirname "$0")/.."
source scripts/wsl_env.sh
for f in scripts/run_qa.py scripts/run_longform.py scripts/run_selfcheck.py scripts/analyze_longform.py scripts/selfrag_repro.py; do
  "$UV_PROJECT_ENVIRONMENT/bin/python" -m py_compile "$f" && echo "compiles: $f"
done
for s in phase5_gen_stage phase5_finish_stage phase5_score_stage; do bash -n "scripts/$s.sh" && echo "bash syntax ok: $s"; done
HF_HUB_OFFLINE=1 bash scripts/py.sh scripts/run_longform.py --dataset bio --split dev --model Qwen/Qwen2.5-7B-Instruct --limit 3 --stop-after decompose 2>&1 | grep -E 'STOP|Error|Trace'
HF_HUB_OFFLINE=1 bash scripts/py.sh scripts/run_longform.py --dataset bio --split dev --model Qwen/Qwen2.5-7B-Instruct --limit 3 --stop-after score --verifier-device cpu 2>&1 | grep -E 'STOP|Error|Trace'
bash scripts/run_tests.sh > /tmp/t.log 2>&1; tail -n 1 /tmp/t.log
