#!/usr/bin/env bash
# Confirmatory C-vs-CF runs (docs/PREREG_C_CONFIRMATORY.md): bio_confirm with G1-G3, same stages and settings as the
# Phase 5 test runs (generate+decompose per generator, GPU verifier scores, full run, judge), then the preregistered
# analysis ONCE. Resumable (every step cached). Run in tmux; log: logs/confirm_runs.log. Do NOT edit while running.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/wsl_env.sh
G1="Qwen/Qwen2.5-7B-Instruct";           R1="a09a35458c702b33eeacc393d103063234e8bc28"
G2="mistralai/Mistral-7B-Instruct-v0.3"; R2="c170c708c41dac9275d15a8fff4eca08d52bab71"
G3="tiiuae/Falcon3-7B-Instruct";         R3="1e57a0ecd176c7c139f289c60a74e57f887c3dfb"
JUDGE="RedHatAI/phi-4-FP8-dynamic";      JREV="8f62f4b85ccc1c7fda9f178976c1162f37e43651"
serve() { docker rm -f vllm > /dev/null 2>&1; echo "===== $(date -u +%FT%TZ) serve $1"; bash scripts/serve_vllm_docker.sh "$1" "$2" 2>&1 | tail -n 1; }
lf() { bash scripts/py.sh scripts/run_longform.py --dataset bio --split confirm "$@" 2>&1 | grep -E 'STOP_AFTER|^\{|^wrote|Error|Traceback|LLM request failed' | cut -c1-600; }

echo "===== CONFIRM_START $(date -u +%FT%TZ) commit $(git rev-parse --short HEAD)"
serve "$G1" "$R1"; echo "##### generate+decompose G1"; lf --model "$G1" --stop-after decompose
serve "$G2" "$R2"; echo "##### generate+decompose G2"; lf --model "$G2" --stop-after decompose
serve "$G3" "$R3"; echo "##### generate+decompose G3"; lf --model "$G3" --stop-after decompose
docker rm -f vllm > /dev/null 2>&1
for M in "$G1" "$G2" "$G3"; do
  echo "##### $(date -u +%FT%TZ) verifier scores $M (GPU)"
  HF_HUB_OFFLINE=1 lf --model "$M" --stop-after score --verifier-device cuda
done
serve "$G1" "$R1"; echo "##### full run G1"; lf --model "$G1" --verifier-device cpu
serve "$G2" "$R2"; echo "##### full run G2"; lf --model "$G2" --verifier-device cpu
serve "$G3" "$R3"; echo "##### full run G3"; lf --model "$G3" --verifier-device cpu
docker rm -f vllm > /dev/null 2>&1
GPU_UTIL=0.92 MAX_MODEL_LEN=4096 bash scripts/serve_vllm_docker.sh "$JUDGE" "$JREV" 2>&1 | tail -n 1
for run in results/confirm/bio/longform__*/; do
  echo "##### $(date -u +%FT%TZ) judge $run"
  bash scripts/py.sh scripts/run_judge.py --run "$run" --judge-model "$JUDGE" 2>&1 | tail -n 1 | cut -c1-400
done
docker rm -f vllm > /dev/null 2>&1
echo "##### $(date -u +%FT%TZ) preregistered analysis (once)"
bash scripts/py.sh scripts/confirmatory_c.py 2>&1 | tail -n 5
echo "===== CONFIRM_END $(date -u +%FT%TZ)"
