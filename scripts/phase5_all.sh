#!/usr/bin/env bash
# Phase 5 master script: all test runs in an order that minimises GPU model swaps. Do NOT edit while running.
# Resumable: every step is cached; re-running repeats only what is missing. Log: logs/phase5_all.log
set -uo pipefail
cd "$(dirname "$0")/.."
G1="Qwen/Qwen2.5-7B-Instruct";        R1="a09a35458c702b33eeacc393d103063234e8bc28"
G2="mistralai/Mistral-7B-Instruct-v0.3"; R2="c170c708c41dac9275d15a8fff4eca08d52bab71"
G3="tiiuae/Falcon3-7B-Instruct";      R3="1e57a0ecd176c7c139f289c60a74e57f887c3dfb"
serve() { docker rm -f vllm > /dev/null 2>&1; echo "===== $(date -u +%FT%TZ) serve $1"; bash scripts/serve_vllm_docker.sh "$1" "$2" 2>&1 | tail -n 1; }

serve "$G1" "$R1"; bash scripts/phase5_gen_stage.sh "$G1" 1
serve "$G2" "$R2"; bash scripts/phase5_gen_stage.sh "$G2" 2
serve "$G3" "$R3"; bash scripts/phase5_gen_stage.sh "$G3" 2
bash scripts/phase5_score_stage.sh "$G1" "$G2" "$G3"
serve "$G1" "$R1"; bash scripts/phase5_finish_stage.sh "$G1" 1
serve "$G2" "$R2"; bash scripts/phase5_finish_stage.sh "$G2" 2
serve "$G3" "$R3"; bash scripts/phase5_finish_stage.sh "$G3" 2
docker rm -f vllm > /dev/null 2>&1
echo "===== $(date -u +%FT%TZ) SelfCheckGPT scores (GPU), 3 seed sets"
for d in asqa bio; do
  run=$(ls -td results/test/$d/longform__Qwen2.5-7B-Instruct__*/ | head -n 1)
  for k in 0 1 2; do
    HF_HUB_OFFLINE=1 bash scripts/py.sh scripts/run_selfcheck.py --run "$run" --stage score --device cuda --seed-set "$k" 2>&1 | grep -E '^\{|Error|Traceback' | cut -c1-300
  done
done
bash scripts/phase5_judge_stage.sh
echo "===== $(date -u +%FT%TZ) Self-RAG on the 5 short-form test sets"
docker rm -f vllm > /dev/null 2>&1
DTYPE=float16 MAX_MODEL_LEN=4096 bash scripts/serve_vllm_docker.sh selfrag/selfrag_llama2_7b 190261383b0779ff66d2f95a73c7ad267d94b820 2>&1 | tail -n 1
for d in popqa nq triviaqa hotpotqa 2wiki; do
  echo "##### selfrag $d test"
  bash scripts/py.sh scripts/selfrag_repro.py --dataset "$d" --split test 2>&1 | grep -E '^\{|Error|Traceback' | cut -c1-400
done
docker rm -f vllm > /dev/null 2>&1
echo "===== PHASE5_ALL_END $(date -u +%FT%TZ)"
