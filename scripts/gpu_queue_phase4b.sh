#!/usr/bin/env bash
# Falcon3 fit test with the full revision (deviation D4), then Self-RAG on NQ and TriviaQA dev (re-run after the
# event-loop fix). Do NOT edit while running.
set -uo pipefail
cd "$(dirname "$0")/.."
echo "##### fit test Falcon3-7B-Instruct"
bash scripts/fit_test.sh tiiuae/Falcon3-7B-Instruct 1e57a0ecd176c7c139f289c60a74e57f887c3dfb 2>&1 | tail -n 6 | cut -c1-500
docker rm -f vllm > /dev/null 2>&1
DTYPE=float16 MAX_MODEL_LEN=4096 bash scripts/serve_vllm_docker.sh selfrag/selfrag_llama2_7b 190261383b0779ff66d2f95a73c7ad267d94b820 2>&1 | tail -n 1
for d in nq triviaqa; do
  echo "##### selfrag baseline $d dev"
  bash scripts/py.sh scripts/selfrag_repro.py --dataset "$d" --split dev 2>&1 | grep -E '^\{|Error|Traceback' | cut -c1-600
done
docker rm -f vllm > /dev/null 2>&1
echo GPU_QUEUE_B_END
