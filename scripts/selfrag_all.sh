#!/usr/bin/env bash
# Serve the official Self-RAG 7B checkpoint (float16, 4096 context: Llama-2) and run: the PopQA reproduction, then
# Self-RAG as a baseline on our five short-form dev sets with our BM25 passages. Do NOT edit while running.
set -uo pipefail
cd "$(dirname "$0")/.."
docker rm -f vllm > /dev/null 2>&1
DTYPE=float16 MAX_MODEL_LEN=4096 bash scripts/serve_vllm_docker.sh selfrag/selfrag_llama2_7b 190261383b0779ff66d2f95a73c7ad267d94b820 2>&1 | tail -n 1
L=$(cat logs/vllm_serve_latest.txt); docker logs vllm > "$L" 2>&1; grep -aE 'Model loading took|GPU KV cache size' "$L" | cut -c1-200
echo "##### reproduction (official PopQA long-tail + Contriever passages)"
bash scripts/py.sh scripts/selfrag_repro.py 2>&1 | grep -E '^\{|Error|Traceback' | cut -c1-900
for d in popqa nq triviaqa hotpotqa 2wiki; do
  echo "##### baseline $d dev"
  bash scripts/py.sh scripts/selfrag_repro.py --dataset "$d" --split dev 2>&1 | grep -E '^\{|Error|Traceback' | cut -c1-600
done
docker rm -f vllm > /dev/null 2>&1
echo SELFRAG_END
