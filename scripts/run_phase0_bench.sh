#!/usr/bin/env bash
# Phase 0: run the throughput benchmark against the already-running vLLM server.
set -euo pipefail
source "$(dirname "$0")/wsl_env.sh"
MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
OUT="results/phase0/vllm_bench_qwen2.5-7b-instruct_bf16.json"
{
  echo "git commit: $(git rev-parse --short HEAD)  date: $(date -u +%FT%TZ)"
  "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/bench_vllm.py --model "$MODEL" --levels 1,8,32,64 --max-tokens 256 --out "$OUT"
  echo "finish reasons: $(grep -o '"finish_reason": "[a-z_]*"' "$OUT" | sort | uniq -c | tr '\n' ' ')"
  echo "BENCH_OK"
} 2>&1 | tee logs/phase0_bench.log
