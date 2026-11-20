#!/usr/bin/env bash
# Fit test (EXPERIMENT_PLAN.md §2): does MODEL load in vLLM on the 20 GB GPU with >= 16,000 tokens of KV cache at
# max-model-len 8192? Starts the server, records load memory / KV cache from the server log, runs a short throughput
# benchmark, writes results/phase4/fit/<model-tag>.json, then stops the server.
# Usage: bash scripts/fit_test.sh MODEL REVISION [extra vllm args]
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="$1"; REV="$2"; shift 2
TAG=$(echo "$MODEL" | tr '/' '_')
OUT="results/phase4/fit/$TAG.json"; mkdir -p results/phase4/fit
bash scripts/serve_vllm_docker.sh "$MODEL" "$REV" "$@" > "logs/fit_$TAG.serve.txt" 2>&1
STATUS=$(tail -n 40 "logs/fit_$TAG.serve.txt" | grep -oE 'VLLM_READY|VLLM_DIED|VLLM_TIMEOUT' | tail -n 1)
LOG=$(cat logs/vllm_serve_latest.txt)
docker logs vllm > "$LOG" 2>&1 || true
LOADGB=$(grep -aoE 'Model loading took [0-9.]+ GiB' "$LOG" | grep -oE '[0-9.]+' | tail -n 1)
KV=$(grep -aoE 'GPU KV cache size: [0-9,]+ tokens' "$LOG" | grep -oE '[0-9,]+' | tr -d , | tail -n 1)
GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
ERR=$(grep -aE 'Error|error:' "$LOG" | grep -vE 'Retrying|stderr' | tail -n 3 | cut -c1-300 | tr '\n' ' ' | tr '"' "'")
BENCH=null
if [ "$STATUS" = "VLLM_READY" ]; then
  source scripts/wsl_env.sh
  "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/bench_vllm.py --model "$MODEL" --levels 1,32 --max-tokens 128 --out "results/phase4/fit/${TAG}_bench.json" > "logs/fit_$TAG.bench.txt" 2>&1 \
    && BENCH="\"results/phase4/fit/${TAG}_bench.json\""
fi
FITS=false
[ "$STATUS" = "VLLM_READY" ] && [ -n "$KV" ] && [ "$KV" -ge 16000 ] && FITS=true
cat > "$OUT" <<EOF
{"model": "$MODEL", "revision": "$REV", "extra_args": "$*", "status": "$STATUS", "model_load_GiB": ${LOADGB:-null},
 "kv_cache_tokens": ${KV:-null}, "gpu_used_MiB_with_server": ${GPU:-null}, "fits_rule_ge_16000_kv_tokens": $FITS,
 "bench": $BENCH, "server_log": "$LOG", "errors": "$ERR", "date_utc": "$(date -u +%FT%TZ)", "git_commit": "$(git rev-parse --short HEAD)"}
EOF
docker rm -f vllm > /dev/null 2>&1 || true
cat "$OUT"
[ -f "logs/fit_$TAG.bench.txt" ] && grep -E '^concurrency' "logs/fit_$TAG.bench.txt"
