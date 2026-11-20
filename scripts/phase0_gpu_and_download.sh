#!/usr/bin/env bash
# Phase 0: torch GPU check, record vLLM CLI flags, download the benchmark model.
set -euo pipefail
source "$(dirname "$0")/wsl_env.sh"
MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
REV="${2:-a09a35458c702b33eeacc393d103063234e8bc28}"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
"$PY" scripts/check_torch_gpu.py 2>&1 | tee logs/phase0_torch_gpu.log
"$UV_PROJECT_ENVIRONMENT/bin/vllm" serve --help=all > logs/phase0_vllm_serve_help.txt 2>&1 || \
  "$UV_PROJECT_ENVIRONMENT/bin/vllm" serve --help > logs/phase0_vllm_serve_help.txt 2>&1 || true
echo "vllm help lines: $(wc -l < logs/phase0_vllm_serve_help.txt)"
echo "--- downloading $MODEL @ $REV"
"$UV_PROJECT_ENVIRONMENT/bin/hf" download "$MODEL" --revision "$REV" 2>&1 | tail -n 5
du -sh "$HF_HOME/hub" | cat
echo "DOWNLOAD_OK"
