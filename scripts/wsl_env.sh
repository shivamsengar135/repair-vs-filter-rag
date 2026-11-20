#!/usr/bin/env bash
# Source this in every WSL shell before running project code:  source scripts/wsl_env.sh
# The repo lives on the Windows drive (/mnt/c, slow 9p I/O); everything heavy
# (venv, uv cache, HF cache, indexes) lives on the WSL ext4 disk.
export PATH="$HOME/.local/bin:$PATH"
export PROJECT_ROOT="/mnt/c/Users/DIAT/Documents/shivam/trial_1_Q1"
export DATA_ROOT="$HOME/halluc_data"                 # large data: corpora, indexes, embeddings
export UV_PROJECT_ENVIRONMENT="$HOME/venvs/halluc"   # venv on ext4, not on /mnt/c
export UV_LINK_MODE=copy
export HF_HOME="$HOME/.cache/huggingface"
# No C compiler in WSL: torch's native Triton op overrides (e.g. bmm_outer_product) cannot build their launcher and
# crash on CUDA. Kill switch -> standard torch/cuBLAS kernels (2026-11-17, Phase 5 score stage).
export TORCH_DISABLE_NATIVE_JIT=1
mkdir -p "$DATA_ROOT" "$PROJECT_ROOT/logs" "$PROJECT_ROOT/results"
cd "$PROJECT_ROOT" || return 1
