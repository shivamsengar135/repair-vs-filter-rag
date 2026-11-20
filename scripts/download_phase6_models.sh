#!/usr/bin/env bash
# All Phase 6 model downloads in one batch (user, 2026-11-17: download everything now, nothing in between).
# Revisions pinned to the SHAs verified via the HF API on 2026-11-17 (scripts/check_hf_ids.sh).
# 2026-11-17 fix: the first bge spec "--exclude onnx/* pytorch_model.bin" made hf treat pytorch_model.bin as the only
# file to download; the bge line below fetches the whole repo except ONNX. Already-complete models are skipped by hf.
cd "$(dirname "$0")/.."
bash scripts/download_model.sh \
  "BAAI/bge-small-en-v1.5 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a --exclude onnx/*" \
  "liuyanyi/AlignScore-large-hf cc13dc572fce8aa32778a437fb66a02d39a5b8e3" \
  "Qwen/Qwen2.5-3B-Instruct aa8e72537993ba99e69dfaafa59ed015b17504d1" \
  "Qwen/Qwen2.5-14B-Instruct-AWQ 539535859b135b0244c91f3e59816150c8056698"
