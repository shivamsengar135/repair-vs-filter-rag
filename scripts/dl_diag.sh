#!/usr/bin/env bash
# Diagnose slow HF download: is there a documented switch to bypass Xet, and how fast is plain HTTPS?
SP=~/venvs/halluc/lib/python3.12/site-packages
echo "--- xet-related env switches in huggingface_hub"
grep -rhoE "HF_HUB_DISABLE_XET|HF_XET_[A-Z_]+|HF_HUB_ENABLE_HF_TRANSFER" "$SP/huggingface_hub" | sort | uniq -c
pip_list=$(ls "$SP" | grep -iE "^(hf_xet|hf_transfer)" ); echo "installed: $pip_list"
echo "--- plain HTTPS speed, 20s sample of shard 4 (discarded)"
curl -L -s -o /dev/null -m 20 -w "http=%{http_code} bytes=%{size_download} speed=%{speed_download} B/s\n" \
  "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/a09a35458c702b33eeacc393d103063234e8bc28/model-00004-of-00004.safetensors"
echo "--- generic speed reference, 15s (discarded)"
curl -L -s -o /dev/null -m 15 -w "http=%{http_code} bytes=%{size_download} speed=%{speed_download} B/s\n" \
  "https://speed.cloudflare.com/__down?bytes=2000000000"
