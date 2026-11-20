#!/usr/bin/env bash
# One-shot status of the machine and all background jobs (use at the start of every session).
cd "$(dirname "$0")/.."
echo "--- $(date -u +%FT%TZ) uptime:$(uptime | sed 's/.*up/ up/' | cut -c1-40)"
echo "--- tmux"; tmux ls 2>&1
echo "--- docker"; docker ps --format '{{.Names}} {{.Image}} {{.Status}}' 2>&1
echo "--- gpu"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "--- git"; git log --oneline | head -1; git status --short | head -5
echo "--- long-form dev log"; tail -n 12 logs/phase4_longform_dev.log 2>/dev/null | cut -c1-1000
echo "--- model downloads"; grep -aE 'DOWNLOAD_|DOWNLOADS_END|=== ' logs/download_models.log 2>/dev/null | tail -n 5 | cut -c1-200
du -sh ~/.cache/huggingface/hub/models--* 2>/dev/null | sed 's#/home/diat/.cache/huggingface/hub/##'
echo "--- factscore KB"; tr '\r' '\n' < logs/dl_factscore.log 2>/dev/null | grep -aE 'DL_OK|FACTSCORE_DL_END|attempt|Too many' | tail -n 3 | cut -c1-160
ls -la ~/halluc_data/factscore | grep -i enwiki | cut -c25-100
echo "--- disk"; df -h ~ /mnt/c | tail -n 2
