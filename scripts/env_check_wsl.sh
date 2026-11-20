#!/usr/bin/env bash
# Phase 0: read-only survey of the WSL2 Ubuntu environment. Makes no changes.
echo "--- os"; lsb_release -ds 2>/dev/null; uname -r; whoami
echo "--- gpu"
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv 2>&1
echo "--- mem/cpu"; free -g | head -2; nproc
echo "--- disk"; df -h / /mnt/c 2>&1
echo "--- tools"
for t in git python3 uv conda tmux nvcc gcc curl docker; do
  printf "%s -> " "$t"; command -v "$t" || echo NOT_FOUND
done
python3 --version 2>&1; git --version 2>&1
echo "--- sudo (non-interactive)"; sudo -n true 2>&1 && echo "passwordless sudo OK" || echo "sudo needs a password"
echo "--- home"; ls -la ~ | head -40
