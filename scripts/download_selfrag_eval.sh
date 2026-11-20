#!/usr/bin/env bash
# Self-RAG task evaluation data (Google Drive file id from the official README, AkariAsai/self-rag, read 2026-11-13).
# Used only for the Phase 4 sanity reproduction (EXPERIMENT_PLAN.md §11). Detached tmux, resumable, slow retries.
set -euo pipefail
source "$(dirname "$0")/wsl_env.sh"
DST="$DATA_ROOT/selfrag_eval"; mkdir -p "$DST"
LOG="$PROJECT_ROOT/logs/dl_selfrag_eval.log"
tmux kill-session -t selfrag 2>/dev/null || true
tmux new-session -d -s selfrag "
  cd $DST
  for attempt in \$(seq 1 30); do
    echo \"=== attempt \$attempt \$(date -u +%FT%TZ)\" >> $LOG
    if $HOME/.local/bin/uv run -q --no-project --with gdown gdown -c 1TLKhWjez63H4uBtgCxyoyJsZi-IMgnDb -O eval_data.zip >> $LOG 2>&1; then
      echo \"DL_OK \$(stat -c %s eval_data.zip) \$(sha256sum eval_data.zip | cut -d' ' -f1)\" >> $LOG; break
    fi
    sleep 1200
  done
  echo SELFRAG_DL_END >> $LOG"
echo "started tmux 'selfrag'; log $LOG"
