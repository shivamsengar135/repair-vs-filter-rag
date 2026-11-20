#!/usr/bin/env bash
# Download FActScore data from the Google Drive file IDs given in the official repo
# (shmsw25/FActScore, factscore/download_data.py, read 2026-11-13). Runs detached in tmux; resumable (gdown -c).
#   data.zip           155exEdKs7R21gZF4G-x54-XN3qswBcPo   prompt entities, human annotations
#   enwiki-20230401.db 1Qu4JHWjpUKhGPaAW5UHhS5RJ545CVy4I   Wikipedia knowledge source (SQLite)
set -euo pipefail
source "$(dirname "$0")/wsl_env.sh"
DST="$DATA_ROOT/factscore"; mkdir -p "$DST"
LOG="$PROJECT_ROOT/logs/dl_factscore.log"
tmux kill-session -t factscore 2>/dev/null || true
tmux new-session -d -s factscore "
  cd $DST
  for spec in '155exEdKs7R21gZF4G-x54-XN3qswBcPo data.zip' '1Qu4JHWjpUKhGPaAW5UHhS5RJ545CVy4I enwiki-20230401.db'; do
    set -- \$spec
    for attempt in \$(seq 1 60); do
      echo \"=== \$2 attempt \$attempt \$(date -u +%FT%TZ)\" >> $LOG
      if $HOME/.local/bin/uv run -q --no-project --with gdown gdown -c \$1 -O \$2 >> $LOG 2>&1; then
        echo \"DL_OK \$2 \$(stat -c %s \$2) \$(sha256sum \$2 | cut -d' ' -f1)\" >> $LOG; break
      fi
      sleep 1800
    done
  done
  echo FACTSCORE_DL_END >> $LOG"
echo "started tmux 'factscore'; log $LOG"
