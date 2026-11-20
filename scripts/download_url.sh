#!/usr/bin/env bash
# Resumable download of one URL into $DATA_ROOT/<subdir>/ in a detached tmux session (network here is flaky).
# Usage: bash scripts/download_url.sh SESSION SUBDIR URL [FILENAME]
# Progress: cat logs/dl_<SESSION>.log     Done marker: line "DL_OK <bytes> <sha256>"
set -euo pipefail
source "$(dirname "$0")/wsl_env.sh"
S="$1"; SUB="$2"; URL="$3"; FN="${4:-$(basename "$URL")}"
DST="$DATA_ROOT/$SUB"; mkdir -p "$DST"
LOG="$PROJECT_ROOT/logs/dl_$S.log"
tmux kill-session -t "$S" 2>/dev/null || true
tmux new-session -d -s "$S" "
  echo \"=== start \$(date -u +%FT%TZ) $URL -> $DST/$FN\" >> $LOG
  for attempt in \$(seq 1 200); do
    if curl -L -sS --fail -C - --retry 3 --connect-timeout 30 --speed-limit 1000 --speed-time 120 -o '$DST/$FN.part' '$URL' >> $LOG 2>&1; then
      mv '$DST/$FN.part' '$DST/$FN'
      echo \"DL_OK \$(stat -c %s '$DST/$FN') \$(sha256sum '$DST/$FN' | cut -d' ' -f1) \$(date -u +%FT%TZ)\" >> $LOG; exit 0
    fi
    echo \"attempt \$attempt failed \$(date -u +%FT%TZ) size=\$(stat -c %s '$DST/$FN.part' 2>/dev/null)\" >> $LOG
    sleep 20
  done
  echo DL_FAILED >> $LOG"
echo "started tmux '$S' -> $DST/$FN ; log $LOG"
