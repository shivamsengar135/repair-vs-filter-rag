#!/usr/bin/env bash
# Robust, resumable model downloads in a detached tmux session, one model after another.
# Xet transfer failed on this network (2026-11-12), so plain HTTPS is forced; it resumes from .incomplete files.
# Usage: bash scripts/download_model.sh "MODEL REVISION [hf-download args]" ["MODEL REVISION ..."] ...
#   e.g. bash scripts/download_model.sh "mistralai/Mistral-7B-Instruct-v0.3 main --exclude consolidated.safetensors"
# Progress: logs/download_models.log ; each model ends with DOWNLOAD_OK <model> <resolved sha> or DOWNLOAD_FAILED.
set -euo pipefail
source "$(dirname "$0")/wsl_env.sh"
LOG="logs/download_models.log"
SPECS="$DATA_ROOT/.download_specs.txt"
printf '%s\n' "$@" > "$SPECS"
tmux kill-session -t dl 2>/dev/null || true
tmux new-session -d -s dl "
  while IFS= read -r spec; do
    set -- \$spec; model=\$1; rev=\$2; shift 2
    ok=0
    for attempt in \$(seq 1 30); do
      echo \"=== \$model @ \$rev attempt \$attempt \$(date -u +%FT%TZ)\" >> $LOG
      if HF_HOME=$HF_HOME HF_HUB_DISABLE_XET=1 $UV_PROJECT_ENVIRONMENT/bin/hf download \$model --revision \$rev \"\$@\" >> $LOG 2>&1; then ok=1; break; fi
      sleep 15
    done
    dir=$HF_HOME/hub/models--\$(echo \$model | sed 's#/#--#')
    if [ \$ok = 1 ]; then echo \"DOWNLOAD_OK \$model \$(ls \$dir/snapshots | tr '\n' ' ') \$(du -sh \$dir | cut -f1)\" >> $LOG; else echo \"DOWNLOAD_FAILED \$model\" >> $LOG; fi
  done < $SPECS
  echo DOWNLOADS_END >> $LOG"
echo "started tmux session 'dl'; log: $LOG"
