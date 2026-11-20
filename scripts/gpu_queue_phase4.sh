#!/usr/bin/env bash
# Remaining Phase 4 GPU work, in order: Self-RAG reproduction + Self-RAG short-form dev baseline, then the G3 fit
# test(s) per deviation D4 (Olmo-3 first, Falcon3 only if Olmo-3 fails). Do NOT edit while running.
set -uo pipefail
cd "$(dirname "$0")/.."
bash scripts/selfrag_all.sh
echo "##### fit test Olmo-3-7B-Instruct"
bash scripts/fit_test.sh allenai/Olmo-3-7B-Instruct 6e5971d9eba42665f5bd5a0fcf047f299ce1dccc 2>&1 | tail -n 6 | cut -c1-500
if ! grep -q '"fits_rule_ge_16000_kv_tokens": true' results/phase4/fit/allenai_Olmo-3-7B-Instruct.json; then
  while ! grep -qE 'DOWNLOAD_(OK|FAILED) tiiuae/Falcon3-7B-Instruct' logs/download_models.log; do sleep 60; done
  echo "##### fit test Falcon3-7B-Instruct"
  bash scripts/fit_test.sh tiiuae/Falcon3-7B-Instruct 1e57a0ecd176 2>&1 | tail -n 6 | cut -c1-500
fi
echo GPU_QUEUE_END
