#!/usr/bin/env bash
# Block up to $2 seconds (default 590) watching log $1; return early on Traceback / Error / *_END markers.
# Prints the last step headers and any new problem lines. Usage: bash scripts/wait_log.sh logs/phase5_all.log [secs]
LOG="$1"; SECS="${2:-590}"
start=$(wc -l < "$LOG")
for ((i = 0; i < SECS; i += 10)); do
  if tail -n +"$((start + 1))" "$LOG" | grep -qE 'Traceback|Error|PHASE5_ALL_END'; then break; fi
  sleep 10
done
tail -n +"$((start + 1))" "$LOG" | grep -E '^#####|^wrote|Traceback|Error|_END' | cut -c1-300
echo "--- $(date -u +%FT%TZ) last step:"; grep '^#####' "$LOG" | tail -1
