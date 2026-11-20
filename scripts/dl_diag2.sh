#!/usr/bin/env bash
# Where is the hf download process actually writing?
PID=$(pgrep -f "hf download" | head -1)
echo "pid=$PID"; date
grep -E "VmRSS|VmSwap" /proc/$PID/status
echo "--- io counters"; cat /proc/$PID/io | grep -E "rchar|wchar|write_bytes"
echo "--- open regular files"; ls -l /proc/$PID/fd 2>/dev/null | grep -vE "socket|pipe|anon_inode|/dev/" | awk '{print $NF}' | sort | uniq -c
echo "--- apparent vs disk size of incomplete blobs"
for f in ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/blobs/*; do
  printf "%s apparent=%s disk=%s\n" "$(basename "$f" | cut -c1-12)" "$(stat -c %s "$f")" "$(du -B1 "$f" | cut -f1)"
done
sleep 20
echo "--- io counters +20s"; cat /proc/$PID/io | grep -E "rchar|wchar|write_bytes"
