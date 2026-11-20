#!/usr/bin/env bash
# List files (with sizes) of a Hub model repo at main, and what is in the local snapshot. Usage: list_repo_files.sh REPO
R="$1"
curl -s -m 40 "https://huggingface.co/api/models/$R/tree/main" | python3 -c '
import sys, json
for x in json.load(sys.stdin):
    print("   remote %10.1f MB  %s" % (x.get("size", 0) / 1e6, x["path"]))
'
ls -la ~/.cache/huggingface/hub/models--$(echo "$R" | sed 's#/#--#g')/snapshots/*/ 2>/dev/null | awk '{print "   local", $NF}'
ls ~/.cache/huggingface/hub/models--$(echo "$R" | sed 's#/#--#g')/refs -R 2>/dev/null
