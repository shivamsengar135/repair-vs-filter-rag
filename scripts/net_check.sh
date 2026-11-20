#!/usr/bin/env bash
# Read-only connectivity check for the hosts this project downloads from.
date -u
for u in https://registry-1.docker.io/v2/ https://auth.docker.io/token https://hub.docker.com https://huggingface.co https://pypi.org/simple/ https://github.com; do
  curl -s -o /dev/null -m 15 -w "%{http_code} %{time_total}s  $u\n" "$u" || echo "FAIL  $u"
done
echo "--- attempts so far"; grep -ac "=== attempt" "$(cat "$(dirname "$0")/../logs/docker_pull_latest.txt" | sed "s#^#$(dirname "$0")/../#")" 2>/dev/null
echo "--- error kinds"; grep -a -v "=== attempt" "$(dirname "$0")/../$(cat "$(dirname "$0")/../logs/docker_pull_latest.txt")" | cut -c1-90 | sort | uniq -c
