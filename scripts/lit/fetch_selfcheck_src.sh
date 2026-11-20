#!/usr/bin/env bash
# Read-only: print the official SelfCheckGPT-NLI implementation so our baseline replicates its scoring.
set -euo pipefail
B=https://raw.githubusercontent.com/potsawee/selfcheckgpt/main/selfcheckgpt
curl -s -m 40 "$B/modeling_selfcheck.py" -o /tmp/sc.py
curl -s -m 40 "$B/utils.py" -o /tmp/sc_utils.py
s=$(grep -n 'class SelfCheckNLI' /tmp/sc.py | cut -d: -f1)
e=$(awk -v s="$s" 'NR>s && /^class /{print NR; exit}' /tmp/sc.py)
sed -n "${s},$((e-1))p" /tmp/sc.py | grep -vE '^\s*$' | cut -c1-170
echo "--- utils"; grep -nE 'nli_model|NLI' /tmp/sc_utils.py | cut -c1-170
