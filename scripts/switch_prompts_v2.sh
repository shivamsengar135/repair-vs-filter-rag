#!/usr/bin/env bash
# Switch long-form scripts' default prompt set to v2 (deviation D3); check no direct verifier calls remain; run tests.
cd "$(dirname "$0")/.."
echo "direct verifier calls left in run_longform.py (should be none):"
grep -nE '\bmc\.|\bnli\.' scripts/run_longform.py || echo "  none"
for f in run_longform run_cove run_rarr run_flare; do
  sed -i 's#default="configs/prompts/v1.yaml"#default="configs/prompts/v2.yaml"#' "scripts/$f.py"
  echo "$f: $(grep -c 'prompts/v2.yaml' "scripts/$f.py") occurrence(s) of v2"
done
bash scripts/run_tests.sh > /tmp/t.log 2>&1; tail -n 2 /tmp/t.log
