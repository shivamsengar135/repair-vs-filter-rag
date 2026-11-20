#!/usr/bin/env bash
# Regression check after adding ablation flags: the default long-form config must keep its original hash
# (no new output directory may appear). Fully cached, so no vLLM server is needed.
cd "$(dirname "$0")/.."
before=$(ls -d results/test/bio/longform__Qwen2.5-7B-Instruct__* | sort)
HF_HUB_OFFLINE=1 bash scripts/py.sh scripts/run_longform.py --dataset bio --split test --model Qwen/Qwen2.5-7B-Instruct --stop-after decompose 2>&1 | grep -E 'STOP_AFTER|Error|Traceback'
after=$(ls -d results/test/bio/longform__Qwen2.5-7B-Instruct__* | sort)
echo "before: $before"; echo "after:  $after"
[ "$before" = "$after" ] && echo HASH_UNCHANGED || echo HASH_CHANGED
