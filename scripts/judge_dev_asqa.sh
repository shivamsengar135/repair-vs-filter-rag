#!/usr/bin/env bash
# Swap the GPU to the judge (Phi-4 FP8), judge every ASQA dev v2 run, then analyse. Do NOT edit while running.
# Prerequisite: all Qwen-dependent dev work finished (this stops the Qwen server).
set -uo pipefail
cd "$(dirname "$0")/.."
JUDGE="RedHatAI/phi-4-FP8-dynamic"; JREV="8f62f4b85ccc1c7fda9f178976c1162f37e43651"; JTAG="phi-4-FP8-dynamic"
docker rm -f vllm > /dev/null 2>&1
# Phi-4 FP8 leaves 1.46 GiB of KV cache at util 0.90 (< one 8192-token sequence, observed 2026-11-16); judge
# prompts are ~1k tokens, so serve with a 4096-token context and 92% of GPU memory (0.94 exceeds what is free
# next to the Windows desktop's ~0.5 GB).
GPU_UTIL=0.92 MAX_MODEL_LEN=4096 bash scripts/serve_vllm_docker.sh "$JUDGE" "$JREV" 2>&1 | tail -n 1
LOG=$(cat logs/vllm_serve_latest.txt); docker logs vllm > "$LOG" 2>&1
grep -aE 'Model loading took|GPU KV cache size' "$LOG" | cut -c1-200
for pat in longform cove-closed_book cove-rag_verify rarr flare; do
  run=$(for d in results/dev/asqa/${pat}__*/; do
          if grep -q '"prompts": "v2"' "$d/summary.json" 2>/dev/null && ! grep -q '"limit"' "$d/summary.json"; then echo "$d"; fi
        done | xargs -r ls -td | head -n 1)
  echo "##### judge $pat -> $run"
  [ -n "$run" ] && bash scripts/py.sh scripts/run_judge.py --run "$run" --judge-model "$JUDGE" 2>&1 | tail -n 1 | cut -c1-500
done
lf=$(for d in results/dev/asqa/longform__*/; do grep -q '"prompts": "v2"' "$d/summary.json" 2>/dev/null && ! grep -q '"limit"' "$d/summary.json" && echo "$d"; done | xargs -r ls -td | head -n 1)
echo "##### analysis $lf"
bash scripts/py.sh scripts/analyze_longform.py --run "$lf" --judge-tag "$JTAG" 2>&1 | tail -n 30
bash scripts/py.sh scripts/compare_longform.py --dataset asqa --split dev --judge-tag "$JTAG" 2>&1 | tail -n 25
bash scripts/py.sh scripts/make_labelling_sample.py --run "$lf" --judge-tag "$JTAG" 2>&1 | tail -n 5
echo JUDGE_ASQA_END
