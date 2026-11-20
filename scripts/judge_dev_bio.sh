#!/usr/bin/env bash
# Serve the judge, validate it on FActScore's human labels, judge every biography dev v2 run, analyse.
# Do NOT edit while running. Prerequisite: no other job needs the GPU.
set -uo pipefail
cd "$(dirname "$0")/.."
JUDGE="RedHatAI/phi-4-FP8-dynamic"; JREV="8f62f4b85ccc1c7fda9f178976c1162f37e43651"; JTAG="phi-4-FP8-dynamic"
docker rm -f vllm > /dev/null 2>&1
GPU_UTIL=0.92 MAX_MODEL_LEN=4096 bash scripts/serve_vllm_docker.sh "$JUDGE" "$JREV" 2>&1 | tail -n 1
echo "##### judge validation on FActScore human labels"
bash scripts/py.sh scripts/validate_judge_factscore.py --judge-model "$JUDGE" 2>&1 | grep -E '^(InstructGPT|ChatGPT|PerplexityAI|pooled|wrote)|Error|Traceback' | cut -c1-900
for pat in longform cove-closed_book cove-rag_verify rarr flare; do
  run=$(for d in results/dev/bio/${pat}__*/; do grep -q '"prompts": "v2"' "$d/summary.json" 2>/dev/null && ! grep -q '"limit"' "$d/summary.json" && echo "$d"; done | xargs -r ls -td | head -n 1)
  echo "##### judge $pat -> $run"
  [ -n "$run" ] && bash scripts/py.sh scripts/run_judge.py --run "$run" --judge-model "$JUDGE" 2>&1 | tail -n 1 | cut -c1-500
done
lf=$(for d in results/dev/bio/longform__*/; do grep -q '"prompts": "v2"' "$d/summary.json" 2>/dev/null && ! grep -q '"limit"' "$d/summary.json" && echo "$d"; done | xargs -r ls -td | head -n 1)
echo "##### analysis $lf"
bash scripts/py.sh scripts/analyze_longform.py --run "$lf" --judge-tag "$JTAG" 2>&1 | grep -E '^A vs|^pilot' | cut -c1-700
bash scripts/py.sh scripts/compare_longform.py --dataset bio --split dev --judge-tag "$JTAG" 2>&1 | tail -n 25
echo "##### ASQA re-analysis (adds SelfCheckGPT)"
alf=$(for d in results/dev/asqa/longform__*/; do grep -q '"prompts": "v2"' "$d/summary.json" 2>/dev/null && ! grep -q '"limit"' "$d/summary.json" && echo "$d"; done | xargs -r ls -td | head -n 1)
bash scripts/py.sh scripts/analyze_longform.py --run "$alf" --judge-tag "$JTAG" 2>&1 | grep -E '^A vs|^pilot' | cut -c1-700
bash scripts/py.sh scripts/compare_longform.py --dataset asqa --split dev --judge-tag "$JTAG" 2>&1 | tail -n 25
echo JUDGE_BIO_END
