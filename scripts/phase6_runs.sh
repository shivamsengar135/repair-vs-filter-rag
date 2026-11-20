#!/usr/bin/env bash
# Phase 6 ablation / sensitivity runs on TEST (EXPERIMENT_PLAN.md §8, D6). Do NOT edit while running.
# Resumable: every step is cached; re-running repeats only what is missing. Log: logs/phase6_runs.log
# Order minimises GPU swaps: fit tests -> G1 generation stage -> 3B and 14B stages -> GPU verifier scores (no
# server) -> G1 full runs -> size full runs -> judge -> dense index build -> dense runs (G1) -> judge again.
set -uo pipefail
cd "$(dirname "$0")/.."
G1="Qwen/Qwen2.5-7B-Instruct";       R1="a09a35458c702b33eeacc393d103063234e8bc28"
S3="Qwen/Qwen2.5-3B-Instruct";       R3="aa8e72537993ba99e69dfaafa59ed015b17504d1"
S14="Qwen/Qwen2.5-14B-Instruct-AWQ"; R14="539535859b135b0244c91f3e59816150c8056698"
step() { echo "##### $(date -u +%FT%TZ) $*"; }
serve() { docker rm -f vllm > /dev/null 2>&1; echo "===== $(date -u +%FT%TZ) serve $1 ${DTYPE:-bfloat16}"; bash scripts/serve_vllm_docker.sh "$1" "$2" 2>&1 | tail -n 1; }
LF() { bash scripts/py.sh scripts/run_longform.py --split test "$@" 2>&1 | grep -E '^\{|STOP_AFTER|^wrote|Error|Traceback' | cut -c1-400; }
# G1 long-form ablation configs (beyond the default run that already exists)
G1_CFGS=("--triage one_axis" "--tau-s 0.3" "--tau-s 0.4" "--tau-s 0.6" "--tau-s 0.7" "--k 3" "--k 10"
         "--decompose sentences" "--verifier nli" "--verifier alignscore" "--verifier generator")

step "fit tests"
for spec in "$S3 $R3" "$S14 $R14"; do
  set -- $spec
  [ -f "results/phase4/fit/$(echo "$1" | tr '/' '_').json" ] && { echo "fit test exists for $1"; continue; }
  if [ "$1" = "$S14" ]; then DTYPE=float16 bash scripts/fit_test.sh "$1" "$2"; else bash scripts/fit_test.sh "$1" "$2"; fi
  cat "results/phase4/fit/$(echo "$1" | tr '/' '_').json" | head -c 400; echo
done

serve "$G1" "$R1"
for d in bio asqa; do
  for c in "--k 3" "--k 10" "--decompose sentences"; do step "G1 $d $c: generate + decompose"; LF --dataset "$d" --model "$G1" $c --stop-after decompose; done
done

for spec in "$S3 $R3 bfloat16" "$S14 $R14 float16"; do
  set -- $spec
  DTYPE="$3" serve "$1" "$2"
  for d in bio asqa; do step "size $1 $d: generate + decompose"; LF --dataset "$d" --model "$1" --stop-after decompose; done
  for d in popqa nq triviaqa hotpotqa 2wiki; do
    step "size $1 short-form $d"
    if ls results/test/"$d"/shortform__"${1##*/}"__*/summary.json > /dev/null 2>&1; then echo "SKIP finished"; continue; fi
    bash scripts/py.sh scripts/run_qa.py --dataset "$d" --split test --method closed_book --model "$1" 2>&1 | grep -E '^\{"n"|Error' | cut -c1-300
    bash scripts/py.sh scripts/run_qa.py --dataset "$d" --split test --method rag --model "$1" 2>&1 | grep -E '^\{"n"|Error' | cut -c1-300
    bash scripts/py.sh scripts/run_shortform.py --dataset "$d" --split test --model "$1" --verifier-device cpu 2>&1 | grep -E '^\{|^wrote|Error|Traceback' | cut -c1-400
  done
done

docker rm -f vllm > /dev/null 2>&1
for d in bio asqa; do
  for c in "--k 3" "--k 10" "--decompose sentences" "--verifier nli" "--verifier alignscore"; do
    step "GPU scores G1 $d $c"; HF_HUB_OFFLINE=1 LF --dataset "$d" --model "$G1" $c --stop-after score --verifier-device cuda
  done
  for m in "$S3" "$S14"; do step "GPU scores $m $d"; HF_HUB_OFFLINE=1 LF --dataset "$d" --model "$m" --stop-after score --verifier-device cuda; done
done

serve "$G1" "$R1"
for d in bio asqa; do
  for c in "${G1_CFGS[@]}"; do step "G1 $d $c: full run"; LF --dataset "$d" --model "$G1" $c --verifier-device cpu; done
done
for spec in "$S3 $R3 bfloat16" "$S14 $R14 float16"; do
  set -- $spec
  DTYPE="$3" serve "$1" "$2"
  for d in bio asqa; do step "size $1 $d: full run"; LF --dataset "$d" --model "$1" --verifier-device cpu; done
done

step "judge all long-form test runs (cached labels are reused)"
bash scripts/phase5_judge_stage.sh

step "dense index build (resumes from finished shards)"
docker rm -f vllm > /dev/null 2>&1
HF_HUB_OFFLINE=1 bash scripts/py.sh scripts/build_dense_index.py 2>&1 | grep -E 'shard|DENSE_INDEX_DONE|Error|Traceback' | cut -c1-300

serve "$G1" "$R1"
for d in bio asqa; do step "G1 $d dense: generate + decompose"; LF --dataset "$d" --model "$G1" --retriever dense --stop-after decompose; done
docker rm -f vllm > /dev/null 2>&1
for d in bio asqa; do step "GPU scores G1 $d dense"; HF_HUB_OFFLINE=1 LF --dataset "$d" --model "$G1" --retriever dense --stop-after score --verifier-device cuda; done
serve "$G1" "$R1"
for d in bio asqa; do step "G1 $d dense: full run"; LF --dataset "$d" --model "$G1" --retriever dense --verifier-device cpu; done
step "judge dense runs"
bash scripts/phase5_judge_stage.sh
docker rm -f vllm > /dev/null 2>&1
echo "===== PHASE6_RUNS_END $(date -u +%FT%TZ)"
