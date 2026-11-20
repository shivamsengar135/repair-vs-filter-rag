#!/usr/bin/env bash
# Read-only preflight for the confirmatory run: GPU, docker, disk, data files, timings of the earlier bio test runs.
cd "$(dirname "$0")/.."
source scripts/wsl_env.sh
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || echo "no GPU"
docker ps --format '{{.Names}} {{.Image}} {{.Status}}' || echo "docker not reachable"
df -h ~ | tail -n 1
ls -la ~/halluc_data/factscore/data/labeled/ 2>/dev/null | head
wc -l ~/halluc_data/factscore/data/labeled/prompt_entities.txt
echo "--- earlier bio test timings (phase5 log)"
grep -nE "long-form pipeline bio test|verifier scores .* bio test|judge results/test/bio/longform|GEN_STAGE_END|SCORE_STAGE_END|FINISH_STAGE_END|JUDGE_STAGE_END|serve " logs/phase5_all.log | head -n 40
