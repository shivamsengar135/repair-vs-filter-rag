#!/usr/bin/env bash
# Phase 2 novelty check: targeted reads. For each paper, (a) does it repair/re-retrieve/revise or only filter/detect,
# (b) contributions, (c) limitations, (d) cost.
cd /mnt/c/Users/DIAT/Documents/shivam/trial_1_Q1
G="python3 scripts/lit/grep_text.py"
REPAIR='(re-retriev|re-generat|regenerat|revis|repair|rewrit|correct the|second retrieval|additional retrieval|targeted retrieval)'
for k in conf_robust conf_multihop conformal_rag conf_mohri; do
  $G $k "$REPAIR" 0 0 5 300
  $G $k '(vacuous|empty|retain|retention|informativ).*(%|percent)' 0 0 4 330
  $G $k '(^## .*Limitation|limitation of|future work|we leave)' 0 1 3 330
done
$G saber '(4-cell|four-cell|decision|abstain).*(PK|CK|parametric)' 0 0 6 360
$G saber "$REPAIR" 0 0 4 300
$G saber '(claim-level|long-form|atomic|short-form|Llama|Qwen|Mistral|fine-tun)' 0 0 6 260
$G saber '(^## .*Limitation|limitation)' 0 2 3 360
$G eaver '(budget|cost|calls per|search calls|fewer)' 0 0 7 300
$G eaver '(fine-tun|train|LoRA|backbone|Qwen|Llama)' 0 0 5 260
$G eaver "$REPAIR" 0 0 3 260
$G cove_ragplus '(dataset|benchmark|baseline|Llama|GPT|28%|human|annotat)' 0 0 8 300
$G arbgraph "$REPAIR" 0 0 3 280
$G arbgraph '(Llama|Qwen|GPT|cost|latency|token|Limitation|abstain)' 0 0 8 280
