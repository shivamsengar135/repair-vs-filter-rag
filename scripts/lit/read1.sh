#!/usr/bin/env bash
# Reading pass 1: locate the main results tables of the core mitigation papers.
cd /mnt/c/Users/DIAT/Documents/shivam/trial_1_Q1
G="python3 scripts/lit/grep_text.py"
for k in selfrag crag flare cove rarr adaptive_rag dragin; do
  $G $k '^(Table|Figure) [0-9]+:' 0 0 9 170
done
