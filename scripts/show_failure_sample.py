"""Compact view of labelling/failure_sample.csv for annotation. Usage: python scripts/show_failure_sample.py START END"""
import csv
import sys

rows = list(csv.DictReader(open("labelling/failure_sample.csv", encoding="utf-8")))
lo, hi = int(sys.argv[1]), int(sys.argv[2])
for r in rows[lo:hi]:
    print(f"[{r['sample_id']}] {r['error_type']} | action={r['action']} origin={r['origin']} s={r['s_original']} score={r['filter_score']} "
          f"c={r['c_closed_book']} con={r['con']} judge_p={r['judge_p_true']}")
    print(f"  Q: {r['question'][:110]}")
    print(f"  CLAIM: {r['claim_final'][:220]}" + (f"  (orig: {r['claim_original'][:150]})" if r['claim_original'] != r['claim_final'] else ""))
    print(f"  BEST: {r['best_passage'][:380]}")
