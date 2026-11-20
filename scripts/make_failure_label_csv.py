"""Write labelling/failure_categories.csv: the 96 failure cases of Section 6 for an independent human categorisation.

Blind design (2026-11-20): the file does NOT contain the assistant's pre-annotation
(labelling/failure_preannotation_assistant.json), so the human categorisation is independent; agreement with the
pre-annotation is computed afterwards (scripts/score_failure_categories.py) and the HUMAN categories are reported.
Columns the annotator fills (category, notes) come right after the claim; UTF-8 BOM for Excel.
"""
import csv
from pathlib import Path

src = list(csv.DictReader(Path("labelling/failure_sample.csv").open(encoding="utf-8")))
assert len(src) == 96
WHAT = {"wrong_but_kept": "KEPT by the filter, but the judge says NOT supported",
        "correct_but_dropped": "REMOVED by the filter, but the judge says SUPPORTED"}
out = Path("labelling/failure_categories.csv")
with out.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f, lineterminator="\r\n")
    w.writerow(["case", "dataset", "what_went_wrong", "topic_or_question", "claim", "category", "notes",
                "score_vs_threshold", "repair", "best_retrieved_passage", "all_retrieved_titles", "answer_excerpt"])
    for r in src:
        s, q = float(r["filter_score"]), float(r["threshold_q_hat"])
        sv = f"score {s:.3f} {'>' if s > q else '<='} threshold {q:.3f} -> {'kept' if s > q else 'removed'}"
        rep = ("re-retrieved: the score comes from a SECOND search whose passages are not shown here"
               if r["origin"] == "reretrieved" else "none (original claim and score)")
        w.writerow([r["sample_id"].rsplit("-", 1)[-1], {"bio": "biography", "asqa": "ASQA question"}[r["dataset"]],
                    WHAT[r["error_type"]], r["question"], r["claim_final"], "", "", sv, rep, r["best_passage"],
                    r["retrieved_titles"], r["answer_excerpt"]])
back = list(csv.DictReader(out.open(encoding="utf-8-sig")))
assert len(back) == 96 and not any("JE" in x["category"] for x in back)
print("wrote", out, len(back), "rows")
