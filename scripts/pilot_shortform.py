"""Pilot quantity R on a short-form dev run (EXPERIMENT_PLAN.md §1).

Preregistered definition: among answers the filter-only method (CF) would drop at target 0.90 (threshold calibrated
on the full dev run), the fraction for which A's repair step OUTPUTS a correct answer (match).
Two stricter variants, reported next to it and labelled as such:
  R_net  : the dropped answer was wrong and the repair output is correct (repair actually fixes something)
  R_kept : the repair output is correct AND its support score exceeds the CF threshold (it would survive the filter)
Usage: python scripts/pilot_shortform.py --run results/dev/hotpotqa/shortform__...
"""
import argparse
import json
from pathlib import Path

from hallu import conformal as C
from hallu import metrics as M
from hallu.store import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    rows = read_jsonl(Path(a.run) / "items.jsonl")
    q = C.calibrate_from_responses([[r["s"]] for r in rows], [[bool(r["rag_match"])] for r in rows], 0.10)
    dropped = [r for r in rows if not (r["s"] > q)]
    att = [r for r in dropped if "repair_answer" in r]
    ok = [r for r in att if M.match_accuracy(r["repair_answer"], r["answers"]) == 1.0]
    n = len(dropped)
    out = {"run": a.run, "q_cf_full_dev_at_0.90": q, "answers_dropped_by_CF": n,
           "of_which_correct_originally": sum(int(r["rag_match"]) for r in dropped), "repair_attempted": len(att),
           "repair_output_correct": len(ok), "R": len(ok) / n if n else float("nan"),
           "R_net": sum(1 for r in ok if r["rag_match"] == 0.0) / n if n else float("nan"),
           "R_kept": sum(1 for r in ok if r["repair_s"] > q) / n if n else float("nan")}
    (Path(a.run) / "pilot.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
