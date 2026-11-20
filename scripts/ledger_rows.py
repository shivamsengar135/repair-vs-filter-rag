"""Print markdown ledger rows from the latest non-limited dev runs (reads summary.json / analysis.json; no model calls).
Usage: python scripts/ledger_rows.py"""
import glob
import json
from pathlib import Path


def runs(pattern):
    out = []
    for d in sorted(glob.glob(pattern)):
        s = Path(d) / "summary.json"
        if s.exists():
            j = json.loads(s.read_text())
            if "limit" not in j["config"]:
                out.append((s.stat().st_mtime, Path(d), j))
    return sorted(out)


print("## short-form (latest run per dataset)")
print("| Dataset | n | RAG acc | A acc (no abstention) | repair fixed/broke | CF AURC | C AURC | A AURC | cov/acc@0.90 CF | C | A | Output dir |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for ds in ("popqa", "nq", "triviaqa", "hotpotqa", "2wiki"):
    r = runs(f"results/dev/{ds}/shortform__*")
    if not r:
        continue
    _, d, s = r[-1]
    an = json.loads((d / "analysis.json").read_text()) if (d / "analysis.json").exists() else None
    m = an["methods"] if an else {}

    def cell(k):
        t = m[k]["targets"]["0.9"]
        return f"{t['coverage']['mean']:.2f}/{t['accuracy_answered']['mean']:.2f}"
    eff = s["repair_effect_on_match"]
    print(f"| {ds} | {s['n']} | {s['rag_match']:.3f} | {s['A_match_no_abstention']:.3f} | {eff['fixed']}/{eff['broken']} | "
          + (f"{m['CF (score s)']['aurc']:.3f} | {m['C (two-signal)']['aurc']:.3f} | {m['A (repair)']['aurc']:.3f} | "
             f"{cell('CF (score s)')} | {cell('C (two-signal)')} | {cell('A (repair)')} | " if m else "- | - | - | - | - | - | ")
          + f"`{d}/` |")
print("\n## long-form v2 (latest run per method)")
for ds in ("asqa", "bio"):
    for pat in ("longform", "cove-closed_book", "cove-rag_verify", "rarr", "flare"):
        r = [x for x in runs(f"results/dev/{ds}/{pat}__*") if x[2]["config"].get("prompts") == "v2"]
        if r:
            _, d, s = r[-1]
            keys = [k for k in ("claims_per_response_mean", "final_claims_per_response", "responses_with_zero_claims", "empty_final_answers",
                                "actions", "repair", "verdicts", "edits_applied", "retrieval_rate_after_first") if k in s]
            print(f"| {pat} | {ds} | {s['n']} | " + "; ".join(f"{k}={s[k]}" for k in keys) + f" | `{d}/` |")
