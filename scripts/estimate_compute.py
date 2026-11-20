"""Tier-1/Tier-2 compute estimate from measured dev costs (EXPERIMENT_PLAN.md §12).

Per method and dataset: generator prompt/completion tokens per question (from each dev run's summary, cache hits
included, i.e. the method's true cost), verifier pairs per question, judge calls per question. Scaled to test-set sizes.
Throughput model (measured on this GPU, see RESULTS_LEDGER): generator prefill ~5,500 prompt tok/s and decode
~650 output tok/s at concurrency 32 (PopQA dev RAG: 420k prompt + 2.5k output tokens in 76.5 s; Phase-0 bench);
judge measured from its own runs; verifiers on GPU measured from the SelfCheck NLI run. All of these are rough.
Usage: python scripts/estimate_compute.py
"""
import glob
import json
from pathlib import Path

TEST_N = {"popqa": 1000, "nq": 1000, "triviaqa": 1000, "hotpotqa": 1000, "2wiki": 1000, "bio": 350, "asqa": 748}
PREFILL, DECODE = 5500.0, 650.0                       # generator tokens / s at concurrency 32 (Qwen2.5-7B, measured)
VERIFIER_PAIRS_PER_S_GPU = 60.0                       # conservative; NLI on GPU (SelfCheck: 14,070 pairs, see log)


def latest(pattern):
    c = []
    for d in glob.glob(pattern):
        s = Path(d) / "summary.json"
        if s.exists():
            j = json.loads(s.read_text())
            if "limit" not in j["config"] and j["config"].get("prompts", "v1") in ("v2", ["v1", "shortform_v1"]) or j["config"].get("pipeline") == "shortform-v1":
                c.append((s.stat().st_mtime, j))
    return max(c, key=lambda x: x[0])[1] if c else None


def tokens(summary):
    by = summary["cost"].get("llm_by_stage") or summary["cost"].get("llm") or {}
    return sum(v["prompt_tokens"] for v in by.values()), sum(v["completion_tokens"] for v in by.values())


def main():
    rows, total_gen_h, total_ver_h = [], 0.0, 0.0
    specs = [(ds, "shortform") for ds in ("popqa", "nq", "triviaqa", "hotpotqa", "2wiki")] + \
            [(ds, p) for ds in ("asqa", "bio") for p in ("longform", "cove-closed_book", "cove-rag_verify", "rarr", "flare")]
    for ds, pat in specs:
        s = latest(f"results/dev/{ds}/{pat}__*")
        if not s:
            continue
        p, c = tokens(s)
        n = s["n"]
        pairs = sum((s["cost"].get("verifier_pairs") or {}).values()) if isinstance(s["cost"].get("verifier_pairs"), dict) else 0
        gen_s = (p / PREFILL + c / DECODE) / n * TEST_N[ds]
        ver_s = pairs / n * TEST_N[ds] / VERIFIER_PAIRS_PER_S_GPU
        total_gen_h += gen_s / 3600
        total_ver_h += ver_s / 3600
        rows.append((ds, pat, n, p / n, c / n, pairs / n, gen_s / 3600, ver_s / 3600))
    print(f"{'dataset':9s} {'method':18s} {'prompt tok/q':>12s} {'out tok/q':>10s} {'pairs/q':>8s} {'gen h (test)':>12s} {'verif h':>8s}")
    for r in rows:
        print(f"{r[0]:9s} {r[1]:18s} {r[3]:12.0f} {r[4]:10.0f} {r[5]:8.1f} {r[6]:12.2f} {r[7]:8.2f}")
    # standard RAG / no-retrieval short-form runs are a subset of the short-form pipeline's first stage: not added
    print(f"\nTier 1 (G1, all methods above, test sets): generator {total_gen_h:.1f} h + verifiers {total_ver_h:.1f} h")
    print("Tier 2 (G2, G3: no-retrieval, standard RAG, CF, C, A = the short-form and long-form pipeline runs only):")
    t2 = sum(r[6] + r[7] for r in rows if r[1] in ("shortform", "longform"))
    print(f"  ≈ {t2:.1f} h per extra generator → {2 * t2:.1f} h for G2 + G3")
    print("Not included: judge (measure from logs/phase4_judge_*.log), SelfCheckGPT samples (5 generations/question),")
    print("Self-RAG, 3 sampling seeds, ablations, tuning (≤ 20 dev configurations per method on 500-question dev sets).")


if __name__ == "__main__":
    main()
