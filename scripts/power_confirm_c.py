"""Power of the preregistered C-vs-CF test at n = 183 biographies, simulated on the EXISTING test runs (exploratory
data, already reported). For each generator: draw 183 of the 350 test answers without replacement (400 draws, seed
2026), run the confirmatory procedure (partition 0, paired bootstrap of supported claims/answer, precision lower
bound; 2,000 resamples to keep it fast) and count how often (a) the difference is significant and positive at the
Holm level an adjusted test would face (alpha/3 = 0.0167, the strictest Holm step), (b) the precision guard holds,
(c) both. Writes results/test/power_confirm_c.json. Used to fix the design in docs/PREREG_C_CONFIRMATORY.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from exploratory_tests import longform_methods, one  # noqa: E402
from hallu import analysis as A  # noqa: E402
from hallu import metrics as M  # noqa: E402

RUNS = {"Qwen2.5-7B-Instruct": "5aa2ad1482e5", "Mistral-7B-Instruct-v0.3": "3f20b417147d", "Falcon3-7B-Instruct": "0b9ff4861d3b"}
N, DRAWS, B = 183, 400, 2000


def once(ma, mb, rng_seed):
    ra, rb = A.evaluate_once(ma, 0.10, seed=0), A.evaluate_once(mb, 0.10, seed=0)
    boot = M.paired_bootstrap(ra["supported_per_response"], rb["supported_per_response"], B, seed=0)
    ev = ra["eval_index"]
    ka = np.array([sum(1 for s, _ in ma[i] if s > ra["q_hat"]) for i in ev], float)
    kb = np.array([sum(1 for s, _ in mb[i] if s > rb["q_hat"]) for i in ev], float)
    sa, sb = ra["supported_per_response"], rb["supported_per_response"]
    idx = np.random.default_rng(0).integers(0, len(ev), size=(B, len(ev)))
    with np.errstate(invalid="ignore", divide="ignore"):
        d = sa[idx].sum(1) / ka[idx].sum(1) - sb[idx].sum(1) / kb[idx].sum(1)
    d = d[np.isfinite(d)]
    lb = float(np.quantile(d, 0.05)) if len(d) else float("nan")
    return boot["diff"], boot["p_value"], lb


def main() -> None:
    out = {}
    rng = np.random.default_rng(2026)
    for g, rid in RUNS.items():
        m = longform_methods(one(f"results/test/bio/longform__{g}__{rid}/"))
        n_all = len(m["C"])
        sig = guard = both = 0
        diffs = []
        for _ in range(DRAWS):
            pick = rng.choice(n_all, size=N, replace=False)
            d, p, lb = once([m["C"][i] for i in pick], [m["CF"][i] for i in pick], 0)
            diffs.append(d)
            s_ok, g_ok = (p <= 0.05 / 3 and d > 0), (lb > -0.02)
            sig += s_ok; guard += g_ok; both += s_ok and g_ok
        out[g] = {"draws": DRAWS, "n": N, "power_sig_positive_at_alpha_over_3": sig / DRAWS, "p_guard_holds": guard / DRAWS,
                  "p_both": both / DRAWS, "mean_diff": float(np.mean(diffs))}
        print(g, out[g], flush=True)
    Path("results/test/power_confirm_c.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
