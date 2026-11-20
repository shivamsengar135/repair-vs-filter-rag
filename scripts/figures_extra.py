"""Additional paper figures (vector PDF + PNG preview), drawn only from existing TEST result files — no new runs.

retention_other_generators: supported claims kept vs conformal target, Mistral-7B and Falcon3-7B (rows) x dataset (cols).
score_distributions: MiniCheck support s and the two-signal score C for judge-supported vs unsupported claims
    (Qwen2.5-7B, all test claims), with the calibrated threshold at target 0.90 (median and range over the same 100
    partitions used everywhere else). Writes the thresholds to results/<split>/figure_numbers.json (ledger-traced).
claim_auroc: AUROC with 95% CI (results/<split>/claim_auroc.json) per score and generator.
precision_vs_content: micro precision vs supported claims per answer for every long-form method with Qwen2.5-7B
    (results/<split>/<ds>/comparison__*.json): filters as lines over targets 0.80-0.95, revising methods as points.
Usage: python scripts/figures_extra.py [--split test]"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

import figures as F  # same palette, rcParams, style() and default-run selection as the main figures
from hallu import analysis
from hallu.runutil import git_commit
from hallu.store import read_jsonl, text_key

plt = F.plt
JT = "phi-4-FP8-dynamic"
DSETS = (("bio", "Biographies"), ("asqa", "ASQA"))
GEN_COLORS = {"Qwen2.5-7B-Instruct": ("#2a78d6", "o", "Qwen2.5-7B"), "Mistral-7B-Instruct-v0.3": ("#eb6834", "s", "Mistral-7B"),
              "Falcon3-7B-Instruct": ("#1baf7a", "^", "Falcon3-7B")}
NEUTRAL = "#7a7974"


def save(fig, out: Path) -> None:
    fig.savefig(out, format="pdf")
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)


def fig_retention_others(split: str, out: Path) -> None:
    gens = [("Mistral-7B-Instruct-v0.3", "Mistral-7B"), ("Falcon3-7B-Instruct", "Falcon3-7B")]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), constrained_layout=True)
    xs = [float(t) for t in F.TARGETS]
    for r, (gen, glab) in enumerate(gens):
        for c, (ds, title) in enumerate(DSETS):
            ax = axes[r][c]
            an = json.loads((F.latest_longform(split, ds, gen) / f"analysis__{JT}.json").read_text())
            ends = []
            for key, (color, marker, label) in F.SERIES.items():
                if key not in an["targets"][F.TARGETS[0]]:
                    continue
                ys = [an["targets"][t][key]["supported_claims_per_response"]["mean"] for t in F.TARGETS]
                ax.plot(xs, ys, color=color, marker=marker, markersize=4.5, linewidth=2, label=label,
                        markeredgecolor=F.SURFACE, markeredgewidth=0.8)
                ends.append((ys[-1], label.split(" (")[0]))
            ax.set_ylim(bottom=0)
            F.direct_labels(ax, xs[-1], ends)
            ref = an["unfiltered"]["standard RAG (all claims)"]["supported_claims_per_response"]["mean"]
            ax.set_title(f"{glab}, {title}  (no filter: {ref:.1f})", fontsize=8.5, color=F.INK, loc="left")
            ax.set_xticks(xs)
            ax.set_xlim(0.79, 0.985)
            if r == 1:
                ax.set_xlabel("Target P(all kept claims supported)")
            if c == 0:
                ax.set_ylabel("Supported claims kept per answer")
            F.style(ax)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside upper center", ncol=3, frameon=False, fontsize=7.5, labelcolor=F.INK)
    save(fig, out)


def claims_with_labels(run: Path) -> list[list[dict]]:
    labels = {r["id"]: r["label"] for r in read_jsonl(run / f"judge__{JT}.jsonl")}
    return [[{**c, "label": labels[text_key(r["id"], c["text"])]} for c in r["claims"]] for r in read_jsonl(run / "items.jsonl")]


def fig_score_distributions(split: str, out: Path, numbers: dict) -> None:
    gen = "Qwen2.5-7B-Instruct"
    # Scores pile up just below 1, so the x axis is the distance from the top, 1 - score, on a log scale (reversed:
    # higher scores to the right). Values below 1e-6 (incl. exactly 1.0) are put in the leftmost bin.
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), constrained_layout=True)
    floor = 1e-6
    bins = np.logspace(np.log10(floor), 0, 37)
    for c, (ds, title) in enumerate(DSETS):
        resp = claims_with_labels(F.latest_longform(split, ds, gen))
        for r, (key, slab) in enumerate((("s", "MiniCheck support $s$ (CF)"), ("score_c", "Two-signal score (C)"))):
            ax = axes[r][c]
            data = [[(cl[key], bool(cl["label"])) for cl in rr] for rr in resp]
            qs = np.array([analysis.evaluate_once(data, 0.10, seed)["q_hat"] for seed in range(100)])
            q_med = float(np.median(qs))
            sup = np.array([v for rr in data for v, lab in rr if lab])
            uns = np.array([v for rr in data for v, lab in rr if not lab])
            share_sup_above = float(np.mean(sup > q_med))
            numbers[f"{ds}/{gen}/{key}"] = {"q_hat_median": q_med, "q_hat_min": float(qs.min()), "q_hat_max": float(qs.max()),
                                            "n_supported": int(len(sup)), "n_unsupported": int(len(uns)),
                                            "share_supported_above_median_q": share_sup_above,
                                            "share_unsupported_above_median_q": float(np.mean(uns > q_med))}
            for vals, color, lab in ((sup, "#2a78d6", "judge: supported"), (uns, "#eb6834", "judge: not supported")):
                w = np.full(len(vals), 1.0 / len(vals))
                ax.hist(np.clip(1 - vals, floor * 1.0001, 1), bins=bins, weights=w, histtype="step", linewidth=1.6, color=color, label=lab)
            ax.set_xscale("log")
            ax.set_xlim(1, floor)  # reversed: high scores on the right
            lo, hi = max(1 - qs.max(), floor), max(1 - qs.min(), floor)
            ax.axvspan(lo, hi, color=F.GRID, alpha=0.8, linewidth=0)
            ax.axvline(max(1 - q_med, floor), color=F.INK, linewidth=1.0, linestyle="--")
            qtxt = f"{q_med:.3f}" if q_med < 0.999 else f"1 $-$ {1 - q_med:.0e}".replace("e-0", "e-")  # 0.99999.. would print as 1.000
            numbers[f"{ds}/{gen}/{key}"]["one_minus_q_hat_median"] = float(1 - q_med)
            ax.set_title(f"{title}: {slab}, threshold {qtxt}", fontsize=8.5, color=F.INK, loc="left")
            if r == 1:
                ax.set_xlabel("1 $-$ claim score (log scale; higher scores to the right)")
            if c == 0:
                ax.set_ylabel("Share of claims in class")
            F.style(ax)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside upper center", ncol=2, frameon=False, fontsize=7.5, labelcolor=F.INK)
    save(fig, out)


def fig_claim_auroc(split: str, out: Path) -> None:
    res = json.loads(Path(f"results/{split}/claim_auroc.json").read_text())["results"]
    scores = [("s", "MiniCheck support $s$"), ("score_c (C)", "Two-signal score (C)"), ("c", "Closed-book confidence $c$"),
              ("1 - con", "1 $-$ NLI contradiction"), ("SelfCheckGPT-NLI", "SelfCheckGPT-NLI")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True, constrained_layout=True)
    for ax, (ds, title) in zip(axes, DSETS):
        ticks = []
        for i, (key, slab) in enumerate(scores):
            y0 = len(scores) - 1 - i
            ticks.append((y0, slab))
            for j, (gen, (color, marker, glab)) in enumerate(GEN_COLORS.items()):
                r = res[f"{ds}/{gen}"].get(key)
                if r is None:
                    continue
                y = y0 + (1 - j) * 0.22
                ax.plot(r["ci95"], [y, y], color=color, linewidth=1.6, solid_capstyle="round")
                ax.plot(r["auroc"], y, marker=marker, color=color, markersize=5, markeredgecolor=F.SURFACE, markeredgewidth=0.8,
                        linestyle="none", label=glab if i == 0 else None)
        ax.axvline(0.5, color=F.INK2, linewidth=0.8)
        ax.annotate("chance", (0.5, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -2), textcoords="offset points",
                    fontsize=7, color=F.INK2, va="top")
        ax.set_yticks([t for t, _ in ticks])
        ax.set_yticklabels([lab for _, lab in ticks], fontsize=7.5)
        ax.set_xlim(0.45, 0.75)
        ax.set_ylim(-0.6, len(scores) - 0.4)
        ax.set_xlabel("AUROC against judge labels (95% CI)")
        ax.set_title(title, fontsize=8.5, color=F.INK, loc="left")
        ax.grid(axis="x", color=F.GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside upper center", ncol=3, frameon=False, fontsize=7.5, labelcolor=F.INK)
    save(fig, out)


def fig_precision_vs_content(split: str, out: Path) -> None:
    revisers = [("standard RAG (all claims)", "standard RAG", "o"), ("CoVe factor+revise (closed-book)", "CoVe, closed-book", "v"),
                ("CoVe factor+revise (retrieval-verified)", "CoVe, retrieval-verified", "^"),
                ("RARR-style revise-all", "RARR-style", "D"), ("FLARE-direct", "FLARE", "P")]
    filters = [("CF (score s)", "CF (score s)"), ("C (two-signal)", "C (two-signal)"), ("A (repair all)", "A (repair all)"),
               ("SelfCheckGPT-NLI", "SelfCheckGPT-NLI")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), constrained_layout=True)
    for ax, (ds, title) in zip(axes, DSETS):
        # one comparison file per dataset (Qwen2.5-7B; named comparison__0.95__... by an old naming bug, see compare_longform.py)
        f = glob.glob(f"results/{split}/{ds}/comparison__*__v2__{JT}.json")
        assert len(f) == 1, f
        rows = json.loads(Path(f[0]).read_text())["rows"]
        assert "Qwen2.5-7B-Instruct" in rows["standard RAG (all claims)"]["run"]
        for key, skey in filters:
            color, marker, label = F.SERIES[skey]
            pts = [rows[f"{key} @ target {t}"] for t in ("0.8", "0.85", "0.9", "0.95")]
            ax.plot([p["factual_precision_micro"]["mean"] for p in pts], [p["supported_claims_per_response"]["mean"] for p in pts],
                    color=color, marker=marker, markersize=4, linewidth=1.6, markeredgecolor=F.SURFACE, markeredgewidth=0.6, label=label)
        for key, lab, marker in revisers:
            p = rows[key]
            x, y = p["factual_precision_micro"]["mean"], p["supported_claims_per_response"]["mean"]
            ax.plot(x, y, marker=marker, color=NEUTRAL, markersize=5.5, linestyle="none", markeredgecolor=F.SURFACE, markeredgewidth=0.6)
            # label right of the point, except where that would cross the filter lines (ASQA CoVe, retrieval-verified)
            below = ds == "asqa" and "retrieval" in key
            ax.annotate(lab, (x, y), xytext=(0, -8) if below else (5, 0), textcoords="offset points", fontsize=6.8,
                        color=F.INK2, va="top" if below else "center", ha="center" if below else "left")
        ax.set_yscale("log")
        ticks = [t for t in (0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20) if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("Micro precision of kept claims (judge)")
        ax.set_title(f"{title} (Qwen2.5-7B)", fontsize=8.5, color=F.INK, loc="left")
        F.style(ax)
    axes[0].set_ylabel("Supported claims per answer (log scale)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, [f"{x}, targets 0.80–0.95" for x in lab], loc="outside upper center", ncol=2, frameon=False, fontsize=7.5,
               labelcolor=F.INK)
    save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    od = Path("results/figures") / a.split
    od.mkdir(parents=True, exist_ok=True)
    numbers: dict = {}
    fig_retention_others(a.split, od / "retention_other_generators.pdf")
    fig_score_distributions(a.split, od / "score_distributions.pdf", numbers)
    fig_claim_auroc(a.split, od / "claim_auroc.pdf")
    fig_precision_vs_content(a.split, od / "precision_vs_content.pdf")
    Path(f"results/{a.split}/figure_numbers.json").write_text(json.dumps(
        {"label": "EXPLORATORY", "script": "scripts/figures_extra.py", "git_commit": git_commit(), "judge_tag": JT,
         "note": "CF / C thresholds at target 0.90 over partitions seeded 0-99 (cal_frac 0.5), Qwen2.5-7B default runs",
         "values": numbers}, indent=1))
    for k, v in numbers.items():
        print(k, {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()})
    print("wrote", sorted(str(p) for p in od.glob("*.pdf")))


if __name__ == "__main__":
    main()
