"""Publication figures (vector PDF), generated only from analysis files written by the analysis scripts.

Fig. 1 retention_vs_target: supported claims kept per answer vs conformal target, CF / C / A / SelfCheckGPT, one panel
       per long-form dataset (separate y-scales: one axis per panel), standard RAG (no filter) as a dashed reference.
Fig. 2 shortform_risk_coverage: selective risk vs coverage for CF / C / A on each short-form dataset (small multiples).
Palette: validated categorical slots 1-4 (dataviz reference palette; light mode; contrast relief = legend + direct
labels + marker shapes). Usage: python scripts/figures.py --split dev|test [--model-tag Qwen2.5-7B-Instruct]
"""
import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hallu import metrics as M  # noqa: E402
from hallu.store import read_jsonl  # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"  # white: figures sit on a white journal page
SERIES = {"CF (score s)": ("#2a78d6", "o", "CF (filter only)"),
          "C (two-signal)": ("#eb6834", "s", "C (two-signal)"),
          "A (repair all)": ("#1baf7a", "^", "A (repair + filter)"),
          "SelfCheckGPT-NLI": ("#eda100", "D", "SelfCheckGPT")}
TARGETS = ["0.8", "0.85", "0.9", "0.95"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def direct_labels(ax, x: float, ends: list[tuple[float, str]]) -> None:
    """End-of-line labels without collisions: series whose end values are within 2% of the axis range (visually
    overlapping) share one label ("A = CF"); remaining labels are pushed apart vertically to a minimum gap of 7%."""
    lo, hi = ax.get_ylim()
    rng = (hi - lo) or 1.0
    groups: list[list[tuple[float, str]]] = []
    for v, name in sorted(ends):
        if groups and abs(v - groups[-1][-1][0]) < 0.02 * rng:
            groups[-1].append((v, name))
        else:
            groups.append([(v, name)])
    placed = []
    for g in groups:
        y = sum(v for v, _ in g) / len(g)
        if placed and y - placed[-1] < 0.07 * rng:
            y = placed[-1] + 0.07 * rng
        placed.append(y)
        ax.annotate(" = ".join(n for _, n in g), (x, sum(v for v, _ in g) / len(g)), xytext=(x + 0.006, y),
                    textcoords="data", fontsize=7, color=INK2, va="center")


def is_default(cfg: dict) -> bool:
    """The frozen main configuration (no Phase 6 ablation / sensitivity option set)."""
    return (cfg.get("prompts") == "v2" and cfg.get("k") == 5 and cfg.get("tau_s") == 0.5 and cfg.get("retriever") == "bm25"
            and cfg.get("verifier") == "MiniCheck-Flan-T5-Large" and not ({"triage", "decompose", "ablation_prompts", "limit"} & set(cfg)))


def latest_longform(split: str, ds: str, tag: str) -> Path:
    """The single default-configuration long-form run of this generator (never an ablation run)."""
    c = [Path(d) for d in glob.glob(f"results/{split}/{ds}/longform__{tag}__*")
         if (Path(d) / "summary.json").exists() and is_default(json.loads((Path(d) / "summary.json").read_text())["config"])]
    if len(c) != 1:
        raise SystemExit(f"expected exactly one default run for {split}/{ds}/{tag}, found {c}")
    return c[0]


def fig_retention(split: str, tag: str, judge: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
    for ax, (ds, title) in zip(axes, (("bio", "Biographies"), ("asqa", "ASQA"))):
        an = json.loads((latest_longform(split, ds, tag) / f"analysis__{judge}.json").read_text())
        xs = [float(t) for t in TARGETS]
        ends = []
        for key, (color, marker, label) in SERIES.items():
            if key not in an["targets"][TARGETS[0]]:
                continue
            ys = [an["targets"][t][key]["supported_claims_per_response"]["mean"] for t in TARGETS]
            ax.plot(xs, ys, color=color, marker=marker, markersize=4.5, linewidth=2, label=label,
                    markeredgecolor=SURFACE, markeredgewidth=0.8)
            ends.append((ys[-1], label.split(" (")[0]))
        direct_labels(ax, xs[-1], ends)
        ref = an["unfiltered"]["standard RAG (all claims)"]["supported_claims_per_response"]["mean"]
        ax.set_title(f"{title}  (standard RAG, no filter: {ref:.1f})", fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("Target P(all kept claims supported)")
        ax.set_ylabel("Supported claims kept per answer")
        ax.set_xticks(xs)
        ax.set_xlim(0.79, 0.985)
        ax.set_ylim(bottom=0)
        style(ax)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside upper center", ncol=4, frameon=False, fontsize=7.5, labelcolor=INK)
    fig.savefig(out, format="pdf")
    fig.savefig(out.with_suffix(".png"), dpi=200)  # preview for visual inspection
    plt.close(fig)


def fig_shortform(split: str, tag: str, out: Path) -> None:
    dsets = [("popqa", "PopQA"), ("nq", "NQ"), ("triviaqa", "TriviaQA"), ("hotpotqa", "HotpotQA"), ("2wiki", "2Wiki")]
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.0), sharey=True, constrained_layout=True)
    for ax, (ds, title) in zip(axes, dsets):
        c = [Path(d) for d in glob.glob(f"results/{split}/{ds}/shortform__{tag}__*") if (Path(d) / "items.jsonl").exists()]
        if len(c) != 1:
            raise SystemExit(f"expected exactly one short-form run for {split}/{ds}/{tag}, found {c}")
        rows = read_jsonl(c[0] / "items.jsonl")
        for key, score, corr in (("CF (score s)", "s", "rag_match"), ("C (two-signal)", "score_c", "rag_match"), ("A (repair all)", "A_score", "A_match")):
            color, _, label = SERIES[key]
            cov, risk = M.risk_coverage([r[score] for r in rows], [bool(r[corr]) for r in rows])
            ax.plot(cov, risk, color=color, linewidth=1.6, label=label)
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.set_xlabel("Coverage")
        ax.set_xlim(0, 1)
        style(ax)
    axes[0].set_ylabel("Selective risk (error rate)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside upper center", ncol=3, frameon=False, fontsize=7.5, labelcolor=INK)
    fig.savefig(out, format="pdf")
    fig.savefig(out.with_suffix(".png"), dpi=200)  # preview for visual inspection
    plt.close(fig)


def fig_deltas(split: str, judge: str, out: Path) -> None:
    """Fig. 3: paired-bootstrap differences in supported claims per answer at target 0.90 (partition 0, 95% CI),
    A − CF and C − CF, for every generator (default configuration) and every G1 ablation / sensitivity run.
    The 3B run is excluded (its decomposition fails; reported separately). Exploratory except the G1 A − CF rows."""
    abl = json.loads(Path(f"results/{split}/ablations.json").read_text())["runs"]
    ex = json.loads(Path(f"results/{split}/exploratory_tests.json").read_text())["longform"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.4), constrained_layout=True)
    for ax, (ds, title) in zip(axes, (("bio", "Biographies"), ("asqa", "ASQA"))):
        rows: list[tuple[str, dict, dict]] = []
        for gen, name in (("Mistral-7B-Instruct-v0.3", "Mistral-7B (default)"), ("Falcon3-7B-Instruct", "Falcon3-7B (default)")):
            an = json.loads((latest_longform(split, ds, gen) / f"analysis__{judge}.json").read_text())
            rows.append((name, an["A_vs_CF_at_0.90"]["supported_per_response"],
                         ex[f"{ds}/{gen}"]["results"]["0.9"]["C vs CF"]["supported_per_response"]))
        g1 = sorted((r for r in abl.values() if r["dataset"] == ds and "3B" not in r["configuration"]),
                    key=lambda r: (r["configuration"] != "default (G1)", "model=" in r["configuration"], r["configuration"]))
        for r in g1:
            name = {"default (G1)": "Qwen2.5-7B (default)"}.get(r["configuration"], r["configuration"])
            name = name.replace("model=Qwen/Qwen2.5-14B-Instruct-AWQ", "Qwen2.5-14B-AWQ (default)").replace("verifier=", "verifier: ")
            name = name.split(" (liuyanyi")[0].replace("DeBERTa-v3-large-mnli-fever-anli-ling-wanli P(entailment)", "NLI entailment")
            name = name.replace("generator P(True) given passage", "generator").replace("AlignScore-large", "AlignScore")
            # readable labels instead of config keys (same wording as Table ablations)
            name = re.sub(r"^tau_s=([\d.]+)$", r"support threshold $\\tau_s$ = \1", name)
            name = re.sub(r"^k=(\d+)$", r"\1 retrieved passages", name)
            name = {"retriever=dense": "dense retriever (bge-small)", "triage=one_axis": "one-axis triage",
                    "decompose=sentences": "sentence-level claims"}.get(name, name)
            rows.append((name, r["tests_at_0.90"]["A vs CF"], r["tests_at_0.90"]["C vs CF"]))
        order = [rows[2]] + rows[:2] + rows[3:]  # G1 default on top, then the other generators, then G1 variants
        y = list(range(len(order)))[::-1]
        for (lab_, color, marker, off), key in ((("A − CF", SERIES["A (repair all)"][0], "^", 0.17), 1), (("C − CF", SERIES["C (two-signal)"][0], "s", -0.17), 2)):
            for yi, row in zip(y, order):
                b = row[key]
                ax.plot([b["ci_low"], b["ci_high"]], [yi + off] * 2, color=color, linewidth=1.6, solid_capstyle="round")
                ax.plot(b["diff"], yi + off, marker=marker, color=color, markersize=5, markeredgecolor=SURFACE, markeredgewidth=0.8,
                        linestyle="none", label=lab_ if yi == y[0] else None)
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in order] if ds == "bio" else [""] * len(order), fontsize=7)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("Δ supported claims per answer vs CF at target 0.90")
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside upper center", ncol=2, frameon=False, fontsize=7.5, labelcolor=INK)
    fig.savefig(out, format="pdf")
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--model-tag", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--judge-tag", default="phi-4-FP8-dynamic")
    a = ap.parse_args()
    od = Path("results/figures") / a.split
    od.mkdir(parents=True, exist_ok=True)
    fig_retention(a.split, a.model_tag, a.judge_tag, od / f"retention_vs_target__{a.model_tag}.pdf")
    fig_shortform(a.split, a.model_tag, od / f"shortform_risk_coverage__{a.model_tag}.pdf")
    if Path(f"results/{a.split}/ablations.json").exists() and Path(f"results/{a.split}/exploratory_tests.json").exists():
        fig_deltas(a.split, a.judge_tag, od / "deltas_vs_cf.pdf")
    print("wrote", sorted(str(p) for p in od.glob("*.pdf")))


if __name__ == "__main__":
    main()
