"""
Figures for the briefing, drawn only from files in results/ (no numbers typed in).

    python src/make_figures.py [--which contamination fairness probes]

    results/figures/contamination_flagged.png   flagged % by n, raw vs. unexplained after review
    results/figures/fairness_slice_differences.png   paired slice differences with 95% CIs
    results/figures/probe_failures_by_family.png    probe counts per family

Colors: two categorical hues validated with the dataviz skill's validate_palette.js
(#2a78d6 / #eb6834, light surface, all checks pass). Identity is also carried by direct
labels, never color alone. A CI that includes zero is drawn hollow ("unresolved").
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def contamination(res, out):
    rev = json.load(open(os.path.join(res, "contamination_review.json")))["rows"]
    df = pd.DataFrame(rev).sort_values(["dataset", "n"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(df) + 1.5))
    for i, r in df.iterrows():
        ax.barh(i, r["pct_flagged"], color=BLUE, height=0.55, zorder=3)
        ax.text(max(r["pct_flagged"], 0) + 1.5, i,
                f"{r['pct_flagged']:.0f}% flagged ({r['flagged_items']} of {r['items']}); "
                f"{r['pct_unexplained']:.0f}% unexplained after review",
                va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"{r['dataset']}  n={r['n']}" for _, r in df.iterrows()], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 150)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% of evaluation items sharing at least one n-gram with train/val", color=MUTED)
    ax.set_title("n-gram overlap sweep (n = 8, 13, 20 words)", loc="left", fontsize=10, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "contamination_flagged.png"), dpi=160)
    plt.close(fig)


def fairness(res, out):
    d = pd.read_csv(os.path.join(res, "fairness_slice_differences.csv"))
    d = d[(d.stratum == "marginal") & (d.metric.isin(["rationale_omission", "unsupported_policy", "completeness", "invalid_or_refusal", "rationale_recall"]))]
    rows = d.sort_values(["slice", "metric"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(rows) + 1.4))
    for i, r in rows.iterrows():
        col = BLUE if r["slice"] == "language" else ORANGE
        excl = r["ci_low"] > 0 or r["ci_high"] < 0
        ax.plot([r["ci_low_pp"], r["ci_high_pp"]], [i, i], color=col, linewidth=2, zorder=3)
        ax.plot([r["estimate_pp"]], [i], marker="o", markersize=7, markeredgecolor=col, markeredgewidth=2,
                markerfacecolor=col if excl else "#fcfcfb", zorder=4)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['metric']}  ({r['contrast']})" for _, r in rows.iterrows()], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("paired difference, percentage points (95% CI)", color=MUTED)
    ax.set_title("Slice differences. Filled = CI excludes 0; hollow = unresolved", loc="left", fontsize=10, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fairness_slice_differences.png"), dpi=160)
    plt.close(fig)


def probes(res, out):
    d = pd.read_csv(os.path.join(res, "probe_summary_by_family.csv"))
    d = d.sort_values("failed_any_run", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    y = range(len(d))
    ax.barh([i for i in y], d["failed_any_run"], color=ORANGE, height=0.55, zorder=3, label="failed on at least one run")
    ax.barh([i for i in y], d["failed_all_runs"], color=BLUE, height=0.55, zorder=4, label="failed on every run")
    for i, (_, r) in zip(y, d.iterrows()):
        ax.text(r["failed_any_run"] + 0.3, i, f"{int(r['failed_any_run'])} of {int(r['n_probes'])}", va="center", fontsize=8, color=INK)
    ax.set_yticks(list(y))
    ax.set_yticklabels(d["family"], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, d["n_probes"].max() + 3)
    ax.set_xlabel("probes (counts, not rates)", color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "probe_failures_by_family.png"), dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--which", nargs="+", default=["contamination", "fairness", "probes"])
    args = ap.parse_args()
    out = os.path.join(args.results, "figures")
    os.makedirs(out, exist_ok=True)
    for w in args.which:
        {"contamination": contamination, "fairness": fairness, "probes": probes}[w](args.results, out)
        print("wrote", w)


if __name__ == "__main__":
    main()
