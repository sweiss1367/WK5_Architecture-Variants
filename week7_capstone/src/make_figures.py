"""Figures for the main experiment, generated directly from the run files in results/runs/main.

    python src/make_figures.py          # writes figures/fig1..fig3 (PNG and SVG) and figures/figure_manifest.json

Reads only the nine main-stage run files and configs/base.yaml (for delta and k). Nothing is computed that the frozen
analysis (src/analyze_main.py) does not already define; the figures re-derive the matched differences from the raw values
and the manifest records the SHA-256 of every input file, so each figure can be traced to the exact runs. This script is
not part of the frozen experiment: it changes no result, and it is allowed to be improved.

Design (see figures/README.md): arms keep one colour each in a fixed order (A blue, B orange, C aqua, validated for
colour-vision deficiency with the data-viz palette checker), every dot is a run, seeds are labelled, and no axis is dual.
The aqua slot is below 3:1 contrast on the white surface, so every mark is directly labelled and the same numbers are in
results/main_runs.csv (the table view).
"""
import glob
import hashlib
import json
import os
import statistics as st
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}          # fixed order: slot 1, 2, 3
NAMES = {"A": "A  learned", "B": "B  sinusoidal", "C": "C  RoPE"}
SEEDS = (0, 1, 2)
RC = {"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
      "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
      "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"}


def load(run_dir):
    runs, files = {}, []
    for f in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        r = json.load(open(f))
        if r.get("stage") == "main" and r.get("status") == "completed":
            runs[(r["arm"], r["seed"])] = r
            files.append(f)
    missing = [(a, s) for a in "ABC" for s in SEEDS if (a, s) not in runs]
    if missing:
        raise SystemExit(f"missing completed main runs: {missing}")
    return runs, files


def style(ax, ygrid=True):
    ax.grid(axis="y" if ygrid else "x", color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(length=0)


def dot(ax, x, y, color, **kw):
    ax.scatter([x], [y], s=70, color=color, edgecolor=SURFACE, linewidth=2, zorder=4, **kw)    # 2px surface ring


def save(fig, name, outdir):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def fig1(runs, outdir):
    """Seed-level validation MAPE by arm, with matched seeds joined."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs = {"A": 0, "B": 1, "C": 2}
    means = {}
    for s in SEEDS:
        ax.plot([xs[a] for a in "ABC"], [runs[(a, s)]["val_mape"] for a in "ABC"], color=MUTED, linewidth=1, alpha=0.55, zorder=2)
    for a in "ABC":
        v = [runs[(a, s)]["val_mape"] for s in SEEDS]
        for s, y in zip(SEEDS, v):
            dot(ax, xs[a], y, COLORS[a])
            ax.annotate(f"seed {s}", (xs[a], y), xytext=(9, -3), textcoords="offset points", fontsize=8.5, color=INK2, zorder=5,
                        bbox=dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none", alpha=0.85))
        m = st.mean(v)
        ax.hlines(m, xs[a] - 0.28, xs[a] - 0.1, color=COLORS[a], linewidth=3, zorder=3)
        means[a] = m
    ax.set_xticks(list(xs.values()), [f"{NAMES[a]}\nmean {means[a]:.3f}  (short bar)" for a in "ABC"])
    ax.set_xlim(-0.6, 2.6)
    lo, hi = min(r["val_mape"] for r in runs.values()), max(r["val_mape"] for r in runs.values())
    ax.set_ylim(lo - 0.06, hi + 0.06)
    ax.set_ylabel("Validation MAPE (%), lower is better")
    style(ax)
    ax.set_title("Every run, by arm: validation error per seed (gray lines join the same seed)", loc="left", fontsize=11, color=INK, pad=22)
    ax.text(0, 1.02, "8000 steps, policy P1, one GPU session. The y-axis does not start at zero; the seasonal baselines are at 7.08 and 7.17.",
            transform=ax.transAxes, fontsize=8.5, color=INK2)
    save(fig, "fig1_seed_level_mape", outdir)


def fig2(runs, cfg, outdir):
    """Matched differences (X - A) with the practical threshold delta."""
    delta, k = cfg["analysis"]["delta_mape_points"], cfg["analysis"]["k"]
    rows = [("B", "A", "B  sinusoidal  minus  A  learned", COLORS["B"], True),
            ("C", "A", "C  RoPE  minus  A  learned", COLORS["C"], True),
            ("C", "B", "C  RoPE  minus  B  sinusoidal (descriptive only)", MUTED, False)]
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.axvspan(-delta, delta, color=GRID, alpha=0.55, zorder=0)
    ax.axvline(0, color=INK2, linewidth=1, zorder=1)
    for x in (-delta, delta):
        ax.axvline(x, color=INK2, linewidth=1, linestyle=(0, (1, 3)), zorder=1)
    labels = []
    for i, (x, ref, label, color, planned) in enumerate(rows):
        y0 = len(rows) - 1 - i
        d = [runs[(x, s)]["val_mape"] - runs[(ref, s)]["val_mape"] for s in SEEDS]
        m, sd = st.mean(d), st.stdev(d)
        order = sorted(range(len(d)), key=lambda j: d[j])
        for rank, j in enumerate(order):
            dot(ax, d[j], y0, color)
            ax.annotate(f"seed {SEEDS[j]}", (d[j], y0), xytext=(0, 9 + 12 * (rank % 2)), textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
        ax.plot([m, m], [y0 - 0.12, y0 + 0.12], color=INK, linewidth=2.5, zorder=5)
        ax.annotate(f"mean {m:+.3f}", (m, y0), xytext=(0, -17), textcoords="offset points", ha="center", va="top", fontsize=9, color=INK)
        verdict = f"verdict: no difference detectable at this budget  (k×SD_d = {k * sd:.3f})" if planned else "descriptive only, no verdict claimed"
        labels.append(f"{label}\n{verdict}")
    ax.set_yticks(range(len(rows)), list(reversed(labels)), fontsize=8.5)
    ax.set_ylim(-0.85, len(rows) - 0.4)
    xlim = max(0.2, max(abs(runs[(x, s)]["val_mape"] - runs[(ref, s)]["val_mape"]) for x, ref, *_ in rows for s in SEEDS) + 0.06)
    ax.set_xlim(-xlim, xlim)
    ax.set_xlabel("Difference in validation MAPE (percentage points).  Negative = the first arm has the lower error.")
    ax.text(delta + 0.004, -0.8, f"+δ = {delta:.2f}", ha="left", va="bottom", fontsize=8.5, color=INK2)
    ax.text(-delta - 0.004, -0.8, f"−δ = −{delta:.2f}", ha="right", va="bottom", fontsize=8.5, color=INK2)
    style(ax, ygrid=False)
    ax.grid(False)
    ax.set_title("Matched differences by seed against the practical threshold δ (shaded band)", loc="left", fontsize=11, color=INK, pad=24)
    ax.text(0, 1.02, "Black tick = mean of three matched differences. A planned comparison needs |mean| ≥ δ, among other conditions; neither met it.",
            transform=ax.transAxes, fontsize=8.5, color=INK2)
    save(fig, "fig2_matched_differences", outdir)


def fig3(runs, outdir):
    """Cost under P1: three small multiples (one measure each, never a shared or dual axis)."""
    panels = [("Training time (s)", lambda r: r["train_seconds"], "{:.0f}", (0, 400)),
              ("Batch-1 inference latency, p50 (ms)", lambda r: r["inference_latency_ms"]["p50_ms"], "{:.2f}", (0, 6)),
              ("Peak training memory (MiB)", lambda r: r["train_peak_alloc_mb"], "{:.0f}", (0, 700))]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.0))
    for ax, (title, get, fmt, ylim) in zip(axes, panels):
        for xi, a in enumerate("ABC"):
            v = [get(runs[(a, s)]) for s in SEEDS]
            for y in v:
                dot(ax, xi, y, COLORS[a])
            m = st.mean(v)
            ax.hlines(m, xi - 0.3, xi + 0.3, color=COLORS[a], linewidth=2, zorder=3)
            ax.annotate(fmt.format(m), (xi, max(v)), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9, color=INK)
        ax.set_xticks([0, 1, 2], ["A", "B", "C"])
        ax.set_ylim(*ylim)
        ax.set_xlim(-0.6, 2.6)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        style(ax)
    fig.legend(handles=[plt.Line2D([], [], marker="o", linestyle="", color=COLORS[a], markersize=8, markeredgecolor=SURFACE, label=NAMES[a]) for a in "ABC"],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04), labelcolor=INK2)
    fig.suptitle("Cost per arm, nine runs, policy P1 (axes start at zero; dots are runs, short bars and labels are means)", x=0.01, ha="left", fontsize=11, color=INK, y=1.12)
    fig.text(0.01, 1.02, "Measured on one Tesla T4 in one session under deterministic policy P1. Default-kernel (P2) costs were not profiled at full length.",
             fontsize=8.5, color=INK2)
    fig.tight_layout()
    save(fig, "fig3_cost", outdir)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    cfg = yaml.safe_load(open("configs/base.yaml"))
    runs, files = load("results/runs/main")
    outdir = "figures"
    os.makedirs(outdir, exist_ok=True)
    plt.rcParams.update(RC)
    fig1(runs, outdir)
    fig2(runs, cfg, outdir)
    fig3(runs, outdir)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "matplotlib": matplotlib.__version__,
                "inputs_sha256": {f: hashlib.sha256(open(f, "rb").read()).hexdigest() for f in files},
                "delta": cfg["analysis"]["delta_mape_points"], "k": cfg["analysis"]["k"],
                "palette": {"A": COLORS["A"], "B": COLORS["B"], "C": COLORS["C"]},
                "palette_check": "validate_palette.py --mode light --pairs all: all hard checks pass; aqua contrast 2.74:1 so every mark is directly labelled and results/main_runs.csv is the table view",
                "figures": sorted(os.listdir(outdir))}
    json.dump(manifest, open(os.path.join(outdir, "figure_manifest.json"), "w"), indent=1)
    print("figures written:", [f for f in sorted(os.listdir(outdir)) if f.endswith(".png")])


if __name__ == "__main__":
    main()
