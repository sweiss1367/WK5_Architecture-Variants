# Figures

Regenerate from the committed run files (no GPU needed):

    pip install -r requirements-figures.txt    # matplotlib, used only here; not part of the frozen lock
    python src/make_figures.py

`figure_manifest.json` records the SHA-256 of each of the nine main-stage run files the figures were drawn from, the
matplotlib version, and the palette check.

| File | Shows |
|---|---|
| `fig1_seed_level_mape` | Validation MAPE for every run (3 arms x 3 seeds). Gray lines join the same seed. The y-axis does not start at zero (stated on the figure). |
| `fig2_matched_differences` | Per-seed differences against arm A (and C minus B, descriptive only) with the pre-registered delta = 0.10 band. |
| `fig3_cost` | Training time, batch-1 p50 latency, peak training memory under policy P1. One panel per measure, axes start at zero. |

Colour: A blue, B orange, C green, in that fixed order. The palette passed the colour-vision-deficiency checks of the
data-viz validator; the green has low contrast on the light background (2.74:1), so every mark is labelled directly and
the same numbers are in `results/main_runs.csv` and `results/paired_differences.csv` (the table view).

The figures show the frozen analysis; they do not add to it. Fig. 2 verdicts are the ones in `results/main_summary.json`.
