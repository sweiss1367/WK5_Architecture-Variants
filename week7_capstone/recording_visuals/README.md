# Recording visuals

Screen-share assets for the Week 7 video. Everything is generated from the repository's frozen files; nothing here
changes an experimental file, a result, a config, or the pre-registration.

## Open it

Double-click `recording_dashboard.html` (no server needed). It is one file with the figures embedded.

- Next and previous: `→` `←`, or the on-screen buttons. Jump to a section with `1` to `5`.
- `H` hides the dark navigation bar so only the slide is in the shared window. The slide scales to any window size and is
  designed for 1920×1080.
- Opening `recording_dashboard.html#4` goes straight to section 4.

| Key | Section | Source |
|---|---|---|
| 1 | Changed vs Fixed | V1, built from `configs/base.yaml`, `configs/arm_{A,B,C}.yaml`, the nine run files |
| 2 | Frozen Rule Applied | V2, built from `results/main_summary.json`, `configs/base.yaml`, `PREREGISTRATION.md` section 5 |
| 3 | Seed-Level Validation MAPE | `figures/fig1_seed_level_mape.png`, unchanged |
| 4 | Matched Differences | `figures/fig2_matched_differences.png`, unchanged |
| 5 | Cost Comparison | `figures/fig3_cost.png`, unchanged |

## Which visual goes with which recording chunk

Chunk numbers follow the 8-chunk screen-share plan (README, pre-registration, data, figures, reproducibility, Annex IV).
Chunks 2, 3, 7 and 8 use repository documents, not the dashboard.

| Chunk | Dashboard section to show |
|---|---|
| 1. Question, arms, controlled settings (0:00–1:45) | Press `1` (V1). It replaces the "Changed vs fixed" visual from the plan. Show `README.md` and the arm config files before or after it. |
| 2. Frozen pre-registration and rule (1:45–3:00) | None. Use `PREREGISTRATION.md` section 5, `configs/base.yaml` lines 70–76, and the terminal `freeze.py --verify`. |
| 3. Dataset and leakage (3:00–3:45) | None. Use `PREREGISTRATION.md` section 9 and the `make_data.py --verify` terminal output. |
| 4. Validation MAPE and seed variation (3:45–4:45) | Press `3` (fig 1). |
| 5. Matched differences and frozen verdicts (4:45–6:00) | Press `4` (fig 2), then press `2` (V2). |
| 6. Training time, latency, memory (6:00–6:45) | Press `5` (fig 3). |
| 7. Reproducibility package (6:45–7:45) | None. Use `README.md` sections "What you can reproduce, and where" and "What was tested in a clean environment". |
| 8. Annex IV, limitations, recommendation (7:45–9:00) | None. Use `docs/ANNEX_IV.pdf` pages 1, 3 and 4. |

Order of keys across the whole recording: `1`, (documents), `3`, `4`, `2`, `5`, (documents).

## Files

| File | What it is |
|---|---|
| `recording_dashboard.html` | The five-section dashboard |
| `v1_changed_vs_fixed.png` / `.svg` | V1 at 1920×1080 |
| `v2_frozen_rule_applied.png` / `.svg` | V2 at 1920×1080 |
| `build_visuals.py` | Generates the SVGs, the dashboard, and (with `--png`) the PNGs |
| `verify_visuals.py` | Recomputes every number from the raw run files and checks the SVGs, embedded figures, and freeze |
| `verification_report.txt` | Output of the last verification run |

## Rebuild and verify

```
python recording_visuals/build_visuals.py --png     # needs PyYAML, Pillow, and Chromium for the PNGs
python recording_visuals/verify_visuals.py          # also runs freeze.py --verify
```

`--png` looks for Chromium at `/opt/pw-browsers/...`. Elsewhere, build without `--png` and export the SVGs from a browser.

## Notes

- Fonts: the SVGs and HTML ask for Inter and fall back to Segoe UI, Helvetica or Arial. The PNGs were rendered with Inter.
  Text widths on another machine can differ by a few pixels; the PNGs are the exact versions.
- V2 shows the mean as +0.0904 in the table and +0.090 in the bottom strip (same value, rounded to three places).
- Leave-one-out means in V2 come from `results/main_summary.json` and match a recomputation from the run files.
- The pass/fail marks use a symbol and a word as well as color.
