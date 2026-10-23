# Video outline (notes, not a script)

Target length **9:00**, hard limits 8:00 to 10:00. Face visible the whole time (screen share in a corner window is fine).
Say it in your own words from these cues. Reading sentences aloud is the thing the rubric penalises, so keep each cue
to a phrase. The finding and its limits get equal time (about 2:15 each).

Format: MP4, MOV or WebM. Check the length on the exported file, not on the recorder.

| Time | Block | Show | What to get across (cues) |
|---|---|---|---|
| 0:00-0:45 | Question | face only, then title card | What was ablated and why this one: one factor (positional encoding), three arms A learned, B sinusoidal, C RoPE. Electricity-demand forecasting, 168 h in, 24 h out. The question is "does the choice matter for accuracy, and what does it cost?" |
| 0:45-2:15 | Design | repo layout, `configs/` | Arms differ in one config field. Same data hash, same shared initial weights (checked by hash). Matched seeds 0, 1, 2. 8000 steps, no early stopping, no per-arm tuning. Why seed variation: one seed would be an anecdote. |
| 2:15-3:30 | Rule set before looking | `PREREGISTRATION.md` | Decision rule frozen before the nine runs: delta = 0.10 MAPE points, k = 2, three same-sign differences, leave-one-out check. These are screening criteria for three seeds, not significance tests; chance of three same-sign differences is 25%. Calibration and pilot never looked at validation results for B or C. Policy P1 chosen on time feasibility only. |
| 3:30-5:15 | What we observed | `fig1`, then `fig2` | Validation MAPE: A 4.667, B 4.658, C 4.757. B vs A: signs mixed, mean -0.009, nothing detectable. C vs A: higher error on all three seeds, mean +0.090, which is 0.010 under the 0.10 threshold, so the rule does not call it. Say both halves: consistent direction and below threshold. Label: no difference detectable at this budget. |
| 5:15-6:15 | Cost | `fig3` | Under P1 on one Tesla T4: C training 343 s vs 313 s (+9.8%, every pair), batch-1 latency 4.69 vs 4.34 ms, memory the same. Narrow claim only: this model, this hardware, this workload, deterministic kernels. Not "RoPE is slower". |
| 6:15-7:30 | Limits | notes or slide of bullets | Three seeds and SD from three numbers. Small synthetic data and a small model. 8000 steps, not converged (fallback rule). One GPU, one session. Pilot SD for A (0.0073) came out lower than the main-run SD (0.0329), so seed noise was underestimated, and delta was not lowered to fit. Extension to seeds 3 and 4 not run; the three-seed results stay primary. |
| 7:30-8:30 | Reproduction | terminal or README | Three levels: (1) no GPU, `analyze_main.py` and `make_figures.py` regenerate tables and figures from the committed run files, (2) CPU tests and `freeze.py --verify`, (3) Colab notebook for the nine runs on a T4. Run files carry commit, GPU id, hashes. Say what was tested and what was not (the GPU path was run once, in the original session). |
| 8:30-9:00 | Recommendation and documentation | face only | Not enough evidence to change positional encoding for accuracy. A and B indistinguishable here. RoPE showed measurable runtime cost without a qualifying accuracy gain. Annex IV-style document written as an academic exercise, not a conformity assessment. One sentence on what you would do next with more budget (more seeds, longer training), clearly marked as not done. |

## Things to avoid saying

- "RoPE is worse" or "RoPE is less efficient" in general. The accurate form is the narrow one above.
- "Significant", "proved", "confirmed". Use "no difference detectable at this budget" and "supported, tentative" as in the rule.
- That C "almost" passed, without the other half: it missed the threshold, and the mean is also the average of three runs.
- Anything about the Annex IV document that implies compliance or certification.

## Before recording

- [ ] Run `python src/freeze.py --verify` once on screen, or have its output ready (shows the freeze holds).
- [ ] Open `figures/fig1`, `fig2`, `fig3` in tabs in that order; zoom so labels are readable at 720p.
- [ ] Have `results/main_summary.json` open to the verdict lines in case you need to point to a number.
- [ ] Camera at eye level, face in frame during every screen share, quiet room, microphone check.
- [ ] Time a rehearsal with a stopwatch. If over 9:45, cut from Design, not from Limits.

## After recording

- [ ] Export MP4 (H.264). Check the duration of the exported file: 8:00 to 10:00.
- [ ] Watch the first 30 seconds and one random minute to confirm face and audio.
- [ ] Upload to the LMS and confirm the upload plays back.
