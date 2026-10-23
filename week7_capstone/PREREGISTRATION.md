# PREREGISTRATION.md: Positional-Encoding Ablation

**Status: FROZEN.** Approved by the user in the working session of 2026-10-23. Nothing below may be changed except through the amendment log (Section 11), and any change to a frozen file blocks main runs (Section 12).
**Freeze record:** frozen 2026-10-23T15:11Z (UTC) by the user's approval. A file cannot contain the hash of its own commit, so the commit and the SHA-256 of every frozen file are in `results/freeze_record.json`, written by `src/freeze.py --write` immediately after the commit that contains this text. Check it with `python src/freeze.py --verify`.
**Rule:** no main-run result (any seed, any arm) may exist before the freeze.

The decision rule, δ and k below were fixed before any main-experiment result exists. The analysis that applies them (`src/analyze_main.py`) was written and tested before the freeze and is itself frozen.

## 1. Research question

For a small Transformer that forecasts the next 24 hours of electricity demand from the previous 168 hours, does the choice of positional encoding (learned absolute, fixed sinusoidal, rotary) change validation MAPE by more than the run-to-run variation from random seeds, with all other settings held fixed?

## 2. Hypotheses

**Primary hypothesis (neutral, no direction).** Positional-encoding choice (learned absolute, fixed sinusoidal, rotary) affects forecasting accuracy on the synthetic electricity-demand traces. The experiment asks whether the observed between-arm differences in validation MAPE exceed run-to-run variation across seeds. No arm is predicted to win, and the study is reported the same way whether the answer is yes, no or inconclusive.

**Secondary hypothesis (directional, conjectural).** RoPE has **higher** validation MAPE than learned absolute encoding: `d_C > 0` (Section 4).

*Reasoning, and why it is only a conjecture.* The model receives demand values only, with no calendar features, windows always start at 00:00, and the head is a mean-pool. Learned and sinusoidal encodings therefore give the model an explicit absolute-position signal, which in these windows corresponds to hour of day. RoPE gives relative offsets only. That is the reason to expect a disadvantage. It is not established, because:
- the demand curve itself carries phase information (the shape of the daily peaks, the weekday/weekend level shift), so a model can infer where it is in the day from content without any absolute encoding;
- relative offsets between tokens still let attention compare "now" to "24 hours ago" and "168 hours ago," which is most of what a seasonal forecast needs;
- the model is small and the training budget short, so differences may be too small to see.

The direction is therefore not treated as expected or likely. It is retained as a secondary hypothesis because the input design (no calendar features) makes it a reasonable thing to test. A null result, or `d_C < 0`, is reported plainly, and the secondary hypothesis never changes the primary verdict logic in Section 5.

**Exploratory.** Sinusoidal vs learned absolute (`d_B`): no direction predicted.

**Preserved at the freeze.** The primary hypothesis stays neutral. The directional statement above is a secondary conjecture, and its direction is not treated as expected.

**Family of comparisons.** There are two planned comparisons (B vs A, C vs A) and one descriptive one (C vs B). No multiplicity correction is applied, and no verdict is described as confirmed.

## 3. Primary metric and arms

- **Primary metric:** validation MAPE (%), over all validation windows and all 24 horizons, in demand units, using final-step weights.
- **Arms:** A learned absolute (reference), B sinusoidal, C rotary (RoPE).
- **Comparisons:** B vs A and C vs A (planned). C vs B descriptive only.
- **Seeds:** main seeds 0, 1, 2, the same three for every arm (matched: identical shared initial weights, data order and dropout stream per seed). Pilot seeds 900, 901, 902 and calibration seed 900 are never mixed into main tables.
- **Arm A is rerun in the main experiment.** It is evaluated under exactly the same frozen protocol as B and C. The pilot's arm A results (seeds 900 to 902) are **not substituted for it** and do not enter any main comparison. The pilot only informed the practical threshold δ and the noise scale.
- **Main experiment protocol.** Nine runs (3 arms x 3 seeds) in **one session on one GPU**, on the frozen commit, 8000 steps, policy P1, the existing dataset. Run order: seed blocks with the arm order rotated by seed (seed 0: A,B,C; seed 1: B,C,A; seed 2: C,A,B). Per run: validation MAPE (and MAE), test MAPE (secondary, never used for the verdict), end-to-end training time, batch-1 inference latency, and peak GPU memory for training and for inference. Every attempt is reported: completed, failed and interrupted. A resumed experiment is disclosed as resumed.
- **Pilot confound, documented.** The pilot's three seeds ran on three different physical GPUs, so seed and GPU are confounded in the noise estimate. Same-GPU reproducibility was exact (repeat of seed 901). Reproducibility across GPUs was not tested. The main comparisons are matched within one session, which removes this confound from the paired differences. Findings apply primarily to the tested environment.
- **Optional extension (seeds 3 and 4): not part of the required experiment and not run.** It may be considered only after the nine main runs and the remaining compute budget have been reviewed, only if the main verdict for a planned comparison is "direction only" or "no difference detectable", and only with the user's explicit written approval (the run script refuses the extension stage without an approval file). If run, both stages are reported, labeled by stage, with the same frozen rule applied to five matched seeds.

## 4. Matched differences

For comparison arm X and seed `s`: `d_s = MAPE(X, s) − MAPE(A, s)`. Negative means X has lower error than A. For each comparison, report all `d_s`, the mean `d̄`, the SD `SD_d` (n−1), and the three leave-one-out means. Seed-level raw MAPE values are reported for every arm.

## 5. Interpretation rule

| Verdict | Conditions (all must hold) |
|---|---|
| **Supported, tentative** | (a) all `d_s` share a sign; (b) \|d̄\| ≥ δ; (c) \|d̄\| ≥ k × SD_d; (d) every leave-one-out mean has the same sign and magnitude ≥ δ |
| **Direction only** | (a) and (b) hold, but (c) or (d) does not |
| **No difference detectable at this budget** | Anything else |

No verdict is called "confirmed." No p-values are reported. Whatever the verdict, the report states the smallest mean difference that would have satisfied (c), `k × SD_d`.

**Sign and magnitude.** Lower MAPE is better, so an improvement over arm A is a **negative** matched difference. The magnitude conditions (b), (c) and (d) use the **absolute** mean difference, so a large improvement and a large deterioration are screened identically. The **sign** of the mean difference says which arm performs better, and is always reported next to the verdict ("the arm has lower error than A" or "higher error than A").

**These are exploratory screening criteria for a three-seed study, not significance tests.** The requirement that the mean be at least k times the standard deviation of the differences (condition c) and the leave-one-out checks (condition d) are pre-declared heuristics. They do not establish statistical significance, and no probability or confidence statement is attached to them. Three same-sign differences occur by chance 25% of the time (2 x 0.5 cubed), so condition (a) cannot carry a claim alone, and the standard deviation of three differences is itself a rough estimate. For that reason the strongest verdict is called "supported, tentative", never "confirmed", and every report states `k × SD_d`.

A "supported, tentative" verdict is a statement about a difference in either direction. For the secondary hypothesis, `d_C` must be positive for the verdict to count in the conjectured direction. A negative `d_C` meeting the conditions is reported as the opposite direction and does not support the secondary hypothesis.

## 6. Meaning of δ and k, and how they are selected

### 6.1 δ (practical threshold)

**Meaning.** δ is the smallest mean paired validation-MAPE difference, in absolute MAPE percentage points, that I would treat as worth acting on in a forecasting release decision. It is a statement about practical importance, not about statistical detectability.

**Selection procedure.**
1. δ is chosen after the pilot and before the first main run.
2. δ is derived from **scale information only**: the pilot's reference-arm (A) validation MAPE level and the seasonal-naive baseline's MAPE. It is **not** derived from any between-arm validation difference, from which arm looked better, or from `SD_d`. The pilot runs validation for arms B and C only as a by-product of the run script. Those values are **not inspected** before the freeze. The pilot analysis prints reference-arm validation results and, for all arms, training loss and timing only.
3. The value is a round number of MAPE points (multiple of 0.05), with a written argument below naming the release decision it corresponds to.
4. δ is not changed after the freeze.

**Feasibility check (does not change δ).** The noise estimate comes from the reference arm alone: `SD_A`, the seed-to-seed SD of arm A's validation MAPE over pilot seeds 900, 901, 902. A paired difference between two arms with independent seed noise of that size would have SD about `√2 × SD_A`. Matching seeds can only reduce this, so the estimate is conservative. After δ is set, compare `k × √2 × SD_A` with δ. If it is far above δ, this design probably cannot separate differences of size δ, and the final report is written as a noise-floor characterization. δ is not lowered to make the study look more decisive, and this check changes no setting.

**δ value: `0.10` MAPE points** (approved by the user).
**Argument for δ.** A change of 0.10 points is about 2.1% of the reference model's validation error (arm A: 4.67%) and about 4% of the value the model adds over a trivial seasonal forecast (2.41 points better than the better seasonal baseline). A smaller difference is too small to justify changing the architecture (δ = 0.05, about 1.1% of the error, was rejected because its practical importance is less convincing). A larger one (δ = 0.25, a 5.4% change in error) sets a substantially higher bar with no reason tied to the forecasting application. The argument uses scale information only: no between-arm result and no variance. The pilot's small seed-to-seed SD (0.00734) is encouraging but is **not** used to justify lowering δ.

**Feasibility check, performed (δ unchanged).** Pilot seeds 900, 901, 902 gave arm A validation MAPE 4.67367, 4.67434, 4.66130 (mean 4.66977, SD 0.00734, range 0.01304; 95% interval for the true SD 0.0038 to 0.0461). With k = 2, `k × √2 × SD_A` is 0.0208 at the estimate (about a fifth of δ) and 0.1305 at the upper end of the interval (above δ). So δ = 0.10 is detectable if the true seed spread is near the estimate and is not guaranteed to be if it is near the upper end. Arms B and C may have different spread. If the main-run `SD_d` makes `k × SD_d` larger than δ, the report says the design could not separate differences of size δ and states the smallest difference it could. δ is not lowered afterwards.

### 6.2 k (signal-to-spread guard)

**Meaning.** k is the multiplier in condition (c): the mean paired difference must be at least k times the SD of the paired differences. It is a guard against calling a result when the individual differences disagree widely.

**Selection procedure.**
1. k was set to **2** before any pilot data existed.
2. The pilot may not change k. It may only be used for the feasibility check in 6.1.
3. A change before the freeze requires an entry in the amendment log with a reason that does not refer to pilot or main results.
4. k is not changed after the freeze.

**k value: `2`** (frozen; approved by the user).

## 7. Step budget (selected from training loss only, staged)

The step count `S` is chosen so that training has reached a plateau. It is **not** chosen from validation error, and no between-arm validation difference is used. The calibration is staged so that it grows only if the earlier stage does not support a defensible budget.

**Stage 1: timing and training-loss calibration (3 runs).** Reference arm A, seed 900, `S ∈ {1000, 2000, 4000}`. Each is a separate run, because the learning-rate schedule depends on `S`. These runs also give per-step timing for the budget projection.
- **Rule.** Choose the smallest `S` such that going to the next larger grid value improves the final training loss (mean over the last 100 steps, training windows only) by less than 2%. The grid and the 2% threshold are provisional and may be edited before Stage 1 starts, not after.
- **Expansion trigger.** If no grid value meets the rule (loss still improving by 2% or more from 2000 to 4000), add `S = 8000` for arm A only if the timing projection (Section 8) allows it. Otherwise use 4000 and record non-convergence as a limitation.

**Stage 2: confirmation on the other arms (4 runs, expansion).** Arms B and C, seed 900, at the Stage-1 value `S*` and at the next larger grid value, applying the same 2% rule per arm. I include this because a plateau on arm A alone does not show the budget is adequate for B and C, and a step budget tuned on one arm would favor it. If either arm fails the rule, `S*` moves up one grid value for **all** arms and the check is repeated. This can be dropped without affecting any other part of the protocol, at the cost of a weaker step-budget claim, which would then be stated as a limitation.

**Noise pilot (2 additional runs).** Arm A only, seeds 901 and 902 at `S*`. Seed 900 at `S*` is reused from Stage 1, because the configuration is identical. The three runs give `SD_A`. The pilot **estimates variability**. It is not used to choose the best encoding, and no between-arm validation difference enters any decision.

**Total pilot and calibration:** 3 + 4 + 2 = 9 short runs, fewer if Stage 2 is dropped.

**Blinding in code.** In the calibration and pilot stages, arms B and C skip all validation and test evaluation, so no between-arm validation number exists to inspect. Arm A is evaluated. Main-stage runs evaluate every arm.

**Enforcement.** The calibration decision is computed by `src/calibrate.py`, which loads run files through a whitelist of training-loss fields. Validation and test values never enter it, and a test checks that adversarial validation values leave the decision unchanged.

**Note on validation reuse.** Pilot and main runs score on the same validation windows. No model, hyperparameter or step-count decision uses validation error. Pilot validation numbers are used only for the reference arm's sanity checks and for `SD_A`. This is stated as a limitation.

**S value:** `8000`, selected by the pre-registered fallback. Stage 1 found no grid value with a final-training-loss improvement below 2% (9.22%, 6.36%, 5.12% for the three doublings from 1000 to 8000), and 8000 is the largest value tried. The models are **not converged**; this is recorded as a limitation. Stage 2 was not called for. Approved by the user.

## 8. Determinism policy selection (feasibility only)

- **P1:** full deterministic (seeds, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, attention backend forced to a deterministic one if needed). **P2:** seeded only, default kernels.
- **Selection is based on projected GPU time only.** `src/choose_policy.py` times the real run script, in the smoke stage with `--timing-only`, under P1 and P2 on identical workloads (same arms, seed, data, batch size and step counts; fresh process per run; warm-up discarded; policy order alternating). It does **not compute or record any validation or test accuracy**. GPU model, CUDA, PyTorch and Python versions, driver, clocks and benchmark conditions are recorded. The measured timing difference is printed and saved **before** the rule is applied.
- **Rule.** Project total GPU time (smoke + 9 pilot and calibration runs + 9 main runs + the declared extension of 6 runs + one retry allowance of 3 runs, each at the largest grid step count, 4000) using measured P1 timings. If the projection fits within the planning ceiling (3 hours), adopt P1. Otherwise adopt P2.
- **Recording and enforcement.** The decision, timings, projection, timestamp and GPU identity are written to `results/determinism_decision.json` and copied below. `determinism.policy` in `configs/base.yaml` is set to match, and the run script refuses any non-smoke run if they disagree. The same policy is then used for all arms, seeds and stages, and is never changed.
- Limits of GPU reproducibility are documented in the package (see `EXPERIMENT_PLAN.md` Section 6.4).

**Policy:** `P1` (selected by the rule in two smoke sessions, `results/determinism_decision.json`; approved by the user)  **Projected total (h):** `1.28` (session 2, commit `3fca186f`; session 1 gave 2.13)

## 9. Leakage prevention

Windows within a split overlap each other by design (context 168 h, stride 24 h). The rule is that no information from a validation or test **target** reaches training, and no future information reaches any input.

1. A window belongs to the split containing its first target hour.
2. Every training window's target **and** context lie entirely before the validation start.
3. Validation window targets lie in the validation period. Their contexts may reach back into the training period, because past history is available at forecast time. Test windows follow the same rule against train and validation.
4. No validation or test target hour appears as a target or context hour in any training window. No test target hour appears in any training or validation window's target.
5. Normalization uses only each window's own context. No scaler, mean or statistic is fitted on data from more than one split.
6. The generator's gap-filling rule uses only past values, so it cannot copy a future value backward.
7. Automated tests assert 2-6 on the generated arrays and fail the run if violated. The test list and results are part of the package.

## 10. Fixed settings, measurement and session policy

All fixed settings are in `configs/base.yaml`. Measurement definitions (training duration, inference latency, peak memory), attempt handling and the session manifest are as specified in `EXPERIMENT_PLAN.md` Sections 6.6 and 7. Arm differences that cannot be removed are listed in `EXPERIMENT_PLAN.md` Section 4.3. A run whose determinism or software state differs from the manifest is flagged, not silently included. The run script also refuses non-smoke runs when the environment differs from `requirements.lock.txt`, when tracked code differs from the committed code, or when the policy decision is missing. `main` and `extension` runs are refused while this file still has unfilled underscore placeholders, and (Section 12) unless the freeze record matches.

## 11. Amendment log

Every change after the freeze, and every pre-freeze change to a provisional value, is recorded here with date, commit hash, old value, new value and a reason that does not refer to main-run results. After the first main run exists, only additions of clarifications that change no rule are allowed, and each is marked as post-hoc.

| Date (UTC) | Commit | Field | Old | New | Reason |
|---|---|---|---|---|---|
| 2026-10-23 | freeze commit | Sections 6.1 and 6.2 | δ and k unset or provisional | δ = 0.10, k = 2 | Approved by the user after the noise pilot; argued from scale information only (Section 6.1). No main-run result existed |
| 2026-10-23 | freeze commit | Section 5 | rule text | added: absolute magnitude with sign giving direction; exploratory-criteria wording | Clarification requested by the user before the freeze; the numeric rule is unchanged. No main-run result existed |
| 2026-10-23 | freeze commit | Section 3 | declared extension, arm A pilot results | extension needs explicit user approval after review; arm A is rerun in the main experiment, pilot arm A not substituted | Requested by the user before the freeze. No main-run result existed |
| 2026-10-19 | 5d9843a6 | Section 7 expansion trigger | grid {1000, 2000, 4000} | grid {1000, 2000, 4000, 8000} for arm A | **Not an amendment.** The pre-declared trigger fired: stage 1 found no grid value with final training-loss improvement under 2% (9.22% for 1000 to 2000, 6.36% for 2000 to 4000), and the timing projection allows S=8000. Recorded for transparency. No threshold or rule was changed |

## 12. What is frozen, and how the freeze is enforced

Frozen files (SHA-256 recorded in `results/freeze_record.json`): this file, `configs/base.yaml` (including the `analysis` section with δ and k), the three arm configs, `requirements.lock.txt`, and the code that runs and analyses the experiment (`src/run_ablation.py`, `src/model.py`, `src/make_data.py`, `src/analyze_main.py`, `src/run_main.py`, `src/check_env.py`, `src/freeze.py`). Notebooks, tests and documentation are not frozen, so they can be corrected without invalidating the freeze.

Main and extension runs are refused unless every frozen file still matches its recorded hash and the recorded commit is an ancestor of the code being run. The extension also needs the user's approval file. Changing a frozen file after the freeze requires a new, logged freeze, and is a reportable event.
