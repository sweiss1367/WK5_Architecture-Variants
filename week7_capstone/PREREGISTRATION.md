# PREREGISTRATION.md: Positional-Encoding Ablation

**Status: DRAFT, NOT FROZEN.** Fields marked `[PILOT]` or `[SMOKE]` are filled in by those stages and then frozen. Nothing in this file may be edited after the freeze except through the amendment log (Section 11).
**Freeze record (filled at freeze):** commit hash `______`, UTC timestamp `______`, frozen by `______`.
**Rule:** no main-run result (seeds 0-4, any arm) may exist before the freeze record is filled in.

The hypothesis in Section 2 and the practical-significance argument for δ are the author's scientific judgment. I drafted them for review, and they should be edited before freezing if they don't reflect what you actually expect.

## 1. Research question

For a small Transformer that forecasts the next 24 hours of electricity demand from the previous 168 hours, does the choice of positional encoding (learned absolute, fixed sinusoidal, rotary) change validation MAPE by more than the run-to-run variation from random seeds, with all other settings held fixed?

## 2. Hypotheses (directional, drafted for your review)

Context: input is demand only (no calendar features), windows always start at 00:00, and the head is a mean-pool, so the only absolute time-of-day information available to the model is whatever the encoding supplies, plus whatever it infers from the shape of the demand curve itself.

- **H1 (directional).** RoPE has **higher** validation MAPE than learned absolute encoding: `d_C > 0`, where `d_C` is defined in Section 4. Reason: RoPE is purely relative, so it cannot tell the model "this token is 17:00," while learned absolute encoding can.
- **H2 (exploratory, no direction).** Sinusoidal vs learned absolute: no directional prediction. Both supply absolute position.
- Either hypothesis may be wrong, and a null or reversed result will be reported as such.

## 3. Primary metric and arms

- **Primary metric:** validation MAPE (%), over all validation windows and all 24 horizons, in demand units, using final-step weights.
- **Arms:** A learned absolute (reference), B sinusoidal, C rotary (RoPE).
- **Comparisons:** B vs A and C vs A (planned). C vs B descriptive only.
- **Seeds:** main seeds 0, 1, 2. Pilot seeds 900, 901, 902 and calibration seed 900 are never mixed into main tables.
- **Declared extension:** seeds 3 and 4 are added only if the Stage-1 verdict (Section 5) is "direction only" or "no difference detectable" for a planned comparison. Both stages are reported, labeled by stage. If Stage 1 gives "supported, tentative," no seeds are added.

## 4. Matched differences

For comparison arm X and seed `s`: `d_s = MAPE(X, s) − MAPE(A, s)`. Negative means X has lower error than A. For each comparison, report all `d_s`, the mean `d̄`, the SD `SD_d` (n−1), and the three leave-one-out means. Seed-level raw MAPE values are reported for every arm.

## 5. Interpretation rule

| Verdict | Conditions (all must hold) |
|---|---|
| **Supported, tentative** | (a) all `d_s` share a sign; (b) \|d̄\| ≥ δ; (c) \|d̄\| ≥ k × SD_d; (d) every leave-one-out mean has the same sign and magnitude ≥ δ |
| **Direction only** | (a) and (b) hold, but (c) or (d) does not |
| **No difference detectable at this budget** | Anything else |

No verdict is called "confirmed." No p-values are reported. Whatever the verdict, the report states the smallest mean difference that would have satisfied (c), `k × SD_d`.

For H1 the sign of `d_C` must be positive for a "supported" verdict in the hypothesized direction. A significant-looking negative `d_C` is reported as the opposite direction and does not support H1.

## 6. Meaning of δ and k, and how they are selected

### 6.1 δ (practical threshold)

**Meaning.** δ is the smallest mean paired validation-MAPE difference, in absolute MAPE percentage points, that I would treat as worth acting on in a forecasting release decision. It is a statement about practical importance, not about statistical detectability.

**Selection procedure.**
1. δ is chosen after the pilot and before the first main run.
2. δ is derived from **scale information only**: the pilot's reference-arm (A) validation MAPE level and the seasonal-naive baseline's MAPE. It is **not** derived from the pilot's between-arm differences, from which arm looked better, or from `SD_d`.
3. The value is a round number of MAPE points (multiple of 0.05), with a written argument below naming the release decision it corresponds to.
4. δ is not changed after the freeze.

**Feasibility check (does not change δ).** After δ is set, compare it to the pilot's `SD_d`. If `k × SD_d` from the pilot is far above δ, this design probably cannot separate differences of size δ. The consequence is that the final report is written as a noise-floor characterization, as stated in the plan. δ is not lowered to make the study look more decisive.

**δ value:** `[PILOT] ______` MAPE points.
**Argument for δ:** `[PILOT] ______`

### 6.2 k (signal-to-spread guard)

**Meaning.** k is the multiplier in condition (c): the mean paired difference must be at least k times the SD of the paired differences. It is a guard against calling a result when the individual differences disagree widely.

**Selection procedure.**
1. k is provisionally **2**, set now, before any pilot data exists.
2. The pilot may not change k. It may only be used for the feasibility check in 6.1.
3. A change before the freeze requires an entry in the amendment log with a reason that does not refer to pilot or main results.
4. k is not changed after the freeze.

**k value:** `2` (provisional until the freeze).

## 7. Step budget (selected from training loss only)

The step count `S` is chosen so that all three arms have trained to a plateau. It is **not** chosen from validation error.

**Calibration (pilot part B).** Seed 900 only, three arms, `S ∈ {1000, 2000, 4000}` (grid provisional, may be edited before the pilot starts). Each is a separate full run, because the learning-rate schedule depends on `S`.
**Rule.** Choose the smallest `S` such that, for every arm, going from this `S` to the next larger value improves the final training loss (mean over the last 100 steps, training windows only) by less than 2%. The 2% threshold is provisional and may be edited before the pilot starts, but not after.
**If no grid value meets the rule,** use the largest `S` affordable within the budget and record the non-convergence as a limitation.
**Note on validation reuse.** The pilot and main runs score on the same validation windows. No model, hyperparameter or step-count decision uses validation error. The pilot's validation numbers are used only for the sanity checks (beats the seasonal-naive baselines) and the `SD_d` feasibility check. This is stated as a limitation.

**S value:** `[PILOT] ______`

## 8. Determinism policy selection (feasibility only)

- **P1:** full deterministic (seeds, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, attention backend forced to a deterministic one if needed). **P2:** seeded only, default kernels.
- **Selection is based on projected GPU time only.** The policy-timing script times the same tiny training configuration under P1 and P2, and it does **not compute or record any validation or test accuracy**. Accuracy plays no part in the decision.
- **Rule.** Project total GPU time (smoke + both pilot parts + 9 main runs + the declared extension of 6 runs + one retry allowance of 3 runs) using measured P1 timings. If the projection fits within the planning ceiling (3 hours), adopt P1. Otherwise adopt P2.
- **Recording.** The decision, timings, projection, timestamp and GPU identity are written to `results/determinism_decision.json` and copied below. The same policy is then used for all arms, seeds and stages, and is never changed.
- Limits of GPU reproducibility are documented in the package (see `EXPERIMENT_PLAN.md` Section 6.4).

**Policy:** `[SMOKE] ______`  **Projected total (h):** `[SMOKE] ______`

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

All fixed settings are in `configs/base.yaml`. Measurement definitions (training duration, inference latency, peak memory), attempt handling and the session manifest are as specified in `EXPERIMENT_PLAN.md` Sections 6.6 and 7. Arm differences that cannot be removed are listed in `EXPERIMENT_PLAN.md` Section 4.3. A run whose determinism or software state differs from the manifest is flagged, not silently included.

## 11. Amendment log

Every change after the freeze, and every pre-freeze change to a provisional value, is recorded here with date, commit hash, old value, new value and a reason that does not refer to main-run results. After the first main run exists, only additions of clarifications that change no rule are allowed, and each is marked as post-hoc.

| Date (UTC) | Commit | Field | Old | New | Reason |
|---|---|---|---|---|---|
| (none yet) | | | | | |
