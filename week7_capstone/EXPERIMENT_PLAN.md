# EXPERIMENT_PLAN.md: CST 627 Week 7 Capstone

**Status:** DRAFT v3, proposed as the execution plan. **The factor is selected: positional encoding**, conditional on one fact only you can supply (Q1, Section 14). Items are tagged [M] mandatory, [A] adopted commitment, [R] recommended, using the tiers defined in `REQUIREMENTS.md` Section 5. No implementation exists. Nothing here has been run, so every number below is a planning estimate and is labeled as one.
**Companion file:** `REQUIREMENTS.md` (approved). Status of decisions D1-D12 is tracked in Section 15 of this file.
**Location rule:** all Week 7 work lives in `week7_capstone/`. Earlier coursework is not modified.

## 1. Recommendation in one paragraph

Ablate **positional-encoding choice** in a small Transformer that forecasts the next 24 hours of electricity demand from the previous 7 days. The three settings are **learned absolute** (the reference, and what the Week 5 classifier uses), **fixed sinusoidal**, and **rotary (RoPE)**. The dataset is a **seeded synthetic hourly demand generator** for 16 co-op-style feeders. The primary accuracy metric is **validation MAPE** at a fixed step count with no early stopping. Each run also records **training duration** and **batch-1 inference latency** as separate quantities, plus peak GPU memory for each phase. The comparison uses **matched per-seed differences**, not seed ranges. Nine main runs (3 settings × 3 seeds) are expected to fit well inside one Colab session. The fallback factor, if positional encoding can't be defended as "carried forward," is **LoRA rank**.

**On diffusion.** This plan does not use a diffusion model or a noise schedule. I evaluated noise schedules as one of the five permitted factors (Section 3.2, candidate 5) and rejected them. A diffusion forecaster needs multi-step sampling, a probabilistic metric, and a schedule that interacts with loss weighting and sampling steps, so it would carry more unresolved confounds and cost more than 9 runs can justify without a pilot. I cannot confirm from here that nine such runs fit, and I would not commit to it. A simpler model with a cleaner comparison is the better trade for both of the top two technical rubric categories.

## 2. Approved and revised design decisions, and where they land

| Decision | How the plan implements it |
|---|---|
| 3 matched seeds per setting to start | Seeds 0, 1, 2 for every arm. The shared backbone gets identical initial weights, data order and dropout stream per seed (6.3). A pre-declared extension to seeds 3, 4 is optional (Section 9). |
| One determinism policy, same for all arms, chosen before benchmarking (D6) | Section 6.4. The policy is selected once in the smoke stage, using a pre-declared budget rule, then frozen for every arm and every stage. |
| Distinguish training duration from inference latency; peak memory consistently (D7) | Section 7. Training duration, batch-1 inference latency, and peak memory for each phase are recorded in every run. |
| Everything fixed except the factor | Section 6.2. |
| No early stopping | Fixed step count, final-step weights are evaluated, no checkpoint selection. |
| Metric and comparison rule defined before running (D9) | Section 8. Matched paired differences with an interpretation rule written in advance. Numeric threshold frozen after the pilot, before main runs. |
| Cautious about three seeds | Section 8 states what three seeds can and cannot show, and reports the smallest difference this design can detect. |
| Save every run independently; distinguish uninterrupted from resumed (D10) | Section 6.6. Per-run files, attempt counters, and a session manifest. |
| Verify Annex IV against the regulation | Section 11. |
| Separate directory | `week7_capstone/`. |

## 3. Evaluating the five candidate factors

Scenario: Cobalt Grid sells short-horizon electricity-demand forecasting to rural utility co-ops, and Dr. Solberg needs one narrow architectural question answered.
Ratings are my judgment from the Week 5 code, the course pages, and general knowledge of these methods. None are measured.

### 3.1 Summary

| Factor | Implementation complexity | Confound risk | Compute cost (T4) | Forecasting relevance | Risk of failure |
|---|---|---|---|---|---|
| **Positional encoding** | Low. A swap inside one attention block, about 15 lines per setting. | Medium, manageable (4.1) | Low | High. Demand is dominated by daily and weekly cycles, which is what position information is for. | Low-medium |
| LoRA rank | Medium. Needs a shared pretrained base plus an adapter stage. | Medium-high (alpha scaling, two-stage design) | Low | Medium | Medium. Effects between ranks are likely tiny. |
| Quantization precision | Medium. bitsandbytes dependency on a T4. | Medium | Low | Low-medium | Medium-high for this study design |
| Multimodal fusion | Medium-high | High (arms differ in capacity) | Low-medium | Very high in practice (weather) | Medium-high |
| Diffusion/flow noise schedule | High | High | Medium-high (sampling is expensive) | Medium (probabilistic forecasts) | High |

### 3.2 Candidate notes

**1. Positional encoding (recommended).**
- *Rubric fit.* The manipulated factor is one module in one block. The config diff is a single line. Each control decision is easy to defend aloud.
- *Scenario fit.* With hourly steps as tokens, the positional encoding is what tells the model "this is hour 17 of a day." That is the structure a demand forecast depends on.
- *Explainability.* Week 5's classifier already has `self.pos_emb = nn.Embedding(MAX_LEN, D_MODEL)`. I'm treating that as evidence the factor was carried forward from earlier weeks. I can only see Week 5, so please confirm (Q1).
- *Known weaknesses:*
  1. Training time and peak memory will probably be close to flat across settings (learned embeddings add about 10k parameters, the others add none). Inference latency may differ more, since RoPE adds operations per layer, and at this model size latency is dominated by per-operation overhead. Either result is reportable, and measuring both is why D7 matters here.
  2. Learned absolute has a parameter-count difference that is intrinsic to the factor. It is disclosed, not hidden.
  3. Relative and absolute encodings treat "hour of day" differently. Section 4.1 controls this.

**2. LoRA rank (fallback).**
- Needs a base model pretrained once, then adapted at ranks such as 4, 8, 16.
- *Confound:* LoRA scales its update by `alpha / r`. Holding alpha fixed makes update scale change with rank. Holding `alpha / r` fixed changes alpha. Either way a second quantity moves with the factor. The lesson calls this learning-rate coupling.
- *Noise-floor concern:* a base model pretrained once and shared means seed spread only reflects adapter initialization and data order, which understates what a stranger would see retraining everything.
- *Result risk:* ranks 4, 8, 16 on a small model will probably be indistinguishable. That is a legitimate null, but a weak story for the video.

**3. Quantization precision.**
- Post-training quantization needs no extra training, so training duration and peak training memory are identical across arms by construction. The informative cost axis would be inference latency alone.
- Week 5 already measured this on a T4. INT8 accuracy was essentially equal to FP32 (0.9764 vs 0.9763 mean), and INT8 was slower (about 11.5 ms vs 5.2 ms median latency). Repeating it on a new dataset adds little.
- Training-precision variants are limited on a T4, which lacks native bfloat16, so FP32 / FP16-AMP / BF16 would not be a clean three-setting ladder.

**4. Multimodal fusion (demand + weather).**
- The most realistic setup, but the arms (early concatenation, late fusion, cross-attention) differ in parameter count and compute by design. "One factor" is hard to defend.
- Synthetic weather would be generated by my own code, so the result mostly reflects how I coupled temperature to demand. Real weather needs a second acquisition and join step.

**5. Diffusion or flow-matching noise schedule (rejected).**
- Needs a conditional denoising forecaster, multi-step sampling for evaluation, and a probabilistic metric such as CRPS. A schedule changes the noise-level distribution and the effective loss weighting, and the number of sampling steps interacts with it. Each is a possible second factor.
- Highest implementation and debugging cost, highest chance of a result I can't explain, and per-run cost I can't bound without building it first.

### 3.3 Why positional encoding wins on your priority list

| Priority | Result |
|---|---|
| 1. Exemplary rubric | Strong on design control and on the seed-honesty story. An honest-null outcome is fully presentable. |
| 2. One-factor control | The best of the five. Confounds are listed in 4.1 and are testable. |
| 3. One Colab session | Easily. Estimated 9 runs of roughly 1-3 minutes of training each, to be measured in the pilot. |
| 4. Interpretable measurements | Accuracy: good. Training cost: probably flat, reported as such. Inference latency: may separate the settings. |
| 5. Explainable | Best of the five. |

## 4. Experimental design

### 4.1 Confounds specific to positional encoding, and the control for each

| # | Confound | Control |
|---|---|---|
| 1 | **Calendar covariates** (hour-of-day, day-of-week inputs) would hand every arm the same time information and hide the effect. | Model input is the demand value only. No calendar features. Stated as a limitation: real systems would add them. |
| 2 | **Window alignment.** If windows start at arbitrary hours, "position 0" doesn't mean midnight and the encodings are unfairly handicapped or helped. | Windows always start at 00:00, so position 24k always lands on midnight for all arms. This favors absolute encodings by design and is disclosed. |
| 3 | **Output head leaking position.** A head that flattens all 168 positions can learn position itself and mask the encoding. | Mean-pool over tokens (as in Week 5), then linear to 24 outputs. Order information has to come through the encoding. Verified in the pilot that the model still learns (V3). |
| 4 | **Parameter count.** Learned absolute adds 168×d parameters. | Intrinsic to the factor. Reported per arm. Not "fixed" by padding other arms. |
| 5 | **Where the encoding acts.** Learned and sinusoidal are added to the input once. RoPE rotates queries and keys in every layer. | Intrinsic to the techniques. Standard implementation of each, documented. |
| 6 | **Magnitude mismatch.** A positional term larger than the value embedding can dominate it. | Log the norm of the value embedding vs the positional term at initialization for each arm (V6). No arm-specific scaling tricks. |
| 7 | **Hyperparameters tuned on one arm.** Tuning learning rate on the reference arm would favor it. | No per-arm tuning. Use Week 5-style AdamW defaults. The pilot only checks that all three arms train to a plateau. If one doesn't, the budget changes for all arms, before the main runs. |
| 8 | **Initialization differing between arms.** Extra parameters in one arm shift the RNG stream, so "seed 1" would not mean the same backbone. | Build the shared backbone first from seed `s`. Build the positional module from a separate seed. Reseed immediately before the training loop (6.3). |

### 4.2 Settings (exactly three)

| Setting | Implementation | Extra parameters |
|---|---|---|
| **A. Learned absolute** (reference) | `nn.Embedding(168, d)` added to the input projection | 168×d |
| **B. Sinusoidal** | Fixed sin/cos table (Vaswani et al.), added to the input projection | 0 |
| **C. Rotary (RoPE)** | Rotation applied to q and k in each attention layer | 0 |

A "no positional encoding" arm would be a useful control, but you asked for exactly three settings, so it is optional and not in the plan.

## 5. Dataset and architecture

### 5.1 Dataset: seeded synthetic hourly demand

- **Why synthetic.** Fully scripted and rerunnable, no download or license risk, and provenance is the generator, its parameters, and a hash.
- **Cost of that choice.** Results describe the seasonality I assumed. This is the first limitation in the Annex IV document, and the video says so.
- **Series.** 16 feeders ("co-ops"), hourly, 3 years each. Per-series parameters are drawn once from documented ranges with a fixed data seed (42), including peak season (summer-peaking vs winter-peaking), base load, and noise level.
- **Generator components (all documented in `make_data.py`).**
  1. Base load plus a daily profile (morning and evening peaks) that differs on weekdays and weekends.
  2. A latent temperature series (annual cycle + daily cycle + autocorrelated anomaly) that drives a nonlinear heating/cooling load. The model never sees temperature.
  3. Multiplicative noise, a few documented outlier hours, and short gaps filled by a documented rule.
  4. A hard positive floor on demand so MAPE is well defined.
- **Splits (temporal, fixed, identical for every run).** Months 1-24 train, 25-30 validation, 31-36 test. No shuffling across time.
- **Windows.** 168-hour context ending at midnight, next 24 hours as the target, stride 24.
- **Estimated size (to be confirmed).** About 11,500 training windows (16 series × roughly 720 days).
- **Normalization.** Each window divided by its own context mean, so the model predicts a ratio. Metrics are computed after converting back to demand units.
- **Same data every run.** Seeds vary model initialization, data order and dropout. They do not vary the dataset. This understates total variability and is stated in the limitations.

### 5.2 Architecture (identical except the positional module)

Small encoder-only Transformer, in the style of the Week 5 classifier.

| Item | Setting (starting values, to be confirmed in the pilot) |
|---|---|
| Input | 168 tokens, each a scalar demand value linearly projected to `d_model` |
| `d_model` / layers / heads / FF | 64 / 3 / 4 / 256 |
| Dropout | 0.1, same for all arms |
| Head | Mean-pool over tokens, linear to 24 outputs |
| Parameters | roughly 150k, plus 10.7k for setting A |
| Precision | FP32 throughout |

### 5.3 Training (identical for all arms)

| Item | Setting |
|---|---|
| Loss | L1 on the normalized target |
| Optimizer | AdamW, weight decay, linear warmup then linear decay, as in Week 5 |
| Batch size | 128 |
| Steps | Fixed. The number is set in the pilot and frozen before main runs. Start at roughly 3,000. |
| Early stopping | None. Final-step weights are evaluated. No checkpoint selection. |
| Gradient accumulation | None. A batch-size fallback that silently happened would be a confound, so the script raises an error instead. |

## 6. Run protocol

### 6.1 Entry point and layout

```
week7_capstone/
  REQUIREMENTS.md
  EXPERIMENT_PLAN.md
  PREREGISTRATION.md        # written before any main run
  README.md
  requirements.lock.txt     # exact versions captured from the Colab runtime
  configs/base.yaml         # everything fixed, including the determinism policy
  configs/arm_{A,B,C}.yaml  # each overrides exactly one field: pos_encoding
  src/make_data.py  src/model.py  src/run_ablation.py  src/analyze.py
  run_all.py                # driver: loops arms x seeds, one subprocess per run
  results/runs/*.json       # one file per run attempt
  results/session_manifest.json
  results/*.csv  figures/
  docs/ANNEX_IV.md
```

`python src/run_ablation.py --config configs/arm_B.yaml --seed 1` is the only way a number is produced. A notebook, if any, calls the same functions.

### 6.2 What is held fixed (the "welded dials")

Dataset and splits, windowing, normalization, model width/depth/heads/FF, dropout, head, loss, optimizer and schedule, batch size, step count, precision, determinism policy, evaluation code, latency protocol, seed list, and run-order rule. The single line that differs between arm configs is `pos_encoding`.

### 6.3 Matched seeds

For seed `s`:
1. Seed everything with `s`, build the shared backbone.
2. Seed with a separate derived value, build the positional module.
3. Create the data-order generator from `s`.
4. Reseed with a derived value immediately before the training loop so the dropout stream is the same across arms.

The smoke test verifies that the backbone's initial weights are bitwise identical across arms for the same seed.

### 6.4 Determinism policy (D6)

**Principle.** One policy, selected once before any benchmark number is recorded, applied identically to every arm, seed, and stage (smoke excepted, see below). It is never changed after the pilot starts.

**Candidate policies.**

| Policy | Settings |
|---|---|
| **P1: full deterministic (preferred)** | Python/NumPy/Torch/CUDA seeded. `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, FP32, no DataLoader workers (or seeded workers). If fused attention backward is nondeterministic, force the math attention backend for all arms. |
| **P2: seeded, not deterministic (fallback)** | Same seeding, default kernels. Remaining nondeterminism is treated as part of the noise floor and documented. |

**Selection rule (written now, applied in the smoke stage).** The smoke stage times the same tiny training configuration under P1 and under P2. These two timing runs are the only place both policies appear, and they are not part of any results table. They are reported only as the "determinism penalty" measurement.
- Project total GPU time (smoke + pilot + main + optional extension + one full re-run allowance) from the P1 timings.
- If the projection fits within the planning ceiling (Section 9), adopt P1 for everything.
- If it does not, adopt P2 for everything, and record the measured penalty as the reason.

**Budget accounting.** All budget projections use the timings of the policy that will actually be used. Deterministic mode is expected to cost something, with the size depending on the attention path, and the measured number goes into the documentation.

**Documented limits of GPU reproducibility (to appear in the package and the Annex IV file).**
1. Bitwise-identical results are only expected on the same GPU model, driver, CUDA and PyTorch versions. Across hardware or releases, only statistical reproducibility is claimed.
2. Deterministic flags control PyTorch's operations. They do not make every library kernel deterministic, and a warning-only mode would still run nondeterministic kernels.
3. Floating-point addition is not associative, so reduction order changes results slightly, and small differences can grow over thousands of steps. The Week 7 readings list covers this (the floating-point non-associativity paper and the PyTorch reproducibility notes). I'll read both before citing any specifics, since I've only seen the course page's descriptions of them.
4. Free-tier Colab may assign a different GPU between sessions. GPU name and UUID are recorded for every run (6.6).
5. Latency is measured under the same determinism policy as training, so it describes this configuration and not a production-tuned one. This is stated wherever latency is reported.

### 6.5 Order and isolation

- One subprocess per run, so each run has a fresh CUDA context and peak-memory counter.
- One discarded warm-up run before timing starts.
- Arm order rotates by seed (seed 0: A,B,C; seed 1: B,C,A; seed 2: C,A,B), and all three arms for a seed run back to back as one **seed block**. This keeps matched pairs on the same hardware and in the same session segment where possible, and keeps GPU thermal state and shared-GPU drift from being tied to one arm.

### 6.6 Interrupted sessions: completed vs resumed experiments (D10)

**Per-run records.**
- Each run attempt writes `results/runs/{arm}_seed{s}_attempt{n}.json`, atomically (write temp file, then rename).
- A `.running` marker is created when the attempt starts and removed on completion. A marker with no result file means the attempt was interrupted. Interrupted attempts are never deleted. They are listed in the results table with status `interrupted`.
- `status` is one of `completed`, `failed`, `interrupted`. A rerun after an interruption is a new attempt number, started from scratch. There is no mid-run checkpoint resume, because runs are short and a resumed run would have a different RNG and timing history from a clean one.
- The primary tables use the completed attempt for each (arm, seed). If one has more than one completed attempt, the plan stops and asks, since that should not happen.

**Session manifest.** Every time the driver starts, it appends a segment record to `results/session_manifest.json`: segment index, start time, host name, GPU name and UUID, driver and CUDA versions, torch version, hash of the pip freeze output, git commit, and determinism policy. Each run JSON carries its segment index.

**Classification stated in the documentation and the video.**

| Outcome | Definition |
|---|---|
| **Uninterrupted** | The nine main runs all completed in one segment on one GPU instance. |
| **Resumed** | The main runs span more than one segment. The documentation lists which runs ran in which segment, why the session ended, and what changed (GPU, driver, library versions). |

**Consequences if resumed.**
- Accuracy comparisons stay valid if the determinism policy and software versions are unchanged, but a different GPU model is disclosed as a possible source of variation.
- Training-time and latency comparisons are restricted to runs within the same segment, and cross-segment time differences are not interpreted.
- The statement "all runs in a single free-tier Colab session" would no longer be accurate, so the video and documentation say how many sessions were actually used.

**Which stages count as "the session."** The assignment says the ablation runs inside a single free-tier session. I'd treat the smoke test and pilot as development sessions, disclosed in the manifest, and run the main grid plus all measurements in one session. Q5 asks you to confirm this reading.

## 7. Measurements per run (D7)

Two cost quantities are recorded separately and never described by the same word in the report. "Training duration" and "inference latency" are the two labels.

| Quantity | Definition |
|---|---|
| **Validation error (primary)** | MAPE (%) over all validation windows and the 24 horizons, in demand units, using final-step weights |
| Secondary errors | Validation MAE (kW). Test MAPE evaluated once at the end of each run, not used for any decision. |
| **Training duration** | Seconds from first to last training step, with `torch.cuda.synchronize()` at both ends. In-loop evaluation time is accumulated separately and excluded. Host-to-GPU data transfer is inside the loop and included. Data generation and model construction are excluded. All stated in the report. |
| **Inference latency** | Batch size 1, one 168-hour window per call, FP32, `torch.inference_mode()`, same determinism policy. Fixed set of 100 validation windows chosen with a fixed index seed. Warm-up 20 calls, then 200 timed calls cycling through the windows, each bracketed by `cuda.synchronize()`. Reported as p50, p95 and mean in milliseconds. Starting values, to be checked for stability in the pilot (V15). |
| **Peak GPU memory (training)** | `torch.cuda.max_memory_allocated()` after `reset_peak_memory_stats()` at the start of the run, read when training ends |
| **Peak GPU memory (inference)** | Optimizer state and gradients are deleted, cache is cleared, resident allocated memory is recorded, then `reset_peak_memory_stats()` and the latency benchmark, then the peak is read. This keeps training state out of the inference figure. |
| Context fields | Seed, arm, attempt, segment index, parameter count, steps, final train loss, policy, torch/CUDA versions, GPU name and UUID, timestamps, git commit |

Peak memory is "allocated" and not "reserved," so it excludes CUDA context and allocator cache. The same code path measures it in every run.

**Reference baselines (no training, for context).** Seasonal-naive (repeat the previous day) and same-hour-last-week, scored with the same MAPE code. If a trained arm cannot beat these, the forecasting model is not informative (V3).

## 8. Primary metric and comparison rule (D9)

**Primary metric:** validation MAPE at the final step.
**Reference arm:** A (learned absolute).
**Planned comparisons:** B vs A and C vs A. C vs B is reported descriptively. With three comparisons and three seeds, none are presented as confirmed.

**Primary evidence: matched per-seed differences.** For each comparison and each seed `s`, compute `d_s = MAPE(arm, s) − MAPE(A, s)`. A negative `d_s` means the arm has lower error than A.

Reported for each comparison:
1. The three individual `d_s`, with seed labels.
2. The mean difference `d̄`.
3. The SD of the differences, `SD_d` (n−1).
4. A leave-one-seed-out check: the mean of the two remaining differences, for each of the three omissions.
5. A paired-difference plot showing every `d_s`, and a seed-level scatter of the raw MAPE values. Min-max ranges are drawn for display only and are not used for any decision.

**Interpretation rule (fixed before results).** It uses a practical threshold δ, in MAPE percentage points.

| Verdict | Conditions (all must hold) |
|---|---|
| **Supported, tentative** | (a) all three `d_s` have the same sign; (b) \|d̄\| ≥ δ; (c) \|d̄\| ≥ 2 × SD_d; (d) every leave-one-out mean has the same sign and is ≥ δ in magnitude |
| **Direction only** | (a) and (b) hold, but (c) or (d) does not |
| **No difference detectable at this budget** | Anything else |

No verdict is called "confirmed." No p-values are reported, because three matched pairs do not support them.

**What three seeds can and can't show, stated in the report.**
- If all three `d_s` share a sign by chance alone, that has probability about 25% (2 × 0.5³) when there is no effect. Condition (a) therefore cannot carry a claim by itself.
- `SD_d` is itself estimated from three numbers and is unstable. Conditions (c) and (d) are guards, not a substitute for more seeds.
- **Detectable-difference statement.** Whatever the verdict, the report states the smallest mean difference that would have satisfied (c) given the observed `SD_d`, that is, 2 × SD_d. A null result is then written as "this design could not distinguish differences smaller than about X MAPE points," which is the form the lesson calls a finding.

**Setting δ.** δ is a practical threshold, meaning the smallest MAPE change that would matter for a forecasting release decision. It is argued in writing in PREREGISTRATION.md, informed by the pilot's observed `SD_d`, and frozen before the first main run. If the pilot shows that 2 × SD_d is already far above any δ worth caring about, the pre-registration says so, and the study is reported as a noise-floor characterization. It is not rescued by adding arms or tuning. δ is never chosen after main-run results are seen. The value of k = 2 in condition (c) is also frozen at that point.

**Optional extension (Q3).** If declared in the pre-registration, seeds 3 and 4 are added only when the Stage-1 verdict is "direction only" or "no difference." The same rule is applied to five matched seeds, and both stages are reported, labeled by stage.

## 9. Compute budget and staging

All times are planning guesses until the pilot measures them.

**Planning ceiling.** The lesson's figure is about three hours of free-tier GPU, which I'll use as the ceiling. You mentioned an overnight budget. I would not plan on one: free-tier Colab sessions can end on idle time or usage limits, the assignment asks for a single session, and I expect the study to need far less than the ceiling. Q5 covers this.

| Stage | Purpose | Runs | Est. time |
|---|---|---|---|
| Smoke | Whole pipeline on a tiny slice, assertions, determinism check, P1 vs P2 timing for the selection rule | a few tiny runs | under 10 min |
| Pilot | Per-run time and memory, convergence of all three arms, latency stability, `SD_d` estimate, δ and step count | 3 arms × seeds 900-902 (never reused) | about 20-30 min |
| Main | The study | 3 arms × seeds 0, 1, 2 = 9, plus the latency benchmark per run | about 15-35 min |
| Optional extension | Seeds 3 and 4, only if declared up front and the Stage-1 verdict calls for it | +6 | about 10-25 min |
| Retry allowance | One full re-run of any interrupted seed block | up to 3 | included in the projection |

Pilot seeds are listed in the package but never mixed into main tables. The selection rule in 6.4 compares the sum of these stages, using measured P1 timings, with the ceiling.

## 10. Items to verify before full implementation

| ID | Question | How | Pass criterion | If it fails |
|---|---|---|---|---|
| V1 | Is positional encoding "carried forward" from earlier weeks? | You confirm (Q1) | Yes | Switch to LoRA rank |
| V2 | Per-run training time, steps per second, peak memory on the Colab GPU | Pilot | Main grid fits with margin for the retry allowance | Reduce steps, series count or model size for all arms |
| V3 | Does each arm learn, and beat seasonal-naive? | Pilot | All three arms plateau and beat both baselines | Fix task or model before main runs, using pilot seeds only |
| V4 | Do P1's flags work with the attention path? Is a same-seed rerun bitwise equal? | Smoke | Identical validation MAPE, or the residual difference is measured and reported | Force math attention backend; if still nondeterministic, report as noise-floor component |
| V5 | Backbone init identical across arms for the same seed? | Smoke assertion | Bitwise equal | Fix seeding order |
| V6 | Is the positional term wildly larger or smaller than the value embedding? | Smoke: log norms | Same order of magnitude | Document; no per-arm scaling |
| V7 | Is MAPE numerically stable (demand floor, no near-zero targets)? | Smoke | Min demand well above zero in all splits | Raise the floor in the generator, before main runs |
| V8 | Do windows align so position 24k is midnight? | Unit test | Passes | Fix windowing |
| V9 | RoPE and sinusoidal correctness | Unit tests (shapes, norm preservation, RoPE relative-position property) | Pass | Fix implementation |
| V10 | Is a plausible δ separable from `SD_d` at three seeds? | Pilot | 2 × `SD_d` is not far above any δ worth caring about | Report as a noise-floor study per Section 8. No rescue by tuning. |
| V11 | Environment pin | `pip freeze` captured on the real Colab runtime | Lockfile recorded with the Colab image date | Pin what Colab provides (torch is preinstalled, not reinstalled) |
| V12 | Run-to-run timing noise on a shared free-tier GPU | Pilot repeats | Timing spread reported with every time claim | Report as measured, no stronger claims |
| V13 | Can I reach EUR-Lex to read Annex IV? | Try to fetch | Official text retrieved | You paste or upload the official text. I do not write Annex IV content from memory. |
| V14 | Size of the deterministic-mode penalty and the resulting policy | Smoke P1 vs P2 timing, then the 6.4 rule | Policy selected and frozen | Adopt P2 and document the measured penalty |
| V15 | Is batch-1 latency stable enough to compare arms? | Pilot: repeat the latency benchmark within and across runs | Spread across repeats small relative to any arm difference, or reported as inconclusive | Raise timed calls; otherwise report latency as inconclusive |
| V16 | Does inference peak memory exclude training state? | Smoke: check resident allocated memory after cleanup is about the weight size | Yes | Move the latency benchmark to a second process that loads saved weights |
| V17 | Does the session manifest capture GPU UUID and versions? | Smoke: inspect the manifest, and simulate an interrupted run | `.running` marker left, attempt 2 recorded | Fix the driver |

## 11. Annex IV documentation plan (3-5 pages)

**Verification first.** Before drafting, I will retrieve the official text of Regulation (EU) 2024/1689 (Article 11 and Annex IV, and Annex III for the classification question) and structure the document from it.

**Classification caveat.** Annex IV applies to high-risk systems. Whether a demand-forecasting tool is high-risk depends on its use, and I will not assert that from memory. The document says it is written as a rehearsal under a stated assumption, and cites the classification text I verified.

**Draft outline (to be reconciled with the official Annex IV list):** system purpose and intended use; development methodology and design rationale (the ablation); data provenance and governance (generator, parameters, seed, hash); evaluation metrics and results with matched differences and spread; limitations (including GPU reproducibility limits and session history); known and foreseeable risks with controls; validation and testing records; crosswalk table (Annex IV item → NIST AI RMF function → ISO/IEC 42001 element → artifact path).

**Rules:** every claim cites its artifact and command. Limitations are a numbered section. NIST AI RMF and ISO/IEC 42001 cross-references are only included where I can state the mapping accurately. Course readings are the primary references. I add no citations you haven't provided or I haven't read.

## 12. Rubric coverage

| Criterion (30 pts each) | Where the plan addresses it |
|---|---|
| Ablation design and control | Sections 4, 5, 6.2, 6.3. Each control has a stated reason in 4.1. |
| Seed honesty | Sections 7 and 8. All runs shown, matched differences with each individual `d_s`, leave-one-out check, a detectable-difference statement, and a null result is a valid outcome. |
| Reproducibility | Sections 6.1, 6.4-6.6, V11. Config-driven, pinned, scripted data, run order, session manifest, README verified on a fresh runtime. |
| Annex IV documentation | Section 11, V13 |
| Video | Outline maps to: the question, the controls (why these dials are welded), the result with spread, the package, and what the study cannot conclude, given equal time. Planned after results. |

### 12.1 Which parts of this plan are graded requirements

| Plan element | Tier | Basis |
|---|---|---|
| One factor, 3 settings, 3 seeds each, single Colab session | [M] | Assignment text |
| One changed factor, controls defended | [M] | Rubric (Design) |
| Self-built, scripted dataset | [M] | Assignment text |
| Means with seed-level spread; plain statement on noise | [M] | Assignment text, rubric (Seed honesty) |
| Latency and peak memory logged with validation error | [M] | Assignment text |
| Pinned deps, seeded config-driven entry point, configs, tables and plots, README, run order captured | [M] | Assignment text, rubric (Reproducibility) |
| 3-5 page Annex IV-style document, six topics, NIST and ISO cross-references | [M] | Assignment text |
| Video requirements | [M] | Assignment text |
| Pre-registration, smoke test, per-run saving | [A] | Approved by you |
| Determinism policy, matched-difference rule, no early stopping, fixed steps | [A] | Approved by you |
| Training duration vs inference latency as separate measures; memory reset per run | [A] | Approved by you |
| Resumed-session disclosure; Annex IV verified against official text | [A] | Approved by you |
| Seed-level scatter plot | [R] | The assignment requires "plots" only |
| Fresh-runtime README test | [R] | Course guidance |
| Explicit driver script for run order | [R] | The rubric requires run order to be captured. A documented command sequence also satisfies it |
| Pilot, optional seeds 3-4, leave-one-out check, session manifest, seasonal-naive baselines, crosswalk table, numbered limitations | [R] | Course guidance or my suggestion |

None of the [R] items is a grading requirement. If time or compute gets tight, they are dropped first.

## 13. Known limitations to carry into the documentation

1. Synthetic data reflects my assumptions about seasonality, not real co-op behavior.
2. One dataset realization. Seeds don't vary the data.
3. Three seeds can't estimate variance well. The statistical claims are limited accordingly.
4. No calendar features and midnight-aligned windows favor absolute encodings, by design.
5. A fixed step budget may favor an arm that converges faster. That is a property of the arm, but it limits any claim about final accuracy.
6. Small model and short context do not predict behavior at production scale.
7. A shared free-tier GPU gives noisy timing. Memory is more reliable than time, and latency is reported with its measured spread.
8. Bitwise reproducibility is not guaranteed across hardware or library versions. Statistical reproducibility is the claim (6.4).
9. Latency is measured under the deterministic policy, and may not match a production-tuned deployment.
10. If the main grid was resumed, the number of sessions and the segment of each run are stated (6.6).

## 14. Remaining question and defaults adopted

**One question genuinely needs you.**
- **Q1.** Was positional encoding covered or used in earlier weeks, beyond the Week 5 classifier's learned embedding? The assignment requires the factor to be "carried forward from earlier weeks." If it wasn't, I switch to LoRA rank and re-derive the controls, since LoRA's alpha scaling needs its own treatment.

**Defaults adopted** for questions I raised earlier. Tell me to change any of them.

| Was | Default adopted | Why |
|---|---|---|
| Q2 dataset | Synthetic seeded generator | Simplest to script and explain, permitted by the assignment, no license risk. Cost: results describe my assumed seasonality (limitation 1) |
| Q3 extension | Declare seeds 3-4 as an optional extension in the pre-registration, triggered only by a "direction only" or "no difference" Stage-1 verdict, and report both stages | Cheap, and declaring it up front prevents post-hoc seed adding. Three seeds still meet the assignment minimum |
| Q4 no-PE control | Excluded | You asked for exactly three settings |
| Q5(a) session | Main grid plus measurements = the single session. Smoke and pilot are disclosed development sessions | Matches the assignment wording most plausibly |
| Q5(b) ceiling | Three-hour planning ceiling, no overnight assumption | Free-tier sessions can end unpredictably |
| Q6 δ | Fixed number of MAPE points, argued in the pre-registration after the pilot | Simplest to state and defend |

## 15. Decision register, D1-D9 (from REQUIREMENTS.md Section 4)

| # | Decision | Status | Resolution or what blocks it |
|---|---|---|---|
| D1 | Factor | **Selected** | Positional encoding. The only open item is Q1. |
| D2 | Dataset | **Decided** | Synthetic seeded generator (default adopted). |
| D3 | Model, task, metric | Proposed | Small Transformer, 24 h ahead from 7 days, validation MAPE (MAE secondary). Numerical details confirmed by smoke and pilot. |
| D4 | Three settings | **Decided** | Learned absolute, sinusoidal, RoPE. |
| D5 | Seeds and run budget | Partly decided | Seeds 0-2 decided. Optional extension declared (default). Step count and budget feasibility need the pilot. |
| D6 | Determinism policy | **Decided (rule)** | P1 preferred, P2 fallback, chosen once in the smoke stage by the 6.4 rule. |
| D7 | Latency and memory | **Decided** | Training duration and batch-1 inference latency recorded separately, plus peak memory for each phase (Section 7). Stability of latency check in the pilot (V15). |
| D8 | Step budget, early stopping | Partly decided | No early stopping, fixed steps. Value set in the pilot, then frozen. |
| D9 | Comparison rule | **Decided (form)** | Matched differences with the Section 8 rule. δ and k frozen after the pilot, before main runs. |
| D10 | Interrupted sessions | **Decided** | Section 6.6. "Session" = main grid plus measurements (default adopted). |
| D11 | Annex IV source | Decided | Official regulation text first (Section 11, V13). |
| D12 | Repo layout | Decided | `week7_capstone/` (6.1). |

**What stands between now and building the repository.** Two things: your answer to Q1, and agreement that the numeric values for steps, δ, and k come from a pilot. The pilot needs working code, so I'd build the minimal `make_data.py`, `model.py` and `run_ablation.py` first, as the same files that later become the package, and not a separate throwaway. If you'd prefer to freeze more before any code exists, the numeric items are the only ones that can't be frozen without measuring.

## 16. Changes from v1

- **Comparison rule (D9):** replaced min-max band non-overlap with matched per-seed differences, their mean and SD, a leave-one-out check, and a detectable-difference statement. Min-max ranges are display only.
- **Determinism (D6):** added a pre-declared selection rule between P1 and P2, tied to the GPU budget, with the measured penalty documented, and listed the limits of GPU reproducibility.
- **Latency (D7):** split training duration from batch-1 inference latency, with separate peak-memory measurement for each phase.
- **Interrupted sessions (D10):** added attempt counters, interruption markers, a session manifest, a definition of uninterrupted vs resumed, and the consequences for timing comparisons and the "single session" claim.
- **Diffusion:** stated explicitly that it is not used and why.
- **Budget:** noted the overnight assumption and set a three-hour planning ceiling pending Q5.

## 17. Changes from v2

- Factor moved from "recommended" to **selected** (positional encoding), conditional on Q1 only.
- Defaults adopted for Q2-Q6 so only Q1 remains open.
- Added Section 12.1, tagging every plan element [M], [A] or [R] so optional techniques aren't mistaken for graded requirements.
- `REQUIREMENTS.md` checklist split into mandatory, adopted and recommended tiers.
