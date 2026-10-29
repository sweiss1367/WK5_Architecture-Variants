# EXPERIMENT_PLAN.md: CST 627 Week 7 Capstone

**Status:** DRAFT v19. **The nine main runs are complete and recorded (Section 32). They are awaiting the user's review.** No final interpretation has been written and the optional extension has not been started. (v18 text follows.) DRAFT v18. **The pre-registration is FROZEN (Section 31).** The main experiment (nine runs) is packaged but has not been run, and no main result exists. (v17 text follows.) DRAFT v17. The noise pilot is complete (Section 30). A δ is proposed for approval; δ and k are still unchanged in the pre-registration, which is not frozen, and the main experiment has not started. (v16 text follows.) DRAFT v16. A repeated seed 901 run reproduced the original bitwise (Section 29). Seed 902 is still outstanding, so the pilot is still incomplete. (v15 text follows.) DRAFT v15. The pilot stopped after seed 901 because of a gate bug (Section 28); the bug is fixed in the code and the pilot still needs seed 902. (v14 text follows.) DRAFT v14. The noise pilot is INCOMPLETE: seed 901 ran, seed 902 did not (Section 27). δ and k are untouched and nothing is frozen. (v13 text follows.) DRAFT v13. `train.steps: 8000` is set (user-approved). The noise-pilot code and notebook are written and tested on CPU; the pilot has not been run. (v12 text follows.) DRAFT v12. The 8000-step expansion has run (Section 25): no grid value qualified, so the pre-registered fallback selects S=8000 with an incomplete-convergence limitation. `train.steps` is still null pending the user's approval. (v11 text follows.) DRAFT v11. Run ledger reconciled and the 8000-step expansion authorised by the user (Section 24). The expansion run itself has not been executed. (v10 text follows.) DRAFT v10. Calibration stage 1 has run (Section 23): no grid value qualified, and the pre-declared expansion to S=8000 for arm A is prepared but not yet run. `train.steps` is still null. (v9 text follows.) DRAFT v9. The user approved P1 and calibration; `determinism.policy` is now `P1` in `configs/base.yaml`, the default-kernel cost profile was declined, and the calibration notebook exists but has not been run. (v8 text follows.) DRAFT v8. Two GPU smoke sessions have been run and reviewed (Section 21). The fixes from the first were verified in the second, and every smoke check passes. The policy decision (P1) is recorded in `results/determinism_decision.json`, but `determinism.policy` in `configs/base.yaml` is **still null** until the user approves it. Calibration has not started. (v6 text follows.) DRAFT v6. Step 3 code (`choose_policy.py`, `smoke_checks.py`, `check_env.py`, `calibrate.py`, Colab smoke notebook) is written. See Section 21 for what the first T4 session showed. (v5 text follows.) Build step 1 (scaffold, configs, draft PREREGISTRATION.md) and step 2 (`make_data.py`, `model.py`, `run_ablation.py`, tests) are done and tested on CPU. Nothing has been run on a GPU. Proposed as the execution plan. **The factor is locked: positional encoding** (Q1 confirmed by you: it was covered in earlier weeks). Items are tagged [M] mandatory, [A] adopted commitment, [R] recommended, using the tiers defined in `REQUIREMENTS.md` Section 5. No implementation exists. Nothing here has been run, so every number below is a planning estimate and is labeled as one.
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

### 4.3 How each arm applies position, and the differences that cannot be removed

The backbone (embedding projection, layer norms, attention, feed-forward, head, loss, optimizer, data) is shared code. The positional module is the only branch.

| | A. Learned absolute | B. Sinusoidal | C. RoPE |
|---|---|---|---|
| Where applied | Added to the input embedding once, before block 1 | Added to the input embedding once, before block 1 | Rotation applied to **queries and keys** in **every** attention layer. Values are not rotated. Nothing is added to the input embedding |
| Position information | Absolute index | Absolute index | Relative offset only |
| Parameters | 168 × 64 = 10,752 learned | 0 (fixed buffer) | 0 (fixed cos/sin tables) |
| Position indices | 0..167 | 0..167 | 0..167, rotary base 10000, head dim 16 |

**Unavoidable differences, all disclosed in the documentation:**
1. **Parameter count.** Arm A has 10,752 more parameters (about 7% of a roughly 150k-parameter model). I am not padding B and C to match, because that would add a second changed factor. The comparison is "standard implementation of each method," not "equal parameters."
2. **Where position enters.** Input-level addition (A, B) vs per-layer rotation of q and k (C). This is what distinguishes the methods, not an implementation choice I can remove.
3. **Absolute vs relative information.** C has no direct access to absolute hour of day. It must infer phase from the shape of the demand curve. This is why H1 is directional (PREREGISTRATION.md Section 2).
4. **Extra compute.** C adds rotation operations per layer. This may show up as small latency and training-time differences, and is reported as a measured cost of the method.
5. **Scale parity.** A (default embedding initialization) and B (unit-amplitude sin/cos) are added to a value embedding of unknown scale. The norm ratio is logged and checked (V6). No arm gets special scaling.

The rest of the model is the same module instance type, built with identical code, and the smoke test asserts the shared backbone's initial weights are bitwise identical across arms for each seed.

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
- **Leakage rules (also in PREREGISTRATION.md Section 9).** Adjacent windows overlap by design (context 168 h, stride 24 h), so splits are defined by target time. A window belongs to the split containing its first target hour. Every training window's target and context end before the validation start. Validation and test contexts may reach back into earlier periods, because past history is known at forecast time, but no validation or test target hour appears as a target or context hour in any training window. Per-window normalization uses only the window's own context, no scaler is fitted across splits, and the gap-filling rule uses past values only. Automated tests enforce each rule (V18, V19).
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
  src/choose_policy.py      # P1 vs P2 timing, feasibility-only selection
  src/smoke_checks.py       # V4, V5, V6, V7, V16 and the unit tests
  src/check_env.py          # capture or verify requirements.lock.txt, never installs
  src/calibrate.py          # step budget from training loss only
  colab/week7_step3_smoke.ipynb  # thin notebook: clone, call scripts, zip outputs (smoke only)
  results/smoke/            # smoke and policy-timing outputs, kept apart from calibration and main
  results/determinism_decision.json  results/env_report.json  results/calibration_decision.json
  run_all.py                # [R] optional driver: loops arms x seeds, one subprocess per run (not written yet)
  results/runs/<stage>/*.json  # one file per run attempt; stages: smoke, calibration, pilot, main, extension
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
| **P1: full deterministic (preferred)** | Python/NumPy/Torch/CUDA seeded. `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, FP32, no DataLoader workers. Attention is forced to the PyTorch math backend for every arm, unconditionally (a 168-token sequence makes this cheap, and it avoids a conditional rule). |
| **P2: seeded, not deterministic (fallback)** | Same seeding, default kernels and default attention backend selection. Remaining nondeterminism is treated as part of the noise floor and documented. |

**Identical workloads.** `choose_policy.py` times the real `run_ablation.py` script, one fresh process per run, in the smoke stage with `--timing-only`, which skips all validation and test evaluation. P1 and P2 receive the same command line apart from `--policy`: arms A, B and C, seed 0, the full 16-series dataset, batch size 128, and two step counts (200 and 600) so that fixed start-up cost and per-step cost can be separated by a two-point fit. Two repetitions, one discarded warm-up run, and the P1/P2 order alternates between repetitions so GPU clock or temperature drift is not tied to one policy.

**Recorded conditions.** GPU name, UUID and driver, CUDA runtime, PyTorch and Python versions, git commit, GPU clocks, temperature and power before and after the benchmark (`nvidia-smi`), the step counts, repetitions, order, batch size, dataset size and the latency protocol. All of it goes into `results/smoke/policy_timing/policy_timing.json` and the decision file.

**Report before selecting.** The script writes the timing file, prints the measured P1-vs-P2 difference (train step time, inference latency, projected run time), and only then applies the rule.

**Selection rule (fixed before any measurement; feasibility only).** Project the total GPU time of the remaining study from the measured P1 timings: 27 runs (3 + 4 calibration, 2 noise-pilot, 9 main, 6 extension, 3 retry allowance), each at the largest candidate step count (4000), using the slowest arm. If the projection is within the planning ceiling (3 h), select P1. Otherwise select P2. If even the selected policy exceeds the ceiling, the script prints a warning and the scope must be reduced before continuing. Accuracy plays no part.

**Recording and enforcement.** The decision is written to `results/determinism_decision.json`. `determinism.policy` in `configs/base.yaml` is then set by hand to match, and `run_ablation.py` refuses any non-smoke run when the config and the decision file disagree.

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
- Each run attempt writes `results/runs/<stage>/{arm}_seed{s}_attempt{n}.json`, atomically (write temp file, then rename).
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

### 6.7 Gates enforced in code

Each safeguard below is implemented in the repository, tested, and not left to the notebook.

| Safeguard | Enforcement |
|---|---|
| Notebook execution order cannot change the experiment | The notebook only clones the repo and calls scripts. The smoke notebook has no calibration, pilot or main cell. |
| Dependencies are not silently changed | No `pip install` anywhere. `check_env.py --auto` captures `requirements.lock.txt` from the first Colab session (never overwriting), then verifies it in later sessions and saves a full package list with a hash. Every non-smoke run is refused if the lock is missing or any pinned package or the Python minor version differs. An override flag exists, is recorded in the run file, and is for debugging only. |
| Code cannot change between runs unnoticed | Non-smoke runs are refused if tracked files differ from the committed code. Each run records the git commit. |
| Policy decided before any calibration, pilot or main run | Non-smoke runs require `results/determinism_decision.json` and a matching `determinism.policy`. |
| Full experiment cannot run before the pre-registration is frozen | `main` and `extension` runs are refused while `PREREGISTRATION.md` contains `______` placeholders. |
| Smoke, calibration and main outputs stay separate | Separate directories: `results/smoke/`, `results/runs/calibration/`, `results/runs/pilot/`, `results/runs/main/`, `results/runs/extension/`. A seed is only accepted in its own stage. |
| Validation cannot influence the step budget | `calibrate.py` reads run files through a whitelist of fields (training loss, step count, arm, seed, status). Validation and test values never enter the module. A test feeds identical training losses with adversarial validation values (NaN, ±1e9) and asserts the decision is identical. |
| Policy timing cannot use accuracy | `--timing-only` skips all evaluation, and `choose_policy.py` asserts no accuracy is present. The determinism comparison in `smoke_checks.py` runs after the decision, and its validation MAPE is never used for a decision. |

## 7. Measurements per run (D7)

Two cost quantities are recorded separately and never described by the same word in the report. "Training duration" and "inference latency" are the two labels.

| Quantity | Definition |
|---|---|
| **Validation error (primary)** | MAPE (%) over all validation windows and the 24 horizons, in demand units, using final-step weights |
| Secondary errors | Validation MAE (kW). Test MAPE evaluated once at the end of each run, not used for any decision. |
| **Training duration** | Seconds from first to last training step, with `torch.cuda.synchronize()` at both ends. There is no in-loop evaluation, so nothing is subtracted. The datasets are uploaded to the GPU once before the timed region (excluded), and the per-step batch-index transfer is included. Data generation and model construction are excluded. All stated in the report. Peak training memory therefore includes the resident dataset tensors, which are identical for all arms. |
| **Inference latency** | Batch size 1, one 168-hour window per call, FP32, `torch.inference_mode()`, same determinism policy. Fixed set of 100 validation windows chosen with a fixed index seed. Warm-up 20 calls, then 200 timed calls cycling through the windows, each bracketed by `cuda.synchronize()`. Reported as p50, p95 and mean in milliseconds. Starting values, to be checked for stability in the pilot (V15). |
| **Peak GPU memory (training)** | `torch.cuda.max_memory_allocated()` after `reset_peak_memory_stats()` at the start of the run, read when training ends | Under P1 this includes a constant of about 65 MiB of cuBLAS workspace that is identical for every arm (Section 21.3), so quote arm differences, not absolute levels, as the cost of an encoding. |
| **Peak GPU memory (inference)** | Optimizer state, gradients and the on-GPU datasets are deleted, cache is cleared, resident allocated memory is recorded, then `reset_peak_memory_stats()` and the latency benchmark, then the peak is read. This keeps training state out of the inference figure. |
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

**Setting δ.** δ is a practical threshold, meaning the smallest MAPE change that would matter for a forecasting release decision. It is argued in writing in PREREGISTRATION.md, derived from scale information only (the reference arm's MAPE level and the seasonal-naive MAPE), and frozen before the first main run. No between-arm validation difference from the pilot is used. The noise estimate for the feasibility check comes from the reference arm alone (`SD_A`, over pilot seeds 900-902), and the check can only change how the final report is framed, never δ or any setting. k = 2 is provisional and is changed only by a logged amendment before the freeze.

**Optional extension (Q3).** If declared in the pre-registration, seeds 3 and 4 are added only when the Stage-1 verdict is "direction only" or "no difference." The same rule is applied to five matched seeds, and both stages are reported, labeled by stage.

## 9. Compute budget and staging

All times are planning guesses until the pilot measures them.

**Planning ceiling.** The lesson's figure is about three hours of free-tier GPU, which I'll use as the ceiling. You mentioned an overnight budget. I would not plan on one: free-tier Colab sessions can end on idle time or usage limits, the assignment asks for a single session, and I expect the study to need far less than the ceiling. Q5 covers this.

| Stage | Purpose | Runs | Est. time |
|---|---|---|---|
| Smoke | Whole pipeline on a tiny slice, assertions, determinism check, P1 vs P2 timing for the selection rule | a few tiny runs | under 10 min |
| Calibration stage 1 | Timing and training-loss calibration, reference arm only. The step count `S` is chosen from **training loss only**. | Arm A, seed 900, `S` in {1000, 2000, 4000} = 3 short runs | about 5-15 min |
| Calibration stage 2 (expansion) | Confirm the plateau on the other arms, so the step budget is not tuned on one arm | Arms B and C, seed 900, at `S*` and the next grid value = 4 runs. Droppable, at the cost of a weaker step-budget claim | about 5-15 min |
| Noise pilot | Estimate seed-to-seed variation of the **reference arm** (`SD_A`) and run the reference-arm sanity checks. It does not compare encodings. | Arm A, seeds 901, 902 at `S*` (seed 900 reused from stage 1) = 2 runs | about 3-10 min |
| Main | The study | 3 arms × seeds 0, 1, 2 = 9, plus the latency benchmark per run | about 15-35 min |
| Optional extension | Seeds 3 and 4, only if declared up front and the Stage-1 verdict calls for it | +6 | about 10-25 min |
| Retry allowance | One full re-run of any interrupted seed block | up to 3 | included in the projection |

Pilot seeds are listed in the package but never mixed into main tables. The selection rule in 6.4 compares the sum of these stages, using measured P1 timings, with the ceiling.

## 10. Items to verify before full implementation

| ID | Question | How | Pass criterion | If it fails |
|---|---|---|---|---|
| V1 | Is positional encoding "carried forward" from earlier weeks? | You confirm (Q1) | Yes | Switch to LoRA rank |
| V2 | Per-run training time, steps per second, peak memory on the Colab GPU | Pilot | Main grid fits with margin for the retry allowance | Reduce steps, series count or model size for all arms |
| V3 | Does the reference arm learn, and beat seasonal-naive? Do all arms reach a training-loss plateau? | Noise pilot and calibration stages | Arm A beats both baselines on validation. All three arms satisfy the plateau rule on training loss. Arms B and C validation values are not inspected before the freeze | Fix task or model before main runs, using pilot seeds only. Final B and C baseline comparisons are reported, not gating |
| V4 | Do P1's flags work with the attention path? Is a same-seed rerun bitwise equal? | Smoke | Identical validation MAPE, or the residual difference is measured and reported | P1 already forces the math attention backend; if still nondeterministic, report as a noise-floor component |
| V5 | Backbone init identical across arms for the same seed? | Smoke assertion | Bitwise equal | Fix seeding order |
| V6 | Is the positional term wildly larger or smaller than the value embedding? | Smoke: log norms | Same order of magnitude | Document; no per-arm scaling |
| V7 | Is MAPE numerically stable (demand floor, no near-zero targets)? | Smoke | Min demand well above zero in all splits | Raise the floor in the generator, before main runs |
| V8 | Do windows align so position 24k is midnight? | Unit test | Passes | Fix windowing |
| V9 | RoPE and sinusoidal correctness | Unit tests (shapes, norm preservation, RoPE relative-position property) | Pass | Fix implementation |
| V10 | Is a plausible δ separable from the reference-arm noise at three seeds? | Noise pilot: compare `k × √2 × SD_A` with δ | The product is not far above any δ worth caring about | Report as a noise-floor study per Section 8. No rescue by tuning, and δ is not lowered |
| V11 | Environment pin | `pip freeze` captured on the real Colab runtime | `requirements.lock.txt` captured in the first Colab session, committed, and verified in every later session. No packages are installed or upgraded | Pin what Colab provides (torch is preinstalled, not reinstalled) |
| V12 | Run-to-run timing noise on a shared free-tier GPU | Pilot repeats | Timing spread reported with every time claim | Report as measured, no stronger claims |
| V13 | Can I reach EUR-Lex to read Annex IV? | Try to fetch | Official text retrieved | You paste or upload the official text. I do not write Annex IV content from memory. |
| V14 | Size of the deterministic-mode penalty and the resulting policy | Smoke P1 vs P2 timing, then the 6.4 rule | Policy selected by projected GPU time only, not by any accuracy figure, and frozen | Adopt P2 and document the measured penalty |
| V15 | Is batch-1 latency stable enough to compare arms? | Pilot: repeat the latency benchmark within and across runs | Spread across repeats small relative to any arm difference, or reported as inconclusive | Raise timed calls; otherwise report latency as inconclusive |
| V16 | Does inference peak memory exclude training state and datasets? | Smoke: resident allocated memory after cleanup is at most 1.5 × parameter bytes + 8 MB | Yes | Move the latency benchmark to a second process that loads saved weights |
| V18 | No train/validation/test leakage in the generated windows | Unit tests on the arrays: max train context and target time < validation start, no validation or test target hour in any training window, per-window normalization only, gap fill uses past values only | All assertions pass | Fix the windowing or generator before any pilot |
| V19 | Do the leakage tests actually catch a violation? | Deliberately build a leaking split in a test and confirm the assertions fail | Test fails on the leaking split | Fix the tests |
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

**No open questions remain.**
- **Q1 (resolved).** Positional encoding was covered in earlier weeks, so it satisfies "carried forward from earlier weeks." LoRA rank stays on file as the fallback only if something later invalidates this choice.

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
| D1 | Factor | **Decided** | Positional encoding. Q1 confirmed: covered in earlier weeks. |
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

**What stands between now and building the repository.** One thing: agreement that the numeric values for steps, δ, and k come from a pilot. The pilot needs working code, so I'd build the minimal `make_data.py`, `model.py` and `run_ablation.py` first, as the same files that later become the package, and not a separate throwaway. If you'd prefer to freeze more before any code exists, the numeric items are the only ones that can't be frozen without measuring.

## 16. Changes from v1

- **Comparison rule (D9):** replaced min-max band non-overlap with matched per-seed differences, their mean and SD, a leave-one-out check, and a detectable-difference statement. Min-max ranges are display only.
- **Determinism (D6):** added a pre-declared selection rule between P1 and P2, tied to the GPU budget, with the measured penalty documented, and listed the limits of GPU reproducibility.
- **Latency (D7):** split training duration from batch-1 inference latency, with separate peak-memory measurement for each phase.
- **Interrupted sessions (D10):** added attempt counters, interruption markers, a session manifest, a definition of uninterrupted vs resumed, and the consequences for timing comparisons and the "single session" claim.
- **Diffusion:** stated explicitly that it is not used and why.
- **Budget:** noted the overnight assumption and set a three-hour planning ceiling pending Q5.

## 18. Changes from v3

- Build step 1 completed: `configs/`, `PREREGISTRATION.md` (draft, unfrozen), `README.md` stub, folder skeleton.
- Added Section 4.3: how each arm applies position (RoPE on queries and keys in every layer), and the unavoidable differences.
- Added leakage rules and tests (V18, V19). Splits are defined by target time because adjacent windows overlap.
- Split the pilot into step calibration (training loss only) and a noise/feasibility pilot, so validation error plays no part in choosing the step count.
- δ derived from scale information only. k provisional at 2, changeable only by logged amendment before the freeze.
- Determinism policy selection made explicitly feasibility-only.

## 17. Changes from v2

- Factor moved from "recommended" to **selected** (positional encoding), conditional on Q1 only.
- Defaults adopted for Q2-Q6 so only Q1 remains open.
- Added Section 12.1, tagging every plan element [M], [A] or [R] so optional techniques aren't mistaken for graded requirements.
- `REQUIREMENTS.md` checklist split into mandatory, adopted and recommended tiers.
- Q1 resolved by the user: positional encoding is locked as the factor. No open questions remain.

## 19. Changes from v4

- **Hypotheses** (PREREGISTRATION.md Section 2): the primary hypothesis is now neutral (positional-encoding choice affects accuracy, test whether differences exceed seed noise). The RoPE disadvantage is a secondary, conjectural hypothesis, with the reasoning for it and the reasons it may be wrong stated explicitly.
- **Calibration** is staged and smaller: 3 reference-arm runs, then an optional 4-run confirmation on B and C, then 2 noise-pilot runs. Total 9 or fewer, down from 18.
- **Noise pilot** estimates reference-arm variability only (`SD_A`). Between-arm validation differences are not inspected and cannot influence δ, steps or any setting.
- **Step 2 code** written: `make_data.py`, `model.py`, `run_ablation.py`, and 34 tests. P1 now includes the math attention backend unconditionally. There is no in-loop evaluation. Datasets are uploaded to the GPU once before timing.
- **Shared initialization:** the shared backbone is built from the run seed, the positional module from a separate derived seed, and tests assert bitwise-identical shared weights across all three arms. Each run also records a hash of the shared initial weights.

## 20. Changes from v5 (Step 3)

- `choose_policy.py`: identical-workload P1/P2 timing in fresh processes, timing-only, conditions recorded, measured difference printed before the feasibility-only selection, decision file enforced against the config.
- `smoke_checks.py`: V4 (determinism, run after the decision), V5, V6, V7, V16, unit tests.
- `check_env.py` and the lock: capture once, verify afterwards, never install. Mismatch blocks non-smoke runs.
- `calibrate.py`: step budget from a whitelist of training-loss fields only, tested against adversarial validation values.
- Gates added to `run_ablation.py` (Section 6.7). The notebook is smoke-only.
- Dataset tensors are now freed before the inference-memory measurement, so inference memory is not inflated by the resident dataset.

## 21. Smoke observations, first Colab session (commit `0541b40`)

**Source.** `results/smoke/session1_0541b40/`: the executed notebook and the full output zip (decision, timing, checks, manifest, env report, package list, data hash, lock). All figures below were read from those files. Everything was one Colab VM, Tesla T4 (UUID `GPU-b4afd0de-...`), driver 580.82.07, CUDA 13.0, PyTorch 2.11.0+cu130, Python 3.13.15, numpy 2.1.3, PyYAML 6.0.3. `requirements.lock.txt` in the repo is the file the session itself captured, and the environment check reported a match.

### 21.1 Policy timing (feasibility only)
- Under P1, 12 timed runs plus the warm-up completed on the T4 with no error.
- Fitted P1 vs P2 train-step time: A 51.75 vs 38.92 ms (+33%), B 53.91 vs 43.54 (+24%), C 69.57 vs 52.91 (+32%). Batch-1 inference p50: P1 4.7, 4.7, 5.3 ms vs P2 3.7, 3.5, 4.2 ms.
- Projected total for 27 runs at 4000 steps, slowest arm: P1 2.13 h, P2 1.65 h, ceiling 3 h. The rule selected **P1** (`accuracy_used: false`, `cpu_debug: false`). Using the slowest single repetition for every arm, P1 is about 2.5 h, still under the ceiling.
- **Timing is noisy.** The same configuration varied 2-32% between repetitions, and the per-run P1 penalty at S=600 ranged from +20% to +45%. GPU temperature went from 43 C at start to 73 C at the end, which is consistent with drift. Quote the penalty as a range of about 20-45%.
- **P1 also costs memory.** Peak training memory under P1 was about 623 MiB against 382 MiB under P2 for arm B (+63%). The likely cause is the math attention backend materialising the attention matrices (not tested). It is far below the 15 GiB available.
- **Arm timing in this benchmark is confounded with position** (arms ran A, B, C every repetition). The 30-45% slowdown seen for C is not a finding about RoPE.

### 21.2 Determinism
- **P1 is bitwise reproducible on this GPU.** Across the three executions of the check script, all 12 P1 runs gave identical final weights and identical validation MAPE within each arm (A: 3 runs, 1 distinct hash; B: 6 runs, 1; C: 3 runs, 1).
- **P2 is not.** All 6 P2 runs of arm B had 6 distinct weight hashes. The validation MAPE spread across them was about 1e-7 MAPE points (8.86937034 to 8.86937045) at 100 steps. This is the size of the residual nondeterminism at that length, and it may grow with more steps. It is tiny, so the choice of P1 buys exact reproducibility and not a noise reduction. Those figures come from the determinism check and were not used for any decision.
- **Shared initialization (V5)** is identical across all arms and all 18 runs.

### 21.3 V16, the failed memory check, diagnosed
- **Data.** Resident GPU memory after cleanup was 65.627 MiB in all six runs (arms A, B, C; P1 and P2), against about 0.6 MiB of parameters. Subtracting parameters and buffers leaves **65.0075 MiB for every arm and both policies**.
- **What that number is.** 65.0075 = 64 + 1 + 0.0075. That is two 32 MiB cuBLAS workspaces (`:4096:8` is eight 4 MiB buffers; one for the forward thread and one for the autograd backward thread is the likely split) plus a 1 MiB cuBLASLt workspace plus about 7 KB of small tensors. It is **not** consistent with a leak: the on-GPU datasets would weigh 12.24 MiB, and datasets plus optimizer state about 14 MiB.
- **Status.** Strong numerical evidence, not yet a direct measurement. The old criterion (resident under 1.5 × parameter bytes + 8 MB) was mis-specified because PyTorch counts library workspaces as allocated memory. It was not loosened. V16 now compares each run with a **fresh-process baseline** that builds the model and does one forward and backward pass on a tiny batch, with no dataset and no optimizer, and passes if the run is within 1 MiB of it. A leak of any size would show as a positive difference. Result: it passed in the second session (Section 21.7).
- **Interference with measurement.** Absolute peak memory includes this constant. It is identical for every arm, so arm differences are unaffected. Peak training memory differed across arms by at most 0.3 MiB under P1 (A 623.237, B 623.153, C 622.924), and inference peak above resident was about 1.3 MiB under P1 (B 1.28, C 1.36) and 0.5 MiB under P2.

### 21.4 Other findings
- **Data generator reproduces across machines.** The Colab-generated dataset hash `65d899e60a00802062ac5c9c0dc5202d8d5ed6e6a0d64cba593e8985fc53f695` is identical to the hash generated on a different machine with a different numpy (2.1.3 on Colab, 2.5.3 locally). The dataset config hash also matches (`7c7d3b8c34`).
- **Unit tests pass on the GPU machine**, including the run gates. Leakage checks passed inside every run.
- **Session labelling.** The manifest shows two "segments" for one VM because `choose_policy.py` and `smoke_checks.py` set their own session ids. That would mislabel a single session as resumed.
- **The check script ran three times in that runtime** (arm B has 12 attempts). All attempts are kept, and the results are consistent.

### 21.5 Changes made in response (code, not yet verified on a GPU)
1. `configure_policy()`: `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set for P1 and **removed** for P2, so P2 is really default. The run file records the setting. This changes what P2 timing means, so policy timing must be re-measured.
2. Policy timing rotates arm order by repetition.
3. Scripts no longer override the session id.
4. V16 compares each run with the fresh-process baseline.
5. The notebook never reuses a stale clone.

### 21.6 Open items (from session 1; resolved in 21.7)
1. Second smoke session on the fixed commit. **Done.**
2. If V16 had not passed, reopen the leak hypothesis. **Not needed, it passed.**
3. Commit the new decision file and set `determinism.policy`. **Decision file committed; the config is deliberately left null until the user approves.**

### 21.7 Second smoke session (commit `3fca186f`), after the fixes
**Source.** `results/smoke/session2_3fca186/` (the whole output zip). A fresh runtime: Tesla T4, driver 580.82.07, PyTorch 2.11.0+cu130, a **different physical GPU** (UUID `GPU-65ec8df7-...`) with an **identical package list** (same `pip freeze` hash as session 1). The environment check verified against the lock and reported a match, so the verify path works. The manifest shows **one** segment, as intended. `cpu_debug: false`, `accuracy_used: false`.

**All checks passed:** V4 under P1 (bitwise identical), V5, V6, V7, **V16**, leakage, unit tests. V4 under P2 is informational and again not bitwise identical (about 4e-8 MAPE points apart).

**V16 resolved by direct measurement.** In a fresh process that only built the model and did one forward and backward pass on a batch of 8, resident memory was 0.626 MiB after the model build and **66.25 MiB after the first forward and backward pass**, then 65.626 MiB after cleanup. The workspace allocation therefore happens in the first matrix multiplications and stays resident, and it is not left over from data or optimizer state. Every real run was within 0.001 MiB (1 KiB) of the baseline for its policy and arm. Under default kernels (P2, workspace variable now unset) the same probe gives 17.88 MiB, and the 47.75 MiB gap to P1 matches the difference in peak training memory between the two sessions' P2 runs.

**Policy decision (same as session 1, now from the fixed code).**
| | Session 1 (`0541b40`) | Session 2 (`3fca186`) |
|---|---|---|
| P1 train step, ms (A / B / C) | 51.8 / 53.9 / 69.6 | 36.3 / 36.9 / 41.6 |
| P2 train step, ms (A / B / C) | 38.9 / 43.5 / 52.9 | 30.2 / 30.8 / 35.3 |
| Fitted P1 penalty | +24% to +33% | +18% to +20% |
| Per-run penalty range | +20% to +45% | +17% to +21% |
| Projected 27 runs at 4000 steps, P1 / P2 | 2.13 h / 1.65 h | 1.28 h / 1.09 h |
| Slowest single repetition, P1 | 2.5 h | 1.29 h |
| Selected | P1 | P1 |

Rule: feasibility only, ceiling 3 h. P1 is selected with a large margin in both sessions.

**What the two sessions say about timing**
- **Absolute speed differs by about 30% between sessions** for identical code and the same GPU model (different physical GPU and VM). Any comparison of training time or latency must be within one session. This is already the plan (Section 6.6), and it is now backed by data.
- **Within a session, timing was much steadier in session 2:** repetition 1 vs repetition 0 ratios were 1.00 to 1.08, against 1.02 to 1.32 in session 1.
- **RoPE (arm C) was still slower than A in both repetitions even though arm order rotated** (repetition 1 ended with A, not C): about 11-19% under P1 and 12-21% under P2. That is consistent with RoPE's extra operations, but it is smoke-length timing and not yet a result.

**Cost measurements depend on the policy. This matters for the report.**
- **Batch-1 inference latency:** P1 4.3-4.8 ms vs P2 (default kernels) 1.2-1.9 ms, so P1 is about 2.5 to 3.6 times slower. In session 1, P2 was 3.5-4.2 ms because it wrongly carried the workspace variable, which suggests that variable, not only the deterministic algorithms, drives the gap.
- **Peak training memory:** P1 623 MiB for all arms (spread 0.3 MiB). Under default kernels, A and B were 334.7 MiB and **C was 366.1 MiB (+31.5 MiB)**. P1 does not show that RoPE overhead.
- **Consequence.** The study runs under P1, so its latency and memory describe the deterministic configuration. They are internally consistent across arms, but P1 hides at least one real cost difference and exaggerates absolute latency. This was already limitation 9; it is now quantified.
- **Optional remedy (not built; the user decides):** a short timing-only cost profile per arm under default kernels, reported next to the P1 numbers, so cost claims are not tied to the deterministic configuration.

### 21.8 Status after two sessions
- All required smoke checks pass; the environment is locked and verified; the data hash is reproduced on three machines.
- **Ready to propose calibration.** Needed first: the user's approval, then `determinism.policy: P1` set in `configs/base.yaml` and committed.

## 22. Calibration as implemented (user-approved)

**Decisions recorded.** Policy P1 approved and set in `configs/base.yaml`. The optional default-kernel cost profile (Section 21.7) was **declined**, so latency and peak memory are reported under P1 only. This is limitation 9, now quantified: P1 batch-1 latency was 4.3-4.8 ms against 1.2-1.9 ms under default kernels in session 2, and peak training memory under P1 does not show the roughly 31 MiB extra that RoPE used under default kernels. The report must say so wherever it quotes latency or memory.

**What runs** (`colab/week7_step4_calibration.ipynb`, driver `src/run_calibration.py`, rule in `src/calibrate.py`):
1. Stage 1: arm A, seed 900, S in {1000, 2000, 4000}. Three runs.
2. Stage 2: arms B and C at S* and the next grid value, moving S* up for all arms if an arm has not plateaued. Four runs if S* passes at once.
3. Stop when the rule is satisfied, or hand to a human if no grid value qualifies (the S=8000 expansion depends on the timing projection). A cap of 12 runs applies.
4. The noise pilot (arm A, seeds 901 and 902) is **not** part of this notebook. It follows once `train.steps` is set from the reviewed calibration result.

**Enforced in code and tested**
- Step budget from training loss only (whitelisted fields).
- Arms B and C compute no validation or test metric in the calibration or pilot stages (`eval_blinded`), and the console line says "not computed".
- The calibration run key includes the step count. This fixed a bug found while testing the driver: without it, a second grid value for the same arm and seed would have been skipped as already completed.
- The calibration notebook has no pilot, main or extension cell, and the run script refuses those stages while their gates are unmet.

**Expected cost** (estimate from session 2 timings, P1, about 36 ms/step plus about 6 s start-up): stage 1 about 5 min, stage 2 about 8 min, so 13-20 minutes.

## 23. Calibration stage 1 result (session on commit `5d9843a6`, a new T4 `GPU-359219fc-...`)

**Source.** `results/runs/calibration/A_seed900_S{1000,2000,4000}_attempt1.json` and `results/calibration_session1_5d9843a/`. All three runs completed. The gates passed in every run (environment equal to the lock, clean code, policy P1 matching the decision). One manifest segment. Package list hash identical to the smoke sessions.

**Rule outcome (training loss only, arm A, seed 900).**
| S | Final training loss (last 100 steps) | Improvement to next S | Qualifies (< 2%) |
|---|---|---|---|
| 1000 | 0.05330 | 9.22% (to 2000) | no |
| 2000 | 0.04839 | 6.36% (to 4000) | no |
| 4000 | 0.04531 | not yet measured | |

**No grid value qualifies, so the pre-declared expansion trigger applies** (Section 7 of the pre-registration): add S=8000 for arm A only if the timing projection allows, otherwise use 4000 and record non-convergence. The driver stopped there on purpose. No threshold or rule was changed. Stage 2 (arms B and C) has not run, because it needs an S*.

**Timing.** About 36 ms per step under P1, as in the smoke sessions (train time 35.7, 69.8, 143.7 s for 1000, 2000, 4000 steps; peak training memory 623 MiB).

**Projection for S=8000** (pre-declared basis: remaining runs at 8000 steps, slowest arm about 15% slower than A, about 346 s per run including start-up):
- 25 runs (A@8000, stage 2 as four 8000-step runs, noise pilot 2, main 9, optional extension 6, retry allowance 3): **2.40 h**, within the 3 h ceiling.
- Without the optional extension: 19 runs, 1.82 h.
- Caution: session-to-session speed differed by about 30% earlier. With that variation the 25-run figure becomes 3.12 h, over the ceiling, and the 19-run figure 2.37 h. The extension is optional and conditional, so the commitment is the 19-run plan. The main grid alone is about 47 minutes at 8000 steps.
- **The projection allows the expansion.**

**What happens next, by the rule.** The notebook is set to `--grid 1000 2000 4000 8000`. It runs arm A at 8000 steps (one run, about 5 minutes) and then:
- If 4000 to 8000 improves by less than 2%, then S*=4000 and arms B and C are checked at 4000 and 8000 (four more runs).
- If not, there is again no qualifying value, and the pre-registered fallback applies: use the largest value (8000) and record non-convergence as a limitation. That is a human decision, and the driver stops.

**What to expect, and why it matters.** Final training loss improved by 9.2% and then 6.4% per doubling, a ratio of about 0.69. Extrapolating that ratio gives roughly 4.4% for 4000 to 8000 and about 3% beyond, so a 2% plateau is probably not reachable at an affordable budget. This is a forecast and not a result. If it holds, the final step budget is 8000 with an explicit non-convergence limitation. The 2% threshold was provisional and fixed before any run, and it will not be changed after seeing results.

**Reference-arm sanity check (not used for any decision).** Arm A's validation MAPE was 5.44, 4.97 and 4.75 at 1000, 2000 and 4000 steps, against 7.08 (previous day) and 7.17 (same hour last week) for the seasonal baselines. The model learns the task and beats both baselines at every step count (V3, reference arm). Validation error was still falling at 4000 steps, which agrees with the training loss: **at any affordable fixed budget the models are somewhat under-trained**, so the study's claims are about comparisons at a fixed 8000-step (or 4000-step) budget and not about fully converged models. This belongs in the limitations.

**A caveat on the rule itself.** It compares final training loss from separate runs whose learning-rate schedules decay over different lengths, so it measures the return from a longer budget, not convergence in the strict sense. This is stated in the documentation.

## 24. Run ledger and budget reconciliation (before the 8000-step expansion)

**Authorisation.** The user instructed the pre-declared expansion to proceed on commit `c8ab0d203cf580ce3910b7a8f529f30692529732`, with the 2% threshold, hypotheses and decision criteria unchanged, calibration from training loss only, no new thresholds or extra steps without approval, no main run and no pre-registration freeze without approval.

### 24.1 Ledger, from the files in the repository (observed)
| Category | Count | Status |
|---|---|---|
| Calibration runs completed | 3 | Arm A, seed 900, S=1000, 2000, 4000. All `completed`, attempt 1, segment 0 |
| Interrupted or failed attempts | 0 | No `.running` markers, no failed files |
| Noise-pilot runs | 0 | Not started |
| Main runs | 0 of 9 required | Not started, gate closed (pre-registration unfrozen) |
| Extension runs | 0 of 6 optional | Not started |
| Smoke runs | 74 (43 in session 1, 31 in session 2) | Development sessions. Excluded from the study budget |

**Manifest.** `results/session_manifest.json` holds 1 segment (calibration session 1, commit `5d9843a6`). The smoke manifests hold 2 segments for session 1 (the session-labelling bug, since fixed) and 1 for session 2. The 8000-step run will be a new session and a new segment of this manifest, so calibration will be reported as spanning more than one session. That is acceptable for calibration. The single-session claim applies to the main experiment.

### 24.2 Remaining runs, by what the rule calls for
| Category | Runs | Required? |
|---|---|---|
| Calibration: arm A at 8000 steps | 1 | Yes (the pre-declared expansion) |
| Calibration stage 2: arms B and C at 4000 and 8000 | 4 | **Only if** 4000 to 8000 improves by less than 2% |
| Noise pilot: arm A, seeds 901 and 902 at the selected S | 2 | Yes |
| Main: 3 arms x 3 seeds at the selected S | 9 | Yes |
| Retry allowance (reserve, not planned work) | up to 3 | Reserve |
| Extension: seeds 3 and 4, 3 arms | 6 | **Optional**, only if declared and triggered |

### 24.3 Projection (estimate, not measured). Per-run time: start-up about 6 s plus 36.9 ms per step (measured on arm A) times an arm factor of 1.00, 1.02, 1.15 for A, B, C (from smoke session 2)
| | Outcome 1: 4000 to 8000 improves < 2%, so S=4000 after stage 2 | Outcome 2: improves >= 2%, so fallback S=8000, no stage 2 |
|---|---|---|
| A@8000 (certain) | 5.0 min | 5.0 min |
| Stage 2 (B, C at 4000 and 8000) | 16.4 min | 0 |
| Noise pilot (2 runs) | 5.1 min | 10.0 min |
| Main (9 runs) | 24.3 min | 47.6 min |
| Retry reserve (3 runs, slowest arm) | 8.8 min | 17.2 min |
| **Required total** | **0.99 h** | **1.33 h** |
| With a 30% slower session | 1.29 h | 1.73 h |
| Plus optional extension | 1.26 h (1.64 h at +30%) | 1.86 h (2.42 h at +30%) |

**All figures stay within the three-hour ceiling, including the optional extension and a 30% slower session (maximum 2.42 h).** The earlier figures in Section 23 (1.82 h, 2.40 h, 3.12 h) were a cruder upper bound: they priced every remaining run at the slowest arm and counted stage 2 as 8000-step runs, which cannot happen together. The arm-by-arm figures above supersede them. The main grid alone is 24 to 48 minutes. The extension is not recommended until the three-seed result and the remaining budget are reviewed.

### 24.4 How the pre-registered rule maps to the outcome (one point needs your confirmation)
Rule (PREREGISTRATION.md Section 7): S* is the **smallest S whose next grid value improves final training loss by less than 2%**.
- **4000 to 8000 improves by less than 2%** gives S*=4000 (the loss has plateaued by 4000). Stage 2 then checks arms B and C at 4000 and 8000. If both also pass, **the selected S is 4000, and no non-convergence limitation is needed.** If B or C fails, S moves up to 8000 for all arms and the grid is exhausted, which triggers the fallback below.
- **4000 to 8000 improves by 2% or more** means no grid value qualifies. The pre-registered fallback applies: **the selected S is the largest value, 8000, and incomplete convergence is recorded as a limitation.** Stage 2 is not called for. No further step count (for example 16000) is added without approval.

**Your instruction listed these two branches the other way round** (below 2% selects 8000 with a convergence limitation; 2% or more follows the stopping procedure). I have followed the pre-registration, because you also asked for the existing rules to apply unchanged, and because selecting 8000 with a non-convergence limitation after a plateau would contradict itself. If you intended to change the rule, that would be an amendment to Section 7 and should be logged before the result is seen. Please say so.

### 24.5 What will be reported when the results arrive
Observed: the 4000 to 8000 improvement for arm A, the per-run timing, memory, environment and provenance, and (if called for) stage-2 improvements for B and C. Selected S and its basis. Estimated: the remaining runtime. Limitations: incomplete convergence (if the fallback applies), fixed-budget conditionality, P1-only latency and memory, and the learning-rate schedule caveat. Arms B and C compute no validation or test metric in calibration.

### 24.6 One stale message to expect
The run on `c8ab0d2` uses the earlier `calibrate.py`. If 4000 to 8000 improves by 2% or more, its last lines may still print "expansion trigger: add S=8000...", which no longer applies because 8000 is already in the grid. Ignore it. The decision is recomputed from the run files with the updated `calibrate.py` (reporting fields only; the rule is unchanged).

## 25. Calibration result: the 8000-step expansion (observed), selected step count, and what follows

### 25.1 Observed (from the run files in `results/runs/calibration/`, arm A, seed 900, training loss only)
| S | Final training loss (last 100 steps) | Improvement to next S | Below 2%? | Train time | Peak training memory |
|---|---|---|---|---|---|
| 1000 | 0.05330 | 9.22% | no | 35.7 s | 623.2 MiB |
| 2000 | 0.04839 | 6.36% | no | 69.8 s | 623.2 MiB |
| 4000 | 0.04531 | **5.12%** (to 8000) | **no** | 143.7 s | 623.3 MiB |
| 8000 | 0.04299 | not measured | | 291.2 s | 623.3 MiB |

- **The 4000 to 8000 improvement is 5.12%, above the 2% threshold.** No grid value qualifies.
- **Selected step count: S = 8000**, by the pre-registered fallback (use the largest grid value and record non-convergence). This follows the rule as written in PREREGISTRATION.md Section 7. Stage 2 (arms B and C) is not called for, so it was not run. The 2% threshold, the grid rule and the hypotheses were not changed. No value beyond 8000 is added without approval.
- Training cost was steady at about 36.4 ms per step (27.5 steps per second). All four runs used the same dataset (hash `65d899e6...`) and the same shared initial weights. The loss curves in the run files are complete (80 blocks of 100 steps for the 8000-step run).
- The earlier forecast (4 to 5% for this step) was close; this is the measured value.

### 25.2 Provenance, including one deviation
- Calibration ran in **two sessions on two physical T4 GPUs** (manifest segments 0 and 1; same package list hash `b506fea2eb`, environment matched the lock in both). The 8000-step run is in segment 1 on GPU `...ec068e0d`. Calibration is therefore reported as spanning more than one session. Timing from segment 1 is not compared with timing from segment 0 except as context.
- **Deviation from the instruction.** The 8000-step run recorded commit `198aad88`, the branch head at that time, **not** the pinned `c8ab0d20`. The most likely reason is that the notebook cloned from GitHub instead of using the uploaded zip. I compared the two commits: they differ only in `calibrate.py` (decision reporting fields), its tests, the plan and a stored notebook. **No file that executes a run changed** (`run_ablation.py`, `model.py`, `make_data.py`, `run_calibration.py`, `check_env.py`, the configs, the lock and the decision file are identical). The run is functionally what `c8ab0d20` would have produced. The three earlier runs in the returned zip are byte-identical to the committed first-session files.
- The decision file in `results/calibration_decision.json` was recomputed from the run files with the current `calibrate.py`. The copy printed by the session is kept in `results/calibration_session2_198aad8/`.

### 25.3 Limitations to carry into the documentation
1. **Incomplete convergence.** Final training loss was still improving by 5.12% when the budget doubled from 4000 to 8000 steps. Models at 8000 steps are not converged, and comparisons between arms are conditional on a fixed 8000-step budget. An arm that learns faster is favoured at a fixed budget, which is a property of the arm but limits claims about converged accuracy.
2. **The rule measures the return from a longer budget, not strict convergence.** Each run uses a learning-rate schedule that decays over its own length.
3. **The step count rests on one arm, one seed.** The decision uses arm A at seed 900 only. Stage 2 was not triggered, so arms B and C have not been run at any step count in calibration. Their convergence behaviour will first be visible in the training curves of the main runs, and it will be reported there.
4. **No uncertainty estimate for the 5.12%.** The seed-to-seed variation will first be measured in the noise pilot.
5. **Cost measurements are under P1 only** (limitation 9 and Section 22). The default-kernel profile was declined.
6. **Two sessions and two GPUs in calibration**, with the commit deviation above.
7. **Context, not an input to any decision.** Arm A's validation MAPE (the reference arm, allowed to be seen) was 5.438, 4.974, 4.748 and 4.674 at 1000, 2000, 4000 and 8000 steps, against 7.081 and 7.171 for the seasonal baselines. The validation gain from 4000 to 8000 steps (1.6%) is much smaller than the training-loss gain (5.12%). That is worth noting as a possible sign that further training mostly lowers training loss. It was not used to select the step count and no threshold was re-evaluated against it.

### 25.4 Remaining workload at S = 8000 (estimates, not measurements)
Basis: measured 36.4 ms per step on arm A, about 6 s start-up, arm factors 1.00, 1.02, 1.15 for A, B, C from smoke session 2.
| Item | Runs | Estimate | Required? |
|---|---|---|---|
| Noise pilot (arm A, seeds 901 and 902) | 2 | 9.9 min | Yes |
| Main (3 arms x 3 seeds) | 9 | 47.0 min | **Yes** |
| Retry reserve | up to 3 | 17.0 min | Reserve only |
| Optional extension (seeds 3 and 4) | 6 | 31.3 min | **No**, not recommended before the three-seed result is reviewed |

- Required work (pilot plus main): **0.95 h**. With the retry reserve: **1.23 h** (1.60 h if the session runs 30% slower).
- With the optional extension: 1.75 h (2.28 h at +30%).
- **All of these are within the three-hour planning ceiling.** Per-run times at S = 8000 are about 297 s (A), 302 s (B) and 340 s (C).

### 25.5 Next required decisions (none taken yet)
1. Approve setting `train.steps: 8000` in `configs/base.yaml` and committing it. Until then the pilot and main stages are blocked by design.
2. Approve the noise pilot (arm A, seeds 901 and 902, about 10 minutes). It estimates seed-to-seed variability of the reference arm (`SD_A`), and no between-arm validation number is produced.
3. After the pilot: write δ (from scale information only) and confirm k, then review and freeze the pre-registration. The main experiment starts only after that and after approval.

## 26. Noise pilot (user-approved), controls and what will be reported

**Approved and set.** `train.steps: 8000` in `configs/base.yaml`. The step count is selected by the pre-registered fallback because the rule requires it (4000 to 8000 improved training loss by 5.12%, above the 2% threshold), **not because the model has converged**. The non-convergence limitation stays visible in the final report. Both commit identifiers are kept in the provenance record: `c8ab0d20` (pinned) and `198aad88` (the commit the 8000-step run actually used).

**Controls, and where each is enforced**
| Control | Enforcement |
|---|---|
| Existing dataset, fixed 8000-step budget | Same `configs/base.yaml`. The summary refuses to treat seed 900 as a pilot seed unless its merged-config hash equals the pilot runs' (it does today: `7d63c8ef26b1f99d`) |
| Only arm A, seeds 901 and 902 | `run_pilot.py` runs `arm_A.yaml` only and refuses any other seed. `analyze_pilot.py` raises if any file is not arm A. Tests cover both |
| Seed-level variation only, no arm comparison | The summary reports one arm. `arms_compared: false` is written into it |
| δ and k unchanged | The script sets neither. It writes `delta_set: false`. k stays provisionally 2 and appears only as scale information |
| Test set unseen | Test metrics are not in the loaded fields, and a test checks they cannot be read |
| Pre-registration not frozen, no main run | `main` and `extension` stay blocked while placeholders remain. The pilot notebook has no such cell |
| Record GPU time, memory, environment, deviations | Every run file stores them. The notebook stops if the commit differs from the one given, and prints whether the code came from the zip or GitHub |

**What the pilot gives, and its limits.** Three reference-arm seeds (900 from calibration, 901 and 902 from the pilot) give a first estimate of seed-to-seed variation in validation MAPE. With three seeds the sample standard deviation is rough: its 95% interval runs from about 0.52 to 6.3 times the estimate. It informs how the results are interpreted. It is **not** a guarantee of statistical significance, and a later three-seed comparison is not made safe by it. Seed 900 comes from a different session and physical GPU than 901 and 902, so any difference includes cross-session variation (bitwise reproducibility across GPUs is not claimed). The pilot arm is the reference arm only, so its spread may not equal the other arms'.

**Reported after the run, for the review you asked for.** Observed validation MAPE per seed, mean, SD (n-1), range, the SD interval, the seasonal baselines and the gap to them (scale information), GPU time, peak memory, environment and any deviations. Then a proposed δ derived from scale information only, and the exact decision rule restated, for approval before the nine-run experiment. Nothing is frozen until then.

## 27. Noise pilot, first session (commit `f9437a57`): incomplete, seed 901 only

**Source.** The returned `week7_pilot_outputs.zip`. It holds one pilot run, `results/runs/pilot/A_seed901_attempt1.json`, plus the environment report and the manifest. It does **not** hold seed 902, a failed record for seed 902, an interrupted-run marker, or `pilot_summary.json`. The run and manifest are now in the repository (`results/runs/pilot/`, `results/session_manifest.json`, `results/pilot_session1_f9437a5/`).

**What is known about the missing run (observed, not explained).**
- Seed 901 started 14:16:12 and finished 14:21:27 UTC. The zip was made about a minute later.
- There is no `.running` marker for seed 902. A run that had started and was then killed would normally leave one, so seed 902 most likely **never began**. The likely causes are an interruption of the driver cell right after the first run, or the session ending between the two runs. That is an inference, and the cell's own output has not been seen.
- No result was lost or overwritten. The driver resumes: finished runs are skipped, so re-running it runs seed 902 only.

**Seed 901 run: provenance (observed).** Stage `pilot`, arm A, 8000 steps, policy P1, workspace variable `:4096:8`, commit `f9437a57` (the pinned one; the notebook's commit guard worked), tracked files clean, all gates passed, environment equal to the lock, package list hash unchanged (`b506fea2eb`). A third session and a third physical T4 (`...caf79b7d`, manifest segment 2). Dataset and merged-config hashes identical to seed 900's (`65d899e6...`, `7d63c8ef26b1f99d`), so seed 900 remains a valid reused pilot seed. Training 312.4 s (seed 900: 291.2 s, +7.3%, a different GPU), peak training memory 623.3 MiB, resident 65.63 MiB, batch-1 latency 4.18 ms.

**Preliminary observation, two seeds only (arm A, validation MAPE; not a result).**
| Seed | Source | Val MAPE % | Val MAE kW | Final train loss |
|---|---|---|---|---|
| 900 | calibration, other session and GPU | 4.67367 | 53.221 | 0.04299 |
| 901 | pilot | 4.67434 | 53.134 | 0.04371 |

The two MAPE values differ by **0.0007 points (0.01%)**. Final training loss differs by 1.7%. Two values cannot estimate a standard deviation. They may be close by chance, so this does not show that seed variation is small, and no δ or interval is proposed from it. The seasonal baselines are 7.08 and 7.17, so arm A is about 2.4 points (34%) better than the better baseline (scale information only).

**Still required before the review you asked for:** seed 902, then the three-seed summary (mean, SD with its interval, range), a proposed δ from scale information only, and the exact decision rule. δ and k remain unchanged, the pre-registration is not frozen, and no main run has started.

## 28. Why seed 902 did not run: a gate bug, found and fixed (commit after `d8a5b09`)

**Observed.** On re-running cell 4 the driver printed `ERROR: tracked files differ from the committed code` for seed 901 and stopped, so seed 902 never started. That also explains the first attempt: seed 901 finished, and the next run was refused before it could write a marker or a failure record, which is why the zip held neither.

**Cause.** `results/session_manifest.json` is a tracked file (it was committed with the calibration sessions), and every run appends a segment to it. After the first pilot run the manifest differed from the commit, so the dirty-tree gate treated a normal run output as a code change and blocked every later run. Two separate defects combined:
1. The dirty check counted generated output folders as code.
2. The gates ran before the "already completed" check, so a resume failed even on a run with nothing to execute.

**Why it matters beyond the pilot.** The same bug would have blocked the main grid after its first run. Calibration was unaffected because the manifest was not tracked then, and the 8000-step run was a single run.

**Fix (tested).**
- `code_is_dirty()` ignores `results/` and `data/`. Changes to source, configs and `PREREGISTRATION.md` still block a run. Tests: a manifest append, a regenerated hash file and an untracked file leave the code clean; `src/`, `configs/` and the pre-registration each block.
- An already completed run is skipped before the gates, and a new run is still refused by them (tested).
- The pilot notebook's gate cell uses the same exclusion.

**Provenance.** Seed 902 will run on the same code as seed 901 (`f9437a57`) by restoring the manifest in the live session, so the whole pilot uses one commit. The fix applies to later sessions, including the main experiment, which must use a commit that contains it.

## 29. A repeated seed 901 run (observed): bitwise reproducibility at full length

**Source.** The returned zip from the old Colab runtime, stored as `results/pilot_repeat_901_f9437a5/A_seed901_REPEAT_attempt1.json`. It is a **repeat of seed 901**, produced when the repository was re-cloned in that runtime (so the original result was not present and seed 901 ran again before the gate bug stopped seed 902). It is **not a new pilot seed** and is kept out of `results/runs/pilot/`, so the pilot analysis cannot count it. It contains no seed 902, so the pilot is still incomplete.

**Comparison with the original seed 901 run** (same physical T4 `...1bf1caf79b7d`, same commit `f9437a57`, same merged config `7d63c8ef26b1f99d`, same dataset `65d899e6...`; started about 10 minutes apart):
| | Original (14:16) | Repeat (14:27) |
|---|---|---|
| Final weights hash | identical | identical |
| Validation MAPE | 4.674338500805252 | 4.674338500805252 |
| Validation MAE (kW) | 53.13385939248752 | 53.13385939248752 |
| Final training loss | 0.04370963200926781 | 0.04370963200926781 |
| Full training-loss curve (80 blocks) | identical | identical |
| Training time | 312.4 s | 310.0 s (-0.8%) |
| Batch-1 latency p50 | 4.183 ms | 4.459 ms |

**What this shows.** Under policy P1, on one GPU and one commit, a complete 8000-step run is **exactly** reproducible: the same seed gives the same weights, the same loss curve and the same validation MAPE, to every digit. This extends the 100-step determinism check from the smoke sessions to a full-length run. It means that any spread across seeds is genuine seed-to-seed variation (initialisation, data order, dropout), with no nondeterministic noise on top of it, at least on one GPU.

**What it does not show.** It was one GPU. Reproducibility across different physical GPUs, drivers or library versions is not claimed. Seed 900's calibration run came from another GPU, so any difference between it and seeds 901 and 902 includes that. Timing and latency are not reproducible (0.8% and 6% apart here) and must be quoted with their spread.

## 30. Noise pilot result and review package (for approval before the nine-run experiment)

### 30.1 Observed (arm A only, 8000 steps, policy P1; `results/pilot_summary.json`)
| Seed | Source | Val MAPE % | Final train loss | Train time | Physical GPU | Commit |
|---|---|---|---|---|---|---|
| 900 | calibration (reused) | 4.67367 | 0.04299 | 291.2 s | `...ec068e0d` | `198aad88` |
| 901 | pilot | 4.67434 | 0.04371 | 312.4 s | `...caf79b7d` | `f9437a57` |
| 902 | pilot | 4.66130 | 0.04362 | 325.9 s | `...cadf781b` | `03d5e16d` |

- Mean **4.66977**, SD (n-1) **0.00734**, range **0.01304** (0.28% of the mean). 95% interval for the true SD from three seeds: **0.0038 to 0.0461** (about 0.52 to 6.3 times the estimate).
- Seasonal baselines (validation MAPE): 7.081 (previous day), 7.171 (same hour last week). Arm A is **2.411 points (34.1%)** better than the better one (scale information).
- Peak training memory 623.3 MiB in every run. Training time 291 to 326 s (a 12% spread across sessions). All gates passed; environment equal to the lock; same package list hash in every session. Test metrics were not read and no arms were compared (`arms_compared: false`, `test_metrics_read: false`, `delta_set: false`).

### 30.2 Provenance and deviations (all recorded)
- **Three seeds, three commits, three physical GPUs, three sessions.** I diffed the commits: `model.py`, `make_data.py`, the arm configs, the dataset and the training logic are identical across them. The differences are the gate logic in `run_ablation.py` (the dirty-tree check and the order of the "already completed" skip) and `base.yaml` `steps`, which seed 900 had as an override with an identical merged-config hash (`7d63c8ef26b1f99d`). Dataset hash `65d899e6...` in all three.
- Seed 901 also ran twice on one GPU and commit. The repeat was bitwise identical (Section 29) and is not counted as a seed.
- The earlier gate bug and the failed attempts are documented in Section 28. No result was lost or altered.

### 30.3 How far to trust the spread
1. **Three seeds is a rough estimate.** The interval above is wide. A sample SD from three values can easily be several times too small or too large. It informs interpretation. It does **not** guarantee statistical significance, and it does not make a three-seed comparison safe.
2. **Seed and GPU are confounded in the pilot.** Each seed ran on a different physical T4. Same-GPU reproducibility is exact (Section 29). Reproducibility across physical GPUs was **not tested**, so the 0.0073 spread may include a GPU component. In the main experiment all nine runs should be in one session on one GPU, so the matched differences are not affected by this.
3. **Reference arm only.** Arms B and C may be noisier or quieter. Their spread is unknown until the main runs.
4. **Not converged.** All values are at a fixed 8000-step budget (limitation from Section 25).

### 30.4 Proposed δ (for the user's decision; nothing is set)
Rule from the pre-registration: δ is the smallest mean paired validation-MAPE difference that would be worth acting on in a forecasting release decision, a multiple of 0.05 points, argued from **scale information only** (arm A's MAPE level and the seasonal baselines), never from between-arm differences or from `SD_d`.

Disclosure: the SD was visible to me before this proposal was written, so I list candidates by their scale first and report the feasibility numbers separately. The user should choose by practical meaning.

| Candidate δ | Share of arm A's MAPE (4.670) | Share of its advantage over the better baseline (2.411) |
|---|---|---|
| 0.05 | 1.07% | 2.07% |
| **0.10 (recommended)** | **2.14%** | **4.15%** |
| 0.25 | 5.35% | 10.37% |

**Recommendation: δ = 0.10 MAPE points.** Argument: a change of 0.10 points is about 2% of the reference model's error and about 4% of the value that the model adds over a trivial seasonal forecast. Smaller than that, a difference is too small to justify changing the architecture. Larger than that (0.25, a 5% change in error) would ignore differences that are still material relative to the model's advantage. This argument uses no between-arm result and no variance.

**Feasibility check (does not change δ; reported as the pre-registration requires).** The pre-registration compares `k x sqrt(2) x SD_A` with δ, with k = 2:
| | `k x sqrt(2) x SD` | δ = 0.05 | δ = 0.10 | δ = 0.25 |
|---|---|---|---|---|
| At the estimate (SD 0.0073) | 0.0208 | 2.4x larger | 4.8x larger | 12x larger |
| At the upper 95% bound (SD 0.0461) | 0.1305 | 0.4x (not separable) | 0.8x (not separable) | 1.9x larger |
If the true seed spread is near the estimate, δ = 0.10 is comfortably detectable. If it is near the upper end of the interval, δ = 0.10 is not. This is the honest uncertainty from three seeds. If the main-run `SD_d` turns out large, the report will say the design cannot separate differences of size δ, and will state the smallest difference it could separate. δ is not lowered afterwards.

### 30.5 The exact decision rule (unchanged from the pre-registration; k stays 2)
For comparison arm X (B or C) and seed s: `d_s = MAPE(X, s) - MAPE(A, s)`; negative means X has lower error. Reported for each comparison: all three `d_s`, the mean `d_bar`, `SD_d` (n-1) and the three leave-one-out means.
| Verdict | Conditions (all must hold) |
|---|---|
| **Supported, tentative** | (a) all `d_s` share a sign; (b) `|d_bar| >= delta`; (c) `|d_bar| >= k x SD_d`; (d) every leave-one-out mean has the same sign and magnitude `>= delta` |
| **Direction only** | (a) and (b) hold, but (c) or (d) does not |
| **No difference detectable at this budget** | Anything else |
No verdict is called "confirmed", no p-values are reported, and every report states `k x SD_d`, the smallest mean difference that would have satisfied (c). Chance alone gives three same-sign differences 25% of the time (2 x 0.5^3), so (a) cannot carry a claim by itself. The primary hypothesis is neutral (the encoding affects accuracy); the RoPE-worse conjecture is secondary and is reported plainly if it fails or reverses. The seed list for the main experiment is 0, 1, 2, matched across arms (identical shared initial weights, data order and dropout stream per seed).

### 30.6 Remaining workload (estimates; basis: measured 8000-step train times 291 to 326 s, mean 310 s, plus about 6 s start-up and arm factors 1.00, 1.02, 1.15)
| Item | Runs | Estimate | Required? |
|---|---|---|---|
| Main: 3 arms x 3 seeds | 9 | 49.9 min | **Yes** |
| Retry reserve | up to 3 | 18.1 min | Reserve only |
| Optional extension (seeds 3 and 4) | 6 | 33.3 min | No; not recommended before the three-seed result is reviewed |
Required including reserve: **1.13 h** (1.47 h in a 30% slower session). Main alone 0.83 h. With the extension 1.69 h (2.19 h at +30%). All are within the three-hour ceiling.

### 30.7 Decisions needed (none taken)
1. Choose δ (recommended 0.10 MAPE points; 0.05 or 0.25 are the alternatives) and confirm k = 2.
2. Approve freezing the pre-registration with those values. I would then fill the placeholders (δ, its argument, the freeze commit and time) and block any later edit except through the amendment log.
3. Approve the nine-run main experiment, in one session, on the frozen commit, with the optional extension left undeclared unless you choose otherwise.

## 31. The freeze (user-approved 2026-10-23) and the main experiment package

### 31.1 What was approved
δ = 0.10 MAPE points and k = 2, with the rationale and the exact decision rule; the neutral primary hypothesis preserved; the nine-run main experiment (3 arms x seeds 0, 1, 2) on the frozen commit in one GPU session, 8000 steps, policy P1; all three arms, including A, run under the same frozen protocol; no optional seeds without a review of the first nine results and the remaining budget; the pilot's cross-GPU confound documented. The approval covers the current design and **not** any revision of the decision criteria after the main results are seen.

### 31.2 Clarifications written into the frozen pre-registration (the numeric rule is unchanged)
- Lower MAPE is better, so an improvement over arm A is a **negative** matched difference. Conditions (b), (c), (d) use the **absolute** mean difference, and the **sign** says which arm is better and is reported next to the verdict.
- The 2 x SD condition and the leave-one-out checks are **exploratory screening criteria for a three-seed study, not significance tests**. The label stays "supported, tentative" and is never "confirmed".
- The pilot's arm A results (seeds 900 to 902) are **not substituted** for the main experiment's arm A.
- The optional extension needs the user's explicit approval after review, and the run script refuses it without an approval file.

### 31.3 Freeze record (verifiable)
| | |
|---|---|
| Frozen content (commit 1) | `09cf317e19d9fb94a5c90e3f5149b2a218805e76` |
| Freeze record (commit 2) | `8c561c196589cc4c7ee7789dced7c95723e805c1`, adds only `results/freeze_record.json` |
| Timestamp | 2026-10-23 15:33 UTC |
| Check | `python src/freeze.py --verify` (13 files) |

Frozen files (SHA-256 prefixes): `PREREGISTRATION.md` cfecb336, `configs/base.yaml` 2ff8b2b0 (contains δ and k), `configs/arm_A/B/C.yaml`, `requirements.lock.txt`, `src/run_ablation.py`, `src/model.py`, `src/make_data.py`, `src/analyze_main.py`, `src/run_main.py`, `src/check_env.py`, `src/freeze.py`. Notebooks, tests and docs are not frozen.

### 31.4 Order of events, so the rule cannot have been shaped by results
1. The rule was stated in the pre-registration (Sections 4 and 5) before the pilot.
2. The pilot ran on arm A only. δ was proposed from scale information and approved.
3. `analyze_main.py` (the rule as code) and its 19 tests were written **before** the freeze and are frozen.
4. All 124 tests passed on the frozen code (11 minutes) before the freeze record was written.
5. The freeze record was written. **No main-experiment result exists.**

### 31.5 Gates and rules enforced in code (tested)
- Main and extension runs are refused if any frozen file differs from its recorded hash, if the freeze commit is not an ancestor of the running code, or if the pre-registration still has an unfilled placeholder.
- The extension stage needs `results/extension_approval.json`, which only the user creates. The driver cannot start it.
- The driver refuses to continue the main experiment in a second session unless told to, and such a run is reported as resumed. Every attempt (completed, failed, interrupted) is listed by the analysis.
- Arm order rotates by seed (seed 0: A,B,C; seed 1: B,C,A; seed 2: C,A,B) with seed blocks kept contiguous.
- The notebook pins the freeze commit and stops if it cannot check it out.

### 31.6 Remaining budget (estimates, not measurements)
Main experiment alone about 50 minutes (0.83 h); with the retry reserve 1.13 h (1.47 h in a 30% slower session). The optional extension is not planned. All figures are within the three-hour ceiling.

### 31.7 What the main run reports
Per run: validation MAPE and MAE, test MAPE (secondary, never part of the verdict), final training loss, end-to-end training time, batch-1 inference latency, peak training and inference memory, GPU, session and commit. Per comparison: the three matched differences, mean, SD (n-1), leave-one-out means, the four conditions, `k x SD_d`, the verdict and the direction. All measured under P1, so latency and memory describe the deterministic configuration.

### 31.8 Limitations that carry into the report
Not converged at 8000 steps; synthetic data; one dataset realisation; three seeds; a single GPU session, so findings apply mainly to the tested environment; cross-GPU reproducibility not tested; reference-arm noise estimate only; cost numbers under P1 only.

## 32. Main experiment: results for review (observed), before any final interpretation

**Source.** `results/runs/main/` (nine files), `results/main_summary.json`, `results/main_runs.csv`, `results/paired_differences.csv`, `results/main_session_8c561c1/`. Produced by `python src/run_main.py` on the freeze commit `8c561c19`, in one Colab session. The analysis was recomputed independently from the raw run files (a separate implementation of the frozen rule), and both gave the same verdicts.

### 32.1 Ledger and provenance (observed)
- **9 completed, 0 failed, 0 interrupted, no unfinished markers.** Classification from the analysis: **uninterrupted** (one segment, one GPU).
- One session (manifest segment 4), one physical T4 (`...0e2060a0`), driver 580.82.07, PyTorch 2.11.0+cu130, package list hash `b506fea2eb` (unchanged since the first smoke session), environment equal to the lock. All gates passed in every run, tracked code clean, commit `8c561c19` in every run.
- Policy P1, 8000 steps, the existing dataset (`65d899e6...`) in all nine. Training-configuration hash per arm: A `7d63c8ef26b1f99d` (identical to the pilot and calibration), B `41a938ca...` and C `0f14c780...` (these differ from A only in the positional-encoding setting, as expected).
- For each seed, the shared initial weights are identical across A, B and C, and the final weights all differ. Parameter counts: A 162,520 (including 10,752 positional), B and C 151,768.
- Order: seed blocks with the arm order rotated (seed 0: A,B,C; seed 1: B,C,A; seed 2: C,A,B). Each arm ran once in each position. 15:40:46 to 16:30:05 UTC, **49.3 minutes** (estimate: about 50).

### 32.2 Validation MAPE (%), the primary metric (observed)
| Arm | Seed 0 | Seed 1 | Seed 2 | Mean | SD (n-1) |
|---|---|---|---|---|---|
| A learned | 4.63441 | 4.70013 | 4.66617 | 4.66690 | 0.03286 |
| B sinusoidal | 4.66283 | 4.63242 | 4.67730 | 4.65752 | 0.02291 |
| C RoPE | 4.74517 | 4.75483 | 4.77203 | 4.75735 | 0.01361 |
The seasonal baselines are 7.081 and 7.171, so every arm is about 34% better than the better baseline. Test MAPE (secondary, not used for any decision) shows the same ordering: A 4.198 to 4.264, B 4.205 to 4.230, C 4.281 to 4.305.

### 32.3 The frozen rule applied (delta = 0.10, k = 2)
Matched differences `d_s = MAPE(X, s) - MAPE(A, s)`; negative means X has the lower error.
| | d (seeds 0, 1, 2) | Mean | SD_d | k x SD_d | Leave-one-out means | (a) | (b) | (c) | (d) | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| B vs A | +0.02841, -0.06771, +0.01114 | -0.00939 | 0.05124 | 0.10249 | -0.0283, +0.0198, -0.0197 | no | no | no | no | **No difference detectable at this budget** |
| C vs A | +0.11076, +0.05471, +0.10587 | +0.09044 | 0.03105 | 0.06210 | +0.0803, +0.1083, +0.0827 | yes | **no** | yes | no | **No difference detectable at this budget** |
| C vs B (descriptive, no verdict claimed) | +0.08235, +0.12242, +0.09473 | +0.09983 | | | | | | | |

- **B vs A:** the signs differ across seeds and the mean (-0.009) is about a tenth of delta.
- **C vs A:** C has the higher error in all three seeds. The mean (+0.0904) is **below delta (0.10) by 0.0096**, so condition (b) fails and the frozen rule gives no verdict. Condition (c) holds, and condition (d) fails because two leave-one-out means are below delta. The rule is applied as frozen: nothing is relabelled, and delta is not revisited.
- The pre-registered statement of what the design could separate: **`k x SD_d` = 0.102 for B vs A and 0.062 for C vs A.** For B vs A the design could not separate differences smaller than about 0.10 points.
- Neutral primary hypothesis: no planned comparison separated from seed noise at this budget. Secondary conjecture (RoPE has the higher error): the frozen reading is "not supported at this budget" (d_C mean +0.0904, direction consistent with the conjecture, practical threshold not met).
- These are exploratory screening criteria for three seeds, not significance tests. Three same-sign differences occur by chance 25% of the time.

### 32.4 Reference-arm spread, against the pilot
Arm A's seed SD in the main experiment is **0.0329**, against the pilot's estimate of 0.0073 (95% interval 0.0038 to 0.0461). The realised spread is inside the interval but 4.5 times the pilot's point estimate. This is the uncertainty of three seeds that the pilot caveat described. The pilot's seeds were also on three different GPUs, and the main spread is within one GPU.

### 32.5 Cost, measured under P1 (observed)
| Arm | Train time (s), mean (range) | Batch-1 latency p50 (ms) | Peak training memory (MiB) | Peak inference memory (MiB) |
|---|---|---|---|---|
| A | 312.8 (310.9-313.8) | 4.34 | 623.27 | 66.91 |
| B | 312.9 (311.9-313.5) | 4.32 | 623.18 | 66.91 |
| C | 343.4 (342.9-343.9) | 4.69 | 622.95 | 67.01 |
- **RoPE (C) was slower in every matched pair:** +30.6 s of training (+9.8%, from +31.9, +29.6, +30.2) and +0.35 ms of latency (+8%). Sinusoidal (B) was indistinguishable from A (+0.06 s, -0.02 ms).
- The arm order was balanced, so this is not an order artefact: C ran in positions 3, 2 and 1 and took 343 s each time.
- Peak memory is flat across arms (differences of 0.08 and 0.31 MiB, deterministic). Under P1 this hides the extra memory RoPE used under default kernels in the smoke sessions (limitation 5).
- Timing within this session was far steadier (SD under 2 s) than across sessions. Latency and memory describe the deterministic P1 configuration and are not production figures.

### 32.6 Compute budget (estimate from the run files)
Study stages used about **75 minutes (1.25 h)** of GPU time against the 3 h ceiling (calibration 9.4 min, pilot 10.8, main 49.4, the repeat of seed 901 5.3; smoke sessions excluded), leaving about 105 minutes. The optional extension (6 runs) would take about 33 minutes.

### 32.7 What the report may and may not say (draft claims, for the user's review)
May say: under P1 at 8000 steps, on this synthetic data, none of the planned comparisons met the pre-registered screen; the differences between encodings in validation MAPE were small (B vs A about -0.009, C vs A about +0.090 points, against the practical threshold 0.10); C had the higher error than A in all three seeds but by less than the threshold; RoPE cost about 10% more training time and 8% more latency; the seed spread of the reference arm was larger than the pilot suggested.
May not say: that RoPE is worse (or that any encoding is better); that the encodings are equivalent (the design could not separate differences below about 0.10 for B vs A); anything about converged models, real utility data, other GPUs, or production latency and memory.

### 32.8 Decisions for the user (none taken)
1. Review the nine runs and the frozen-rule outputs above.
2. The optional extension (seeds 3 and 4): its pre-declared trigger condition is met (both verdicts are "no difference detectable"), but it needs the user's explicit approval and a budget review, which is above. Running it would require the analysis to handle five seeds. `analyze_main.py` is frozen for three, so that would be a logged post-hoc amendment (a mechanical generalisation, no change to the rule), and the three-seed result would stay the primary result with the five-seed result reported as a labelled extension.
3. Approve the remaining deliverables: figures (seed-level scatter, paired differences, cost), the Annex IV technical documentation (verified against the official text), the reproducibility README with a clean-environment rerun, and the video outline.
