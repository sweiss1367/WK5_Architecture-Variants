# EXPERIMENT_PLAN.md: CST 627 Week 7 Capstone

**Status:** DRAFT v1 for review. No implementation exists. Nothing here has been run, so every number below is a planning estimate and is labeled as one.
**Companion file:** `REQUIREMENTS.md` (approved).
**Location rule:** all Week 7 work lives in `week7_capstone/`. Earlier coursework is not modified.

## 1. Recommendation in one paragraph

Ablate **positional-encoding choice** in a small Transformer that forecasts the next 24 hours of electricity demand from the previous 7 days. The three settings are **learned absolute** (the reference, and what the Week 5 classifier uses), **fixed sinusoidal**, and **rotary (RoPE)**. The dataset is a **seeded synthetic hourly demand generator** for 16 co-op-style feeders. The primary metric is **validation MAPE** at a fixed step count with no early stopping. Nine main runs (3 settings × 3 seeds) should take well under half an hour on a T4, which leaves room for a smoke test and a pilot. The fallback if you can't defend positional encoding as "carried forward" is **LoRA rank**.

## 2. Approved design decisions and where they land

| Your decision | How the plan implements it |
|---|---|
| 3 matched seeds per setting to start | Seeds 0, 1, 2 for every arm. The shared backbone gets identical initial weights, data order and dropout stream per seed (Section 6.3). A pre-declared extension to seeds 3, 4 is optional (Section 9). |
| One determinism policy for all runs | Section 6.4. Same flags in every arm, verified in the smoke test. |
| End-to-end training time and peak allocated GPU memory | Section 7. Each run is its own process, timed with CUDA synchronization. |
| Everything fixed except the factor | Section 6.2 table. |
| No early stopping | Fixed step count, final-step weights are evaluated, no checkpoint selection. |
| Metric and comparison rule defined before running | Section 8. Numeric threshold is frozen after the pilot and before the main runs. |
| Matched per-seed comparison | Section 8: paired differences, means, SDs, every individual difference. |
| Cautious about three seeds | Section 8 states the chance-agreement fact for n=3 and the three-tier verdict. |
| Save every run independently | One JSON per run, written atomically, skip-if-exists on rerun. |
| Verify Annex IV against the regulation | Section 11. |
| Separate directory | `week7_capstone/`. |

## 3. Evaluating the five candidate factors

Scenario: Cobalt Grid sells short-horizon electricity-demand forecasting to rural utility co-ops, and Dr. Solberg needs one narrow architectural question answered.
Ratings are my judgment from the Week 5 code, the course pages, and general knowledge of these methods. None are measured.

### 3.1 Summary

| Factor | Implementation complexity | Confound risk | Compute cost (T4) | Forecasting relevance | Risk of failure |
|---|---|---|---|---|---|
| **Positional encoding** | Low. A swap inside one attention block, about 15 lines per setting. | Medium, manageable (Section 4.1) | Low | High. Demand is dominated by daily and weekly cycles, which is what position information is for. | Low-medium |
| LoRA rank | Medium. Needs a shared pretrained base plus an adapter stage. | Medium-high (alpha scaling, two-stage design) | Low | Medium. Adapting a base model to a new co-op is realistic. | Medium. Effects between ranks are likely tiny. |
| Quantization precision | Medium. bitsandbytes dependency on a T4. | Medium | Low | Low-medium | Medium-high for this study design |
| Multimodal fusion | Medium-high | High (arms differ in capacity) | Low-medium | Very high in practice (weather) | Medium-high |
| Diffusion/flow noise schedule | High | High | Medium-high (sampling is expensive) | Medium (probabilistic forecasts) | High |

### 3.2 Candidate notes

**1. Positional encoding (recommended).**
- *Why it fits the rubric.* The manipulated factor is one module in one block. The config diff is a single line. Each control decision is easy to defend out loud.
- *Why it fits the scenario.* In a Transformer that sees hourly steps as tokens, the positional encoding is what tells the model "this is hour 17 of a day." That is exactly the structure a demand forecast depends on.
- *Why it's explainable.* Week 5's classifier already has `self.pos_emb = nn.Embedding(MAX_LEN, D_MODEL)`. I'm treating that as the evidence that the factor was carried forward from earlier weeks. I can only see Week 5, so please confirm positional encoding was covered earlier (Question Q1).
- *Known weaknesses:*
  1. Cost measurements will likely be nearly flat across settings. Learned embeddings add about 10k parameters, and the other two add none. Time and memory will probably differ at the noise level. That is still an honest finding, but it makes this weaker than quantization on priority 4 (interpretable measurements).
  2. Learned absolute has a built-in parameter-count difference. It is intrinsic to the factor and will be disclosed, not hidden.
  3. Relative vs absolute encodings treat the "hour of day" signal differently. Section 4.1 controls this.

**2. LoRA rank (fallback).**
- Needs a base model pretrained once, then adapted at ranks such as 4, 8, 16.
- *Confound:* LoRA's scaling is `alpha / r`. Holding alpha fixed means the update scale changes with rank. Holding `alpha / r` fixed changes alpha. Either way a second quantity moves with the factor. The lesson calls this learning-rate coupling. It can be handled, but it has to be defended.
- *Noise-floor concern:* if the base model is pretrained once and shared, the seed spread only reflects adapter initialization and data order. That understates the noise a stranger would see retraining everything.
- *Result risk:* ranks 4, 8, 16 on a small model will probably be indistinguishable. That is a legitimate null, but it is a weak story for a video.

**3. Quantization precision.**
- Post-training quantization needs no extra training, so training time and peak training memory are identical across arms by construction. Your approved measurement plan (training time and peak allocated memory) would show nothing. The informative cost axis would be inference latency, which you did not select.
- Week 5 already measured this on a T4. INT8 accuracy was essentially equal to FP32 (0.9764 vs 0.9763 mean), and INT8 was slower (about 11.5 ms vs 5.2 ms median latency). Repeating it on a new dataset adds little.
- Training-precision variants are limited on a T4. It has no native bfloat16, so FP32 / FP16-AMP / BF16 would not be a clean three-setting ladder.

**4. Multimodal fusion (demand + weather).**
- The most realistic forecasting setup, but the arms (early concatenation, late fusion, cross-attention) have different parameter counts and compute by design. "One factor" is hard to defend.
- Synthetic weather would be generated by my own code, so the result mostly reflects how I coupled temperature to demand. Real weather needs a second acquisition and join step.

**5. Diffusion or flow-matching noise schedule.**
- Needs a conditional diffusion forecaster, multi-step sampling for evaluation, and a probabilistic metric. A schedule changes the noise-level distribution and effective loss weighting, and sampling-step count interacts with it. Each is a possible second factor.
- Highest implementation and debugging cost, highest chance of a result I can't explain. You asked me not to favor it because the discussion uses it as an example, and I'm not.

### 3.3 Why positional encoding wins on your priority list

| Priority | Result |
|---|---|
| 1. Exemplary rubric | Strong on design control and on the seed-honesty story. Honest-null outcomes are fully presentable. |
| 2. One-factor control | The best of the five. Confounds are listed in Section 4.1 and are testable. |
| 3. One Colab session | Easily. Estimated 9 runs of roughly 1-3 minutes each, to be measured in the pilot. |
| 4. Interpretable measurements | Accuracy: good. Cost: probably flat, reported as such. |
| 5. Explainable | Best of the five. |

## 4. Experimental design

### 4.1 Confounds specific to positional encoding, and the control for each

| # | Confound | Control |
|---|---|---|
| 1 | **Calendar covariates** (hour-of-day, day-of-week inputs) would hand every arm the same time information and hide the effect. | Model input is the demand value only. No calendar features. Stated as a limitation: real systems would add them. |
| 2 | **Window alignment.** If windows start at arbitrary hours, "position 0" doesn't mean midnight and the encodings are unfairly handicapped or helped. | Windows always start at 00:00. Position 24k therefore always lands on midnight for all arms. This favors absolute encodings by design and is disclosed. |
| 3 | **Output head leaking position.** A head that flattens all 168 positions can learn position itself and mask the encoding. | Mean-pool over tokens (as in Week 5) then linear to 24 outputs. Order information has to come through the encoding. Verified in the pilot that the model still learns (Section 10, V3). |
| 4 | **Parameter count.** Learned absolute adds 168×d parameters. | Intrinsic to the factor. Reported per arm. Not "fixed" by padding other arms. |
| 5 | **Where the encoding acts.** Learned and sinusoidal are added to the input once. RoPE rotates queries and keys in every layer. | Intrinsic to the techniques. Standard implementation of each, documented. |
| 6 | **Magnitude mismatch.** A PE of larger scale than the value embedding can dominate it. | Log the norm of value embedding vs positional term at initialization for each arm. Verified in the pilot (V6). No arm-specific scaling tricks. |
| 7 | **Hyperparameters tuned on one arm.** Tuning learning rate on the reference arm would favor it. | No per-arm tuning. Use Week 5-style AdamW defaults. The pilot only checks that all three arms train to a plateau. If one doesn't, the budget changes for all arms, before the main runs. |
| 8 | **Initialization differing between arms.** Extra parameters in one arm shift the RNG stream, so "seed 1" would not mean the same backbone. | Build the shared backbone first from seed `s`. Build the PE module from a separate seed. Reseed immediately before the training loop (Section 6.3). |

### 4.2 Settings (exactly three)

| Setting | Implementation | Extra parameters |
|---|---|---|
| **A. Learned absolute** (reference) | `nn.Embedding(168, d)` added to the input projection | 168×d |
| **B. Sinusoidal** | Fixed sin/cos table (Vaswani et al.), added to the input projection | 0 |
| **C. Rotary (RoPE)** | Rotation applied to q and k in each attention layer | 0 |

A "no positional encoding" arm would be a useful control, but you asked for exactly three settings, so it is an optional extra and not in the plan.

## 5. Dataset and architecture

### 5.1 Dataset: seeded synthetic hourly demand

- **Why synthetic.** Fully scripted and rerunnable, no download or license risk, and provenance is just the generator, its parameters and a hash.
- **Cost of that choice.** Results describe the seasonality I assumed. This is the first limitation in the Annex IV document, and the video says so.
- **Series.** 16 feeders ("co-ops"), hourly, 3 years each. Per-series parameters are drawn once from documented ranges with a fixed data seed (42), including peak season (summer-peaking vs winter-peaking), base load, and noise level.
- **Generator components (all documented in `make_data.py`).**
  1. Base load plus a daily profile (morning and evening peaks) that differs on weekdays and weekends.
  2. A latent temperature series (annual cycle + daily cycle + autocorrelated anomaly) that drives a nonlinear heating/cooling load. The model never sees temperature.
  3. Multiplicative noise and a few documented outlier hours and short gaps, filled by a documented rule.
  4. A hard positive floor on demand so MAPE is well defined.
- **Splits (temporal, fixed, identical for every run).** Months 1-24 train, 25-30 validation, 31-36 test. No shuffling across time.
- **Windows.** 168-hour context ending at midnight, next 24 hours as the target, stride 24.
- **Estimated size (to be confirmed).** About 11,500 training windows (16 series × roughly 720 days).
- **Normalization.** Each window divided by its own context mean, so the model predicts a ratio and the scale differs by feeder without hurting. Metrics are computed after converting back to demand units.
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
  PREREGISTRATION.md      # written before any main run
  README.md
  requirements.lock.txt   # exact versions captured from the Colab runtime
  configs/base.yaml       # everything fixed
  configs/arm_{A,B,C}.yaml  # each overrides exactly one field: pos_encoding
  src/make_data.py  src/model.py  src/run_ablation.py  src/analyze.py
  run_all.py              # driver: loops arms x seeds, one subprocess per run
  results/runs/*.json     # one file per completed run
  results/*.csv  figures/
  docs/ANNEX_IV.md
```

`python src/run_ablation.py --config configs/arm_B.yaml --seed 1` is the only way a number is produced. The notebook (if any) calls the same functions.

### 6.2 What is held fixed (the "welded dials")

Dataset and splits, windowing, normalization, model width/depth/heads/FF, dropout, head, loss, optimizer and schedule, batch size, step count, precision, determinism flags, evaluation code, seed list, and run order rule. The single line that differs between arm configs is `pos_encoding`.

### 6.3 Matched seeds

For seed `s`:
1. Seed everything with `s`, build the shared backbone.
2. Seed with a separate derived value, build the positional module.
3. Create the data-order generator from `s`.
4. Reseed with a derived value immediately before the training loop so the dropout stream is the same across arms.

The smoke test verifies that the backbone's initial weights are bitwise identical across arms for the same seed.

### 6.4 Determinism policy (one policy, all runs)

Python/NumPy/Torch/CUDA seeded. `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, FP32, DataLoader workers seeded (or no workers). If the fused attention backward is nondeterministic under this policy, the plan is to force the math attention backend for every arm. This needs to be verified (V4). The PyTorch version and snapshot date of the reproducibility notes are cited in the documentation. Deterministic mode may slow things down. It affects all arms equally and the cost is reported.

### 6.5 Order and isolation

- One subprocess per run, so each run has a fresh CUDA context and peak-memory counter.
- One discarded warm-up run before timing starts.
- Arm order rotates by seed (seed 0: A,B,C; seed 1: B,C,A; seed 2: C,A,B) so GPU thermal state and shared-GPU drift are not tied to one arm.
- Each run writes `results/runs/{arm}_seed{s}.json` atomically. The driver skips existing files, so a disconnected session resumes where it stopped.

## 7. Measurements per run

| Quantity | Definition |
|---|---|
| Validation error (primary) | MAPE (%) over all validation windows and the 24 horizons, in demand units, using final-step weights |
| Secondary errors | MAE (kW) on validation, and test MAPE evaluated once at the end of each run (not used for any decision) |
| Training wall-clock | Seconds from first to last training step with `torch.cuda.synchronize()` at both ends. Evaluation time is accumulated separately and excluded. Data loading to GPU excluded and stated. |
| Peak GPU memory | `torch.cuda.max_memory_allocated()` after `reset_peak_memory_stats()` at run start. Reported as allocated, not reserved, so it excludes CUDA context and allocator cache. |
| Context fields | Seed, arm, parameter count, steps, final train loss, determinism flags, torch/CUDA versions, GPU name, timestamps, git commit |

**Reference baselines (no training, reported for context).** Seasonal-naive (repeat the previous day) and same-hour-last-week, both scored with the same MAPE code. If a trained arm cannot beat these, the forecasting model is not informative (V3).

## 8. Primary metric and comparison rule (to be frozen in PREREGISTRATION.md)

**Primary metric:** validation MAPE at the final step.
**Reference arm:** A (learned absolute).
**Pairwise comparisons:** B−A, C−A (the two planned ones). C−B is reported descriptively. With three comparisons and three seeds, none are presented as confirmed.

**What gets reported for each comparison**
- Per-arm mean, SD (n−1), min and max across the three seeds.
- The three paired differences (one per matched seed), with their mean and SD.
- A seed-level scatter plot and a paired-difference plot, with all dots visible.

**Verdict categories** (decided before results, using a practical threshold δ):

| Verdict | Condition |
|---|---|
| **Separated** | The min-max bands of the two arms do not overlap, and \|mean paired difference\| ≥ δ |
| **Consistent direction, not separated (tentative)** | All three paired differences have the same sign and \|mean paired difference\| ≥ δ, but the bands overlap |
| **No difference detectable at this budget** | Anything else |

**Why the caution matters.** With three seeds, three paired differences share the same sign by pure chance about 25% of the time (2 × 0.5³) if there is no effect at all. The "tentative" tier is therefore weak evidence and is worded that way. No p-values or significance tests are reported.

**δ (the smallest difference I'd call meaningful).** To be set from the pilot's measured seed spread plus a practical argument, and written into PREREGISTRATION.md before the first main run. It is **not chosen from main-run results**.

## 9. Compute budget and staging

All times are planning guesses until the pilot measures them.

| Stage | Purpose | Runs | Est. time |
|---|---|---|---|
| Smoke | Whole pipeline on a tiny slice, assertions, determinism check | 2-3 tiny | under 5 min |
| Pilot | Per-run time and memory, convergence of all three arms, noise estimate, δ, step count | 3 arms × seeds 900-902 (not reused) | about 20-30 min |
| Main | The study | 3 arms × seeds 0, 1, 2 = 9 | about 10-30 min |
| Optional extension | Only if declared up front in the pre-registration: seeds 3 and 4 if the Stage-1 verdict is "tentative" or "no difference" | +6 | about 10-20 min |

Stage-1 and extended results are both reported, labeled by stage. The pilot seeds are listed in the package but are never mixed into the main tables. Total expected GPU time is well within one session, even allowing for a slow shared T4.

## 10. Items to verify before full implementation

| ID | Question | How | Pass criterion | If it fails |
|---|---|---|---|---|
| V1 | Is positional encoding really "carried forward" from earlier weeks? | You confirm (Q1) | Yes | Switch to LoRA rank (Section 3.2) |
| V2 | Per-run time, steps per second, and peak memory on the Colab T4 | Pilot | Main grid fits comfortably, leaving margin for one retry | Reduce steps, series count or model size for all arms |
| V3 | Does each arm learn, and beat seasonal-naive? | Pilot | All three arms reach a training-loss plateau and beat both baselines | Fix task or model before main runs (using pilot seeds only) |
| V4 | Do deterministic flags work with the attention path? Are two identical runs bitwise equal? | Smoke | Same-seed rerun gives identical validation MAPE, or the residual difference is measured and reported | Force math attention backend; if still nondeterministic, report as noise floor component |
| V5 | Backbone init identical across arms for the same seed? | Smoke assertion | Bitwise equal | Fix the seeding order |
| V6 | Is the PE term wildly larger or smaller than the value embedding? | Smoke: log norms | Same order of magnitude | Document; do not apply per-arm scaling |
| V7 | Is MAPE numerically stable (demand floor, no near-zero targets)? | Smoke | Min demand well above zero in all splits | Raise the floor in the generator (before main runs) |
| V8 | Are windows aligned so position 24k is midnight? | Unit test | Test passes on the generated arrays | Fix windowing |
| V9 | RoPE and sinusoidal correctness | Unit tests (known shapes, rotation preserves norm, relative-position property for RoPE) | Tests pass | Fix implementation |
| V10 | Is the seed spread small enough to separate any plausible effect? | Pilot noise estimate | Spread is not obviously far larger than δ | An honest null is acceptable. Do not add arms or tune to rescue it. |
| V11 | Environment pin | `pip freeze` captured on the real Colab runtime in the smoke test | Lockfile recorded with Colab image date | Pin what Colab provides (torch is preinstalled, not reinstalled) |
| V12 | Run-to-run timing noise on a shared free-tier GPU | Pilot repeats | Timing spread reported with every time claim | Report as measured, no stronger claims |
| V13 | Can I reach EUR-Lex from this environment to read Annex IV? | Try to fetch | Official text retrieved | You paste or upload the official text. I do not write Annex IV content from memory. |

## 11. Annex IV documentation plan (3-5 pages)

**Verification first.** Before drafting, I will retrieve the official text of Regulation (EU) 2024/1689 (Article 11 and Annex IV, and Annex III for the classification question) and structure the document from it.

**Classification caveat.** Annex IV applies to high-risk systems. Whether a demand-forecasting tool is high-risk depends on its use, possibly Annex III's critical-infrastructure entry, and I will not assert that from memory. The document will say it is written as a rehearsal under the stated assumption, and cite the classification text I verified.

**Draft outline (to be reconciled with the official Annex IV list):** system purpose and intended use; development methodology and design rationale (the ablation); data provenance and governance (generator, parameters, seed, hash); evaluation metrics and results with spread; limitations; known and foreseeable risks with controls; validation and testing records; crosswalk table (Annex IV item → NIST AI RMF function → ISO/IEC 42001 element → artifact path).

**Rules:** every claim cites its artifact and command. Limitations are a numbered section. NIST AI RMF and ISO/IEC 42001 cross-references are only included where I can state the mapping accurately. The course readings are the primary references, and I add no citations you haven't provided or I haven't read.

## 12. Rubric coverage

| Criterion (30 pts each) | Where the plan addresses it |
|---|---|
| Ablation design and control | Sections 4, 5, 6.2, 6.3. Each control has a stated reason in Section 4.1. |
| Seed honesty | Sections 7 and 8. All runs shown, per-seed scatter, min-max, verdict tiers, a null result is a valid outcome. |
| Reproducibility | Sections 6.1, 6.4, 6.5, V11. Config-driven, pinned, scripted data, run order, README verified on a fresh runtime. |
| Annex IV documentation | Section 11, V13 |
| Video | Outline will map to: the question, the controls (why these are welded), the result with spread, the package, and what the study cannot conclude, given equal time. Planned after results. |

## 13. Known limitations to carry into the documentation

1. Synthetic data reflects my assumptions about seasonality, not real co-op behavior.
2. One dataset realization. Seeds don't vary the data.
3. Three seeds can't estimate variance well, so the statistical claims are limited.
4. No calendar features and midnight-aligned windows favor absolute encodings, by design.
5. Fixed step budget may favor an arm that converges faster. That is a property of the arm, but it limits any claim about final accuracy.
6. Small model and short context do not predict behavior at production scale.
7. A shared free-tier GPU gives noisy timing. Memory is more reliable than time.
8. Bitwise reproducibility is not guaranteed across hardware or library versions. Statistical reproducibility is the claim.

## 14. Open questions for you

- **Q1.** Was positional encoding discussed or used in earlier weeks (beyond the Week 5 classifier's learned embedding)? If not, I recommend LoRA rank as the substitute.
- **Q2.** Is synthetic data acceptable for your video story, or do you prefer an open dataset (more acquisition risk, more realism)?
- **Q3.** Do you want the optional extension to seeds 3 and 4 declared in the pre-registration?
- **Q4.** Is a "no positional encoding" control worth adding as an optional fourth arm? It would exceed the "exactly three settings" plan, so I'd leave it out unless you want it.
