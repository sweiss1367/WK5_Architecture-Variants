# REQUIREMENTS.md: CST 627 Week 7 Capstone

**Title:** Seed-Varied Ablation Study and an EU AI Act Annex IV Reproducibility Package
**Status:** DRAFT v3. Checklist now separates mandatory deliverables from adopted commitments and recommended enhancements. No implementation has started.
**Points:** 150 (Project, CLO4 / CLO5 / CLO6)

## 0. Sources and tags

| Tag | Meaning |
|---|---|
| **[ASSIGN]** | Stated in the Week 7 assignment text you provided. Mandatory. |
| **[RUBRIC]** | Stated in the grading rubric. Mandatory because it is graded. |
| **[USER]** | Operating principle you stated in this conversation. Treated as binding project rules. |
| **[COURSE]** | Lesson/intro/interactive guidance. Not part of the assignment text, so it is **optional** unless it also supports a rubric criterion. |

Examples in the course pages (LoRA rank 4/8/16, MAPE, 11-12 minute runs, five seeds) are examples, not instructions. Run times and thresholds in them are not requirements.

## 1. Explicit assignment requirements (all mandatory)

### 1.1 Experiment

| ID | Requirement | Source |
|---|---|---|
| A1 | Choose **ONE** factor carried forward from earlier weeks, from this list: LoRA rank, positional-encoding choice, diffusion/flow-matching noise schedule, quantization precision, multimodal fusion strategy. | [ASSIGN] |
| A2 | Controlled ablation across **at least three settings** of that factor. | [ASSIGN], [USER] |
| A3 | **At least three random seeds per setting.** | [ASSIGN], [USER] |
| A4 | All runs inside **a single free-tier Colab GPU session**. | [ASSIGN], [USER] |
| A5 | Build the dataset yourself: either synthetic electricity-demand traces (Claude Code or Python), or an open series from data.gov, an open Kaggle dataset, or an open Hugging Face dataset. Acquisition or generation must be **scripted and rerun cleanly**. | [ASSIGN] |
| A6 | Report **means with seed-level spread**. | [ASSIGN], [USER] |
| A7 | State **plainly** whether observed differences exceed run-to-run noise. | [ASSIGN], [USER] |
| A8 | Log **wall-clock latency and peak memory alongside accuracy** (accuracy = validation error). | [ASSIGN], [USER] |
| A9 | Exactly one factor changes between arms. All else held identical. | [RUBRIC], [USER] |
| A10 | Report all results, including unsuccessful and inconclusive runs. | [USER], [RUBRIC] (spread reported even when it swamps the effect) |

Scenario context (Cobalt Grid, short-horizon electricity-demand forecasting for rural utility co-ops; Dr. Ingrid Solberg as Head of Research, Marcus Adeyemi as compliance lead) is the framing for the task and audience. It does not add a separate deliverable.

### 1.2 Reproducibility package (repo)

| ID | Requirement | Source |
|---|---|---|
| R1 | Pinned dependencies / environment. | [ASSIGN] |
| R2 | Seeded, config-driven entry point(s). | [ASSIGN] |
| R3 | Config files. | [ASSIGN] |
| R4 | Results tables and plots. | [ASSIGN] |
| R5 | README a stranger can follow without contacting you. | [ASSIGN] |
| R6 | Rubric standard: environment, seeds, data generation and run order are captured so another engineer could rerun it **end to end**. | [RUBRIC] |

### 1.3 Annex IV-style technical documentation summary

| ID | Requirement | Source |
|---|---|---|
| D1 | **3-5 pages.** | [ASSIGN] |
| D2 | Must cover: system purpose, data provenance, training methodology, evaluation metrics, limitations, known risks. | [ASSIGN] |
| D3 | Cross-reference **NIST AI RMF** and **ISO/IEC 42001** where your controls map. | [ASSIGN] |
| D4 | Rubric standard: addresses the required Annex IV elements **accurately**, in language a regulator or auditor could follow. Misstating what Annex IV requires drops the score to Developing. | [RUBRIC] |

### 1.4 Video

| ID | Requirement | Source |
|---|---|---|
| V1 | Walk Dr. Solberg and Marcus through **the study, the package, and what the study cannot conclude**. | [ASSIGN] |
| V2 | Duration **8-10 minutes**, maximum 10. (Assignment says 8-10; submission rules say max 10. I will aim for 8:30-9:30 to satisfy both.) | [ASSIGN] |
| V3 | Face visible on camera for the **entire** video. Picture-in-picture during screen share is acceptable. Voice-over slides without a visible face are not permitted. | [ASSIGN] |
| V4 | Own words. Reading verbatim from a script is not permitted. A brief outline or speaker notes for structure is fine. | [ASSIGN] |
| V5 | Clear audio: quiet space, speak toward the microphone. | [ASSIGN] |
| V6 | Format MP4 (preferred), MOV, or WebM. | [ASSIGN] |
| V7 | Upload via the LMS assignment dropbox. | [ASSIGN] |
| V8 | States the finding **and its limits with equal clarity**. Overstating the finding or leaning on notes lowers the score. | [RUBRIC] |

## 2. Grading rubric (Capstone Ablation Study Rubric, 150 points)

Five criteria at 30 points each. Each has four levels. **The pasted rubric lists level totals of 150 / 128 / 105 / 75 in the header, but the per-criterion cells are 30 / 26 / 21 / 15, which sum to 150 / 130 / 105 / 75. Treat the per-criterion cells as authoritative and note the 128 vs 130 mismatch as a possible typo (not worth asking the instructor about unless it matters).**

| Criterion | CLO | Max | Exemplary (30) | Proficient (26) | Developing (21) | Beginning (15) |
|---|---|---|---|---|---|---|
| Ablation Design and Control | 4 | 30 | Exactly one variable changes, all else fixed, **and student can defend each control decision** | One confound left unaddressed | Multiple variables change at once | No genuine controlled comparison |
| Seed Variation and Statistical Honesty | 4 | 30 | Multiple seeds per arm, spread reported **including when it swamps the effect the student hoped to show** | Minor gaps in spread reporting | Spread downplayed or omitted | Single run per arm presented as a finding |
| Reproducibility Package Quality | 6 | 30 | Environment, seeds, data generation, run order captured; **rerunnable end to end** | One missing detail | Requires guesswork | Not reproducible |
| Annex IV Technical Documentation | 6 | 30 | Required Annex IV elements **accurate**, auditor-readable | Most elements, accurate | Partial or misstates Annex IV | No usable documentation |
| Video Communication | 5 | 30 | Camera-on, own words, **finding and its limits with equal clarity** | Clear, minor pacing issues | Overstates finding or reads from notes | Camera off, or read verbatim |

## 3. Evidence needed per criterion

| Criterion | Evidence/artifact |
|---|---|
| Ablation design (30) | One base config + per-arm config diff showing a single changed field. A written design document with one defended decision per control (data order, tokenizer/input pipeline, optimizer and LR, precision, step budget, batch size, early stopping). Explicit handling of known confounds that the factor could couple into. |
| Seed honesty (30) | Per-seed results CSV with the seed values listed. Mean plus min-max (or SD) per arm. Seed-level scatter plot. An explicit sentence stating whether the effect exceeds seed spread, including if it does not. All runs shown, including failures. |
| Reproducibility (30) | Lockfile / pinned requirements, seeded entry-point script, configs, data script (with checksum if downloaded), driver for run order, README quickstart that I have run in a fresh Colab runtime, and logs. |
| Annex IV (30) | 3-5 page document with the six required topics, an accurate account of what Annex IV asks for (from the Regulation itself), and a crosswalk to NIST AI RMF and ISO/IEC 42001. |
| Video (30) | MP4 8-10 min, face visible throughout, own words, covering study + package + limits. |

## 4. Design decisions the assignment leaves open

Each needs your approval before implementation. These are choices, not requirements.

> **Status tracking:** the current status and resolution of D1-D12 is kept in `EXPERIMENT_PLAN.md` Section 15. The table below is the original list as approved.

| # | Decision | Notes |
|---|---|---|
| D1 | **Which factor.** Assignment: "carried forward from earlier weeks", from five options. Your Week 5 repo contains architecture variants, quantization, distillation, a batch-1 benchmark harness on a T4, and scaling-law work. I don't know which of the five options you used in weeks 1-6. | **Quantization precision** is the closest match to what I can see in the repo, but I haven't verified it matches "carried forward". It also has a subtlety: post-training quantization has no training randomness, so seed variation would come from the trained model's seed, and latency/memory differences would be large and nearly deterministic. That may make the "does the effect exceed noise" question trivial for latency and meaningful only for accuracy. Alternatives (LoRA rank, positional encoding) vary seed-sensitive training outcomes more naturally. **Your call; I need to know what you built earlier.** |
| D2 | **Dataset.** Synthetic seeded demand traces vs. an open series. Synthetic is cheapest and fully scripted but carries a representativeness limitation that must be documented. | Annex IV data provenance either way. |
| D3 | **Model and task.** Which model carries the factor, and what is forecast (next-hour demand?). Validation error metric (MAPE is the course's running example, but the assignment says only "validation error"). MAPE is unstable near zero demand. | Affects metric validity. |
| D4 | **Which three settings.** At least three levels of the chosen factor. | Follows D1. |
| D5 | **Seed count per arm and run budget.** Minimum is 3. The lesson's heuristics (a three-hour window, a pilot noise run, 5 seeds) come from the course pages, not the assignment. I'd treat a 3-hour limit as a planning ceiling, not a stated rule. | Needs a measured per-run time first. |
| D6 | **Determinism policy.** Enable PyTorch deterministic algorithms (slower, affects latency) vs. accept and document nondeterminism as part of the noise floor. | Latency is a measured outcome, so this choice interacts with A8. |
| D7 | **Latency and peak-memory protocol.** Training step time vs inference time vs end-to-end run time. Warm-up, CUDA synchronization, memory-counter reset between runs, batch size. The assignment says "wall-clock latency" without defining it. | Metric validity and defensibility. |
| D8 | **Step budget and early stopping.** Fix gradient steps and disable early stopping, or justify otherwise. | Prevents a confound between arms. |
| D9 | **What counts as "exceeds noise".** Decision rule (for example, gap versus min-max band overlap). Writing it before results guards against rewriting the success criterion after seeing results ([USER] principle: no conclusions before experiments finish). | A7 and the Seed Honesty criterion. |
| D10 | **Interrupted-session policy.** Save results after every run. Report partial runs as partial. | A4 and A10. |
| D11 | **Annex IV content source.** The Regulation excerpt supplied in the readings does not reproduce Annex IV. I'd read Annex IV itself from EUR-Lex and cite the exact structure. Do not rely on the aiacto.eu or arXiv 2508.08804 excerpt for legal claims (the readings page itself rates them low rigor / commercial). | Rubric penalizes misstating Annex IV. |
| D12 | **Repo layout.** Existing repo is Week 5 content. New Week 7 work should live in its own subfolder on this branch so Week 5 artifacts aren't disturbed. | Housekeeping. |

## 5. Three tiers: what is graded, what we committed to, what is optional

Nothing is treated as a grading requirement unless it traces to the assignment text or the rubric.

| Tier | Meaning | Consequence of skipping |
|---|---|---|
| **[M] Mandatory** | Stated in the assignment text or rubric | Costs rubric points or breaks the assignment |
| **[A] Adopted** | A design commitment you approved for scientific validity. Binding on this project, but **not** an assignment requirement | No rubric penalty by itself. Skipping it weakens the study, so drop it only on purpose |
| **[R] Recommended** | Course guidance or my suggestion. One sensible way to meet a mandatory item, not the only way | None. Replace or drop freely |

**Adopted [A] (approved by you):** pre-registration file, smoke test, per-run independent saving, one documented determinism policy, matched-seed comparison with an interpretation rule fixed before results, no early stopping, fixed step budget, separate training-duration and inference-latency measurement, peak memory reset before each run, disclosure of resumed sessions.

**Recommended [R]:** seed-level scatter plot (the assignment requires "plots", not this plot), fresh-runtime README test, explicit run-order driver script (the rubric requires run order to be *captured*, which a documented command sequence also satisfies), pilot run, optional extension seeds, leave-one-out check, session manifest, rotating arm order, reference baselines (seasonal-naive), test-set evaluation, crosswalk table format, numbered limitations section, `--smoke` flag name.

**Reading and legal-source extras [R]:** reading the Regulation beyond Article 11 and Annex IV. (Verifying Annex IV against the official text is [A], because the Annex IV rubric criterion penalizes misstating it.)

## 6. Reading and citation notes (from the readings page)

- Required: floating-point non-associativity paper (note the vendor-affiliation caveat on the Groq LPU section); PyTorch Reproducibility notes (version-volatile, so cite version and snapshot date); Regulation (EU) 2024/1689; arXiv 2508.08804v1 excerpt (single paragraph, v1 preprint, low rigor).
- Supplementary only: aiacto.eu (commercial advisory, no author/date) and PyTorch issue #109856 (an unreviewed bug report).
- Do not invent citations beyond what the course provides or what I read from the primary text.

## 7. Final submission checklist

### 7A. Mandatory deliverables [M]

**Experiment**
- [ ] One factor chosen from the five allowed (A1)
- [ ] ≥3 settings, ≥3 random seeds per setting (A2, A3)
- [ ] Exactly one factor changes between arms, all else held fixed, and each control decision defended (A9, Design rubric)
- [ ] All runs inside a single free-tier Colab GPU session, or any deviation disclosed accurately (A4)
- [ ] Dataset built by me, with scripted generation or acquisition that reruns cleanly (A5)
- [ ] Means reported with seed-level spread (A6)
- [ ] Plainly stated whether observed differences exceed run-to-run noise, including when spread swamps the effect (A7, Seed rubric)
- [ ] Wall-clock latency and peak memory logged alongside validation error (A8)
- [ ] Every run reported, including unsuccessful and inconclusive ones (A10)

**Repo**
- [ ] Pinned dependencies (R1)
- [ ] Seeded, config-driven entry point (R2) and config files (R3)
- [ ] Results tables and plots (R4)
- [ ] README a stranger can follow without contacting me (R5)
- [ ] Environment, seeds, data generation and run order captured so the study can be rerun end to end (R6)

**Annex IV-style documentation**
- [ ] 3-5 pages (D1)
- [ ] System purpose, data provenance, training methodology, evaluation metrics, limitations, known risks (D2)
- [ ] NIST AI RMF and ISO/IEC 42001 cross-references where controls map (D3)
- [ ] Annex IV described accurately (D4)

**Video**
- [ ] 8-10 minutes, never over 10 (V2)
- [ ] Face visible the entire time (V3)
- [ ] Own words, notes/outline only, no verbatim reading (V4)
- [ ] Clear audio (V5)
- [ ] MP4 preferred, MOV or WebM accepted (V6)
- [ ] Covers the study, the package, and what the study cannot conclude, with the finding and its limits given equal clarity (V1, V8)
- [ ] Uploaded to the LMS assignment dropbox (V7)

### 7B. Adopted commitments [A]

- [ ] Pre-registration file committed before any main run
- [ ] Smoke test passed before the pilot
- [ ] Each run saved independently, with a completed/interrupted/failed status
- [ ] One determinism policy chosen before benchmarking, applied to every arm, limits documented
- [ ] Matched per-seed differences with mean, SD and every individual value; interpretation rule fixed before results
- [ ] No early stopping, fixed step count, final-step evaluation
- [ ] Training duration and inference latency reported as separate quantities, measured with CUDA synchronization, inference after warm-up
- [ ] Peak GPU memory counters reset before each run, measured the same way in every run
- [ ] Resumed vs uninterrupted execution disclosed
- [ ] Annex IV content verified against the official regulation text

### 7C. Recommended enhancements [R] (drop any without penalty)

- [ ] Seed-level scatter plot
- [ ] README quickstart tested in a fresh Colab runtime
- [ ] Explicit driver script for run order
- [ ] Pilot on separate seeds
- [ ] Optional extension to seeds 3 and 4
- [ ] Leave-one-seed-out check
- [ ] Session manifest with GPU UUID and versions
- [ ] Seasonal-naive reference baselines
- [ ] Framework crosswalk table
- [ ] Numbered limitations section

## 8. Highest-risk rubric requirements

1. **Seed Variation and Statistical Honesty (30).** The exemplary level explicitly rewards reporting spread when it swamps the effect. The risk is a result that looks like a win at 3 seeds. Mitigation: fix the decision rule before running (D9) and use min-max bands, not just SD, with 3 seeds.
2. **Ablation Design and Control (30).** "Defend each control decision" means documenting *why* each held-fixed item is fixed, and neutralizing couplings specific to the chosen factor (D1/D8). One unaddressed confound costs 4 points. Which couplings exist depends on D1.
3. **Annex IV Documentation (30).** The supplied legal excerpt omits Annex IV, and "misstates what Annex IV requires" drops two levels. The 3-5 page limit makes a thorough mapping tight. Needs the primary text (D11).
4. **Video (30).** Camera-on for the full 10 minutes, no verbatim reading, and equal weight on finding and limits. The 8-10 minute window is easy to miss on either side. Your content is mostly decided by the other four criteria, so the risk is delivery and pacing.
5. **Reproducibility (30).** "End to end" includes data generation and run order. A free-tier Colab session can drop mid-run, so per-run result saving and resumability matter (D10). A README that was never run in a clean runtime is the usual single missing detail.
6. **A8 latency and peak memory.** It is not a rubric line by itself, but the assignment requires it, and it is easy to measure incorrectly (D6, D7).
