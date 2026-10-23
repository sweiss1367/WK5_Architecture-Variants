---
title: "Annex IV-Style Technical Documentation Summary: Positional-Encoding Ablation for Hourly Electricity-Demand Forecasting"
subtitle: "CST 627 Week 7 capstone. Academic exercise, not a conformity assessment."
---

## 0. What this document is, and what it is not

This is an academic exercise. It organises the evidence from a small forecasting experiment using the headings that
Annex IV of Regulation (EU) 2024/1689 sets for technical documentation under Article 11. It is **not** a completed
conformity assessment. No system was placed on the market or put into service. No EU declaration of conformity, notified-body
review, quality management system or post-market monitoring plan exists, and none is claimed.

**Sourcing.** The Annex IV headings and the Article 11 requirement used here come from a secondary summary of the
Regulation supplied for this assignment, not from the Official Journal text, which I could not open from my working
environment. Before submission, each heading and the Article 11(1) wording should be checked against the Regulation on
EUR-Lex. Application dates and amending acts are left out because they do not affect this exercise and I could not verify
them.

**Classification assumption.** Annex IV applies to providers of high-risk systems, so the first step in a real file is
classification under Article 6. I have not done that analysis against the Regulation's text. This exercise treats the
forecaster as if it were a candidate high-risk system, for example one feeding grid operations planning, because that
use gives the documentation a purpose. A load forecaster used only for planning may well fall outside the high-risk
categories. That question stays open here.

## 1. Purpose and general description (Annex IV, section 1)

**Intended purpose of the study.** To decide whether the positional encoding of a small time-series transformer should be
changed, judged on forecast accuracy and on runtime cost. The model reads 168 hours of demand for one meter and predicts
the next 24 hours. The positional encoding is the only thing that differs between the three arms.

| Arm | Positional encoding | Parameters |
|---|---|---|
| A | learned absolute embedding, 168 positions | 162,520 |
| B | fixed sinusoidal table added at the input | 151,768 |
| C | rotary (RoPE) on queries and keys in every layer | 151,768 |

**Not intended for.** Use with real customer data, operational dispatch decisions, or any claim about transformers or
RoPE in general. The data are synthetic and the model is small.

**Hardware and software.** Nine runs on one Tesla T4 in one Colab session, Python 3.13, `torch==2.11.0+cu130`,
`numpy==2.1.3`, `PyYAML==6.0.3` (`requirements.lock.txt`). The repository runs the analysis and tests on CPU. There are no
deployers, no instructions for use and no versioned release, because nothing was deployed.

## 2. Development and design process (Annex IV, section 2)

**Design choice and its rationale.** A one-factor ablation. Shared backbone weights are initialised from the seed, the
positional module from `seed + 10000` and training from `seed + 20000`, so the three arms start from bitwise-identical
shared weights (checked by hash). Seeds 0, 1 and 2 are matched across arms, and differences are analysed per seed.
Using matched seeds removes seed-to-seed variation in the shared weights from the comparison. It does not remove
everything: arm A's three seeds differ by a standard deviation of 0.033 MAPE points.

**Architecture.** Three transformer encoder layers, model width 64, four heads, feed-forward width 256, dropout 0.1,
mean pooling over positions and a linear head (output `1 + head(mean-pool)`).

**Trade-offs and fixed choices.**

- **Training length.** 8000 steps. Calibration looked for a step count where training loss improved by less than 2%
  between doublings. None of the tried grid values got there (4000 to 8000 still gave 5.12%), so the pre-declared
  fallback of 8000 was used. The models are not converged.
- **Determinism policy P1.** Deterministic algorithms, a fixed cuBLAS workspace, deterministic cuDNN and the math attention
  backend. P1 was chosen on feasibility alone (projected runtime against a three-hour ceiling), before seeing any
  accuracy result. A run repeated on one GPU matched bitwise. The cost is slower kernels, so latencies are not those of
  a tuned deployment.
- **No per-arm tuning, no early stopping, final-step weights only.**

**Data (see section 3), human oversight and pre-determined changes.** Human oversight and pre-determined changes are
not applicable: the system has no deployers. What did exist was oversight of the experiment itself. The decision rule was
frozen before the nine runs, and every stage needed explicit approval before the next began.

## 3. Data provenance and governance (Annex IV, section 2, data)

| Item | Detail |
|---|---|
| Source | Synthetic. A seeded generator (`gen-v1`, data seed 42) builds hourly demand for 16 meters over 3 years starting 2021-01-01. |
| Generator | A daily profile, a weekend factor, a latent temperature effect and AR(1) noise, with rare outlier hours (0.6 to 1.5 times) and short gaps of 1 to 6 hours. Gaps are filled with the last earlier value. |
| Fingerprint | SHA-256 `65d899e60a00802062ac5c9c0dc5202d8d5ed6e6a0d64cba593e8985fc53f695`, identical across machines. |
| Split | By time, no shuffling: 24 months train, 6 validation, 6 test. Windows are split by target time, start at midnight, stride 24 h. |
| Window counts | 11,568 train, 2,912 validation, 2,928 test. |
| Leakage control | A leakage check runs with every data build (and in the test suite). Calibration used training loss only, and code gates block validation and test metrics for arms B and C in the calibration and pilot stages. |
| Personal data | None. Real data would bring Article 10 questions (representativeness, bias, personal data), which this exercise does not address. |

**Honest limit.** Synthetic data can show whether the arms differ on this generator. It cannot show how they would behave on
real load curves, whose structure the generator only imitates. The generator's weekly and daily patterns may also favour
some positional schemes over others.

## 4. Training methodology (Annex IV, section 2)

AdamW, learning rate 0.001, weight decay 0.01, 100 warmup steps then linear decay, batch size 128, no gradient
accumulation, fp32, normalised L1 loss. Main stage: three arms times seeds 0, 1 and 2 in one GPU session on the frozen
commit, with the arm order rotated by seed (A,B,C / B,C,A / C,A,B) so that thermal or session drift does not line up
with one arm. A calibration stage and a noise pilot (arm A only, seeds 901 and 902) came first, and their results were not
mixed into the main tables.

**Pre-registration.** Before the main runs, `PREREGISTRATION.md` and twelve other files were hashed
(`results/freeze_record.json`). The runner refuses to start unless the record verifies. After the runs,
`python src/freeze.py --verify` still reports that 13 files match.

## 5. Evaluation metrics and results (Annex IV, section 4)

**Metric choice.** Primary metric: validation MAPE (percent), because a percentage error is comparable across meters of
different size and is the form a demand planner reads. Secondary: validation MAE in kW, test MAPE, training time, batch-1
inference latency (p50) and peak memory. MAPE is undefined near zero demand, so the generator enforces a positive floor;
MAPE also weights errors at low demand more heavily than errors at peak demand, which may not match what matters
operationally. Two seasonal baselines score 7.08 and 7.17.

**Decision rule (frozen).** For each comparison with arm A, d is the per-seed MAPE difference. A tentative call needs all
three differences to share a sign, |mean| at least 0.10 points, |mean| at least 2 times the standard deviation of d, and
every leave-one-out mean to stay beyond 0.10 in the same direction. These are exploratory screening criteria for three
seeds, not significance tests. Three same-sign differences occur by chance 25% of the time.

| Arm | Validation MAPE, seeds 0, 1, 2 | Mean | Difference from A (per seed) | Mean difference | Frozen-rule verdict |
|---|---|---|---|---|---|
| A | 4.634, 4.700, 4.666 | 4.667 | n/a | n/a | reference |
| B | 4.663, 4.632, 4.677 | 4.658 | +0.028, -0.068, +0.011 | -0.009 | no difference detectable at this budget |
| C | 4.745, 4.755, 4.772 | 4.757 | +0.111, +0.055, +0.106 | +0.090 | no difference detectable at this budget |

**Reading it.** C had higher error than A on all three seeds, but its mean difference (+0.090) is below the pre-registered
threshold of 0.10, so the rule does not call it. B and A show mixed signs. Test MAPE moves the same way (A 4.20 to 4.26, B 4.21 to 4.23, C 4.28 to 4.31).

**Cost under P1 on one T4.** Training time 312.8 s (A), 312.9 s (B) and 343.4 s (C), so C was 9.8% slower in every matched
pair. Batch-1 p50 latency 4.34, 4.32 and 4.69 ms. Peak training memory about 623 MiB in all arms. These describe this model, hardware, workload and deterministic
configuration. They are not a general statement about RoPE.

**Release recommendation (within the scope of this experiment).** The evidence is insufficient to recommend changing the
positional encoding for accuracy. Learned and sinusoidal encodings showed no detectable difference at the tested budget.
RoPE added a measurable runtime cost without an accuracy improvement that met the rule.

## 6. Limitations, known risks and residual risk (Annex IV, section 5)

| Risk or limit | Why it matters | What was done | Residual |
|---|---|---|---|
| Three seeds | The spread of three differences is a rough estimate, and a three-seed sign pattern can arise by chance. | Frozen, exploratory rule; verdicts worded as "no difference detectable at this budget". | A real difference below about 0.1 points would not be reliably detected. |
| Noise underestimated | The arm A pilot SD (0.0073) was well below the main-run SD (0.0329). | Reported; delta was not lowered to fit the pilot. | Seed noise at this budget is larger than the pilot suggested. |
| Not converged | At 8000 steps the loss was still falling, and the ranking could change with longer training. | Stated as a fallback choice; extension not run. | Conclusions apply to this budget only. |
| Synthetic data | Real load data has structure, outliers and drift the generator lacks. | Provenance and hash documented. | No evidence about real data. |
| Single GPU and policy | Latency and memory depend on kernels; P1 is slower than defaults. | Environment pinned; hardware stated. | Cost figures do not transfer to other hardware or to default kernels. |
| Metric choice | MAPE over-weights low-demand hours. | MAE in kW and test MAPE also recorded. | An operational cost metric was not evaluated. |
| Bias and subgroup testing | A real file would test performance across customer groups and conditions. | Not applicable to synthetic meters. | Not assessed. |
| Cybersecurity and robustness | A real system needs Article 15-style checks. | Not assessed. | Not assessed. |
| Optional extension (seeds 3 and 4) | More seeds would tighten the estimate. | Not run; the three-seed results stay primary. | Open. |

## 7. Lifecycle, standards, declaration and monitoring (Annex IV, sections 6 to 9)

| Annex IV section | Status in this exercise |
|---|---|
| 3. Monitoring, functioning and control | Run files record commit, GPU id, session, policy, config hash and data and weight hashes. No deployed system exists, so no operational logging or alerts. |
| 6. Changes through the lifecycle | Changes are logged in `EXPERIMENT_PLAN.md`, the amendment log in `PREREGISTRATION.md`, the git history and `session_manifest.json`. One deviation is recorded: the 8000-step calibration used commit `198aad8` rather than the pinned `c8ab0d2`, and a diff showed no run-executing code changed. |
| 7. Harmonised standards | None applied. The next section maps the work to two frameworks, which is not a claim of conformity to any standard. |
| 8. EU declaration of conformity | **Not produced.** |
| 9. Post-market monitoring plan | **Not produced.** There is no market placement to monitor. |

## 8. Crosswalk to NIST AI RMF and ISO/IEC 42001

This maps activities to the framework headings at topic level. It does not claim conformity, and I have not cited
clause numbers.

| NIST AI RMF function | What this project did | ISO/IEC 42001 theme (AI management system) |
|---|---|---|
| **Govern** | Staged approvals, a frozen pre-registration, a written plan and amendment log, defined roles (analyst approves each step). | Leadership, policy, roles, documented information |
| **Map** | Stated purpose and non-purposes, assumptions, data provenance, a classification caveat (section 0). | Context of the organisation, scope, AI system impact and risk identification |
| **Measure** | Pre-declared metric and decision rule, matched seeds, seed variation reported, cost measured, baselines. | Performance evaluation, internal review of results |
| **Manage** | Limitations table with residual risk, deviations logged, optional extension deferred, no deployment. | Risk treatment, continual improvement, change control |

Where the project does not meet the framework's intent (no incident response, no deployer feedback, no organisational
accountability beyond one analyst), the gap is real and is not covered by this table.

## 9. Open items before submission

1. Check the Annex IV heading list and the Article 11(1) wording against the Regulation on EUR-Lex.
2. Analyse Article 6 classification against the Regulation's text and replace the assumption in section 0.
3. Confirm the crosswalk level of detail against what the course materials ask for.
