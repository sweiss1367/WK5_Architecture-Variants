# Week 7 Capstone: Positional-Encoding Ablation with Seed Variation

CST 627. A small time-series transformer forecasts 24 hours of electricity demand from the previous 168 hours. The one
factor that changes is the positional encoding:

| Arm | Positional encoding | Parameters |
|---|---|---|
| A | learned absolute (`nn.Embedding(168, 64)`) | 162,520 |
| B | fixed sinusoidal | 151,768 |
| C | RoPE on queries and keys, every layer | 151,768 |

Three matched seeds (0, 1, 2) per arm, 8000 steps, deterministic policy P1, nine runs in one GPU session. Everything
except `model.pos_encoding` is identical across arms, including the initial shared weights (checked by hash).

## Result in brief

All numbers are in `results/main_summary.json` and are validation MAPE in percent, lower is better.

| | mean | matched difference from A (seed 0, 1, 2) | mean difference | verdict under the frozen rule |
|---|---|---|---|---|
| A learned | 4.667 | n/a | n/a | reference |
| B sinusoidal | 4.658 | +0.028, -0.068, +0.011 | -0.009 | no difference detectable at this budget |
| C RoPE | 4.757 | +0.111, +0.055, +0.106 | +0.090 | no difference detectable at this budget |

C had higher error than A on all three seeds, but the mean difference (+0.090) is below the pre-registered practical
threshold of 0.10 points, so the frozen rule does not call it. The rule (delta = 0.10, k = 2) is a set of exploratory
screening criteria for three seeds, not a significance test. Under P1 on one Tesla T4, arm C trained about 9.8% slower
(343 s against 313 s) and had about 8% higher batch-1 latency (4.69 ms against 4.34 ms). Peak training memory was the same
in all arms. These cost numbers describe this model, this hardware, this workload and this deterministic configuration;
they are not a general statement about RoPE.

Recommendation within the scope of the experiment: no evidence to change the positional encoding for accuracy.

## What you can reproduce, and where

| Goal | Needs | Command |
|---|---|---|
| Recompute the analysis from the committed run files | CPU, `PyYAML` | `python src/analyze_main.py` |
| Regenerate the figures | CPU, `PyYAML`, `matplotlib` | `python src/make_figures.py` |
| Check that the frozen files are untouched | CPU, `PyYAML` | `python src/freeze.py --verify` |
| Regenerate the dataset and verify its hash | CPU, `numpy`, `PyYAML` | `python src/make_data.py --config configs/base.yaml --verify` |
| Run the test suite (about 10 minutes) | CPU, `torch`, `numpy`, `PyYAML`, `pytest` | `python -m pytest tests -q` |
| Re-run the nine training runs | one GPU, the pinned environment | `colab/week7_step6_main.ipynb` |

The nine training runs were made on a Colab Tesla T4. Training times and latencies will differ on other hardware, and
bitwise-identical weights are only expected on the same GPU model with the pinned library versions.

### 1. Quick check without a GPU

```
git clone https://github.com/sweiss1367/WK5_Architecture-Variants.git
cd WK5_Architecture-Variants/week7_capstone
git checkout claude/trusting-pascal-1qjsp0
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-figures.txt
python src/freeze.py --verify          # expect: freeze holds: 13 files match
python src/analyze_main.py             # rewrites results/main_summary.json, main_runs.csv, paired_differences.csv
git diff --stat results                # expect only the "created_utc"-style timestamp line to differ
python src/make_figures.py             # writes figures/*.png, figures/*.svg, figures/figure_manifest.json
```
`analyze_main.py` reads only the nine files in `results/runs/main/` and `configs/base.yaml`. Its output should equal the
committed `main_summary.json` apart from the timestamp.

### 2. Full environment and tests

The pinned environment is in `requirements.lock.txt` (Python 3.13, `torch==2.11.0+cu130`, `numpy==2.1.3`,
`PyYAML==6.0.3`). The `+cu130` build of torch comes from the PyTorch package index, so on a machine without that
build use:
```
pip install torch==2.11.0 numpy==2.1.3 PyYAML==6.0.3 pytest
python src/check_env.py --verify       # reports exactly what differs from the lock
python -m pytest tests -q
```
The test suite uses CPU and checks the data and leakage rules, the model, the run gates (environment, clean code,
frozen pre-registration, resume refusal, blinded evaluation), calibration, the analysis rule and the freeze record.
A real run refuses to start if the environment differs from the lock, so `check_env.py --verify` failing on a CPU machine
is expected and not a fault in the repository.

### 3. Re-running the experiment on Colab

1. Open `colab/week7_step6_main.ipynb` on a T4 runtime. The notebook pins the frozen commit
   (`8c561c196589cc4c7ee7789dced7c95723e805c1`).
2. Run the cells in order. The cell that checks the gates must pass before any run starts: environment equals the lock,
   code equals the frozen commit, the freeze record holds, policy P1, 8000 steps.
3. The nine runs take about 50 to 60 minutes. Do not interrupt the run cell. If it stops, re-running resumes but the
   driver then requires `--allow-resume` and the resume is recorded in `results/session_manifest.json`.

The earlier stages are kept for the record, in order: `week7_step3_smoke.ipynb` (determinism policy timing),
`week7_step4_calibration.ipynb` (training-loss-only step calibration), `week7_step5_noise_pilot.ipynb` (arm A noise
pilot). Calibration and pilot never use validation metrics for arms B or C.

## Layout

```
configs/    base.yaml (everything fixed, includes analysis delta and k) and arm_{A,B,C}.yaml (one field each)
src/        make_data.py, model.py, run_ablation.py, run_main.py, analyze_main.py, freeze.py, check_env.py   (frozen)
            make_figures.py, choose_policy.py, smoke_checks.py, calibrate.py, run_calibration.py, run_pilot.py,
            analyze_pilot.py                                                                               (not frozen)
tests/      CPU tests for the code above
colab/      one notebook per stage
results/    runs/{calibration,pilot,main}/ per-run JSON, main_summary.json, main_runs.csv, paired_differences.csv,
            freeze_record.json, determinism_decision.json, session_manifest.json, per-session provenance folders
figures/    fig1 seed-level MAPE, fig2 matched differences, fig3 cost, figure_manifest.json, README.md
docs/       Annex IV-style technical documentation
data/       generated series (not committed; make_data.py regenerates it, hash 65d899e6...)
```

Planning and record-keeping documents: `REQUIREMENTS.md`, `EXPERIMENT_PLAN.md` (the running log, including every
decision and deviation), `PREREGISTRATION.md` (frozen before the main runs).

## How the design is kept honest

- **Pre-registration.** `PREREGISTRATION.md` and 12 other files are hashed in `results/freeze_record.json`. The main
  driver refuses to run unless the freeze record verifies and its frozen commit is an ancestor of the current commit.
  `python src/freeze.py --verify` shows that nothing frozen changed after the main runs.
- **Matched seeds.** The shared backbone is initialised from the seed, the positional module from `seed + 10000`, and
  training is reseeded from `seed + 20000`, so arms start from identical shared weights.
- **Decision rule.** For each comparison with arm A, d_s = MAPE(X, s) - MAPE(A, s). A tentative call needs all three
  signs equal, |mean| >= 0.10, |mean| >= 2 x SD of d, and every leave-one-out mean still beyond 0.10 in the same
  direction. These are exploratory screening criteria for three seeds, not significance tests. Three same-sign
  differences occur by chance 25% of the time.
- **Policy P1.** Deterministic algorithms, fixed cuBLAS workspace, deterministic cuDNN, math attention backend. It was
  chosen on feasibility alone (time budget), not on how it affected any result.
- **No tuning per arm, no early stopping, final-step weights only.**
- **Provenance in every run file.** Commit, GPU UUID, session id, policy, config hash, data hash and weight hashes.

## Known limitations

- Three seeds per arm. The data is synthetic and the model small (about 150,000 parameters). The step budget (8000) was
  chosen by a pre-declared fallback rule and the model was not trained to convergence.
- One GPU model and one session. Cost results are specific to that setup and to the deterministic kernels.
- The optional two-seed extension (seeds 3 and 4) was not run. That decision was made after the nine runs were reviewed,
  and the three-seed results are the primary evidence.
- The arm A pilot SD (0.0073) was lower than the SD across the three main arm A seeds (0.0329). The pilot SD was not used
  to change delta.
- Local test environments here use newer library versions than the lock, so tests on CPU do not reproduce the
  GPU-session numbers. Only the Colab session did.

## What was tested in a clean environment

Fresh clone of branch `claude/trusting-pascal-1qjsp0`, fresh virtual environment, Python 3.13, CPU only, following the
commands above:

- Section 1 (no GPU): `freeze.py --verify` reports 13 files match. `analyze_main.py` output equals the committed
  `main_summary.json` except for the `created_utc` line. `make_figures.py` reproduces the PNGs byte for byte; the SVGs
  and `figure_manifest.json` differ only in metadata such as the creation time.
- Section 2 (`torch==2.11.0`, `numpy==2.1.3`, `PyYAML==6.0.3`): `make_data.py --verify` regenerates the dataset with
  hash `65d899e6...` and passes the leakage check. 124 tests pass (about 9.5 minutes).
- `check_env.py --verify` reports one difference from the lock: torch installs as `2.11.0` from PyPI, the lock says
  `2.11.0+cu130`. Real runs need the lock to match, as stated above.

Not tested outside the original session: the GPU training path (the nine runs and the Colab notebooks), because no GPU was
available for the clean check.
