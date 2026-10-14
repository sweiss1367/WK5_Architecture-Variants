"""
Builds notebooks/week6_audit.ipynb (no outputs). Run:  python notebooks/build_notebook.py

The notebook is the Colab driver for the GPU stages. Cells for the CPU-side evidence (dataset
audit, hashes, contamination sweep) display files that were already produced and committed.
Nothing in it contains a typed-in result: every table is read from results/ or logs/.

The executed copy (with saved outputs) must be downloaded from Colab and saved as
notebooks/week6_audit_executed.ipynb; that is the graded notebook.
"""
import json
import os

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(True)})


md("""
# Week 6: Red team the claim, audit the model

Audit of one QLoRA-adapted `Qwen/Qwen2.5-0.5B-Instruct` model that drafts structured appeal summaries
from synthetic denial notices, framed as a health insurer's appeals-drafting pilot. Everything is
synthetic (fictional members, fictional `MRP-####` policies). Set the runtime to **T4 GPU** before running.

**Evidence rules.** Every number below is read from `results/` or `logs/` produced by a run in this
notebook, or by the CPU-side build (contamination sweep, dataset audit). Week 3 findings are prior work
and are not Week 6 evidence. The plan this notebook follows is `analysis_plan.md`, frozen and hashed in
`MANIFEST.json` before any model output existed.
""")

md("## 0. Setup")
code("""
import os
REPO = "https://github.com/sweiss1367/WK5_Architecture-Variants"
BRANCH = "claude/practical-pasteur-l1mibx"
if not os.path.exists("/content/WK5_Architecture-Variants"):
    !git clone -q -b {BRANCH} {REPO} /content/WK5_Architecture-Variants
else:
    !git -C /content/WK5_Architecture-Variants pull -q origin {BRANCH}
%cd /content/WK5_Architecture-Variants/week6
""")
code("""
!pip install -q -U transformers peft accelerate bitsandbytes
""")
code("""
import sys, json, pandas as pd
sys.path.insert(0, "src")
pd.set_option("display.width", 220, "display.max_colwidth", 80, "display.max_rows", 200)
def show_csv(path, **kw):
    df = pd.read_csv(path, **kw); display(df); return df
""")

md("""
## 1. E0: preflight and frozen-manifest check
Verifies every frozen file's SHA-256 against `MANIFEST.json`, records exact package and GPU versions, and
prints one real prompt. Output: `results/env_manifest.json`, `logs/pip_freeze.txt`.
""")
code("""
!python src/preflight.py
""")

md("""
## 2. CPU-side evidence (already produced before any GPU run)
### 2a. Datasets and structural audit
""")
code("""
print(json.dumps(json.load(open("data/dataset_summary.json")), indent=1)[:2500])
audit = json.load(open("data/dataset_audit.json"))
print("all hard checks passed:", audit["all_hard_checks_passed"])
pd.DataFrame([{"check": k, "passed": v["passed"]} for k, v in audit["checks"].items() if isinstance(v, dict) and "passed" in v])
""")
md("### 2b. Frozen hashes")
code("""
man = json.load(open("MANIFEST.json"))
print("manifest created (UTC):", man["created_utc"])
pd.DataFrame([{"file": k, "sha256": v["sha256"][:16] + "...", "bytes": v["bytes"]} for k, v in man["files"].items()])
""")
md("""
### 2c. n-gram contamination sweep (n = 8, 13, 20)
Evaluation text vs. training + validation text, after normalising case, punctuation and whitespace. Every
match is in `results/contamination_matches.csv`. **A negative n-gram result does not rule out paraphrase,
translation, or structural/template contamination, nor contamination of the base model's pretraining data.**
The n=8 hits are all the intentionally shared synthetic policy vocabulary (MRP identifier + "Service Coverage
Policy" wording), classified as expected shared-domain vocabulary, not confirmed leakage. Zero matches at n=13 and
n=20 do not prove the evaluation set is uncontaminated. The positive control is a synthetic record with a training
summary copied in; it is not naturally occurring contamination and is excluded from the percentages.
""")
code("""
s = show_csv("results/contamination_summary.csv")
""")
code("""
rev = json.load(open("results/contamination_review.json"))
print({k: rev[k] for k in ("n_distinct_spans_read", "n_rows_verified", "n_rows_unexplained")})
show_csv("results/contamination_unique_spans.csv")
""")
code("""
m = pd.read_csv("results/contamination_matches.csv", keep_default_na=False)
display(m[m.is_synthetic_control.astype(str) == "True"][["dataset", "item_id", "field", "n", "span_tokens", "verdict"]])
""")

md("""
## 3. E3: train the one QLoRA adapter
`--smoke` first (2 steps, API check only, never evidence), then the real run (about 75 optimizer steps).
Outputs: `outputs/qlora_adapter/`, `logs/experiment_log_train.json`, `logs/trainer_log_history.json`,
`logs/token_length_audit.json`, `logs/adapter_sha256.json`.
""")
code("""
!python src/train_qlora.py --smoke
""")
code("""
!python src/train_qlora.py
""")
code("""
print(json.dumps(json.load(open("logs/experiment_log_train.json")), indent=1))
hist = pd.DataFrame(json.load(open("logs/trainer_log_history.json")))
display(hist[[c for c in ("epoch", "step", "loss", "eval_loss", "grad_norm") if c in hist.columns]])
print(json.dumps(json.load(open("logs/token_length_audit.json")), indent=1))
""")
code("""
# Optional: keep the adapter if the Colab session disconnects (re-training is the alternative).
from google.colab import drive
drive.mount("/content/drive")
!mkdir -p /content/drive/MyDrive/week6_adapter && cp -r outputs/qlora_adapter /content/drive/MyDrive/week6_adapter/ && cp logs/adapter_sha256.json /content/drive/MyDrive/week6_adapter/
""")

md("""
## 4. E4: task baseline (greedy; context for the audit, not a headline)
`val` uses the training templates (in-template reference). `eval_h` uses held-out templates and vocabulary.
Outputs: `results/generations/{val,eval_h}_greedy.jsonl`, `results/task_scores_*.csv`, `results/task_baseline.csv`.
""")
code("""
!python src/generate_batched.py --stage task --smoke
""")
code("""
!python src/generate_batched.py --stage task
!python src/run_task_eval.py --item val=data/val.jsonl:results/generations/val_greedy.jsonl --item eval_h=data/eval_h.jsonl:results/generations/eval_h_greedy.jsonl
""")

md("""
## 5. E5: safety probe suite (100 probes, greedy + 3 sampled seeds)
Counts per family, not rates (20 probes per family is too few for rates). Every probe's ID, failure
hypothesis, expected behavior and scoring rule is in `probes/probe_catalog.csv`.
Outputs: `results/generations/probes_*.jsonl`, `results/probe_*.csv`, `results/probe_failure_transcripts.jsonl`.
""")
code("""
!python src/generate_batched.py --stage probes
""")
code("""
!python src/score_probes.py --gen greedy=results/generations/probes_greedy.jsonl --gen seed1001=results/generations/probes_seed1001.jsonl --gen seed1002=results/generations/probes_seed1002.jsonl --gen seed1003=results/generations/probes_seed1003.jsonl
""")
code("""
show_csv("results/probe_summary_by_family.csv")
by_probe = show_csv("results/probe_summary_by_probe.csv")
""")
code("""
# Every failing probe, with the model's raw output, for manual reading.
fails = [json.loads(l) for l in open("results/probe_failure_transcripts.jsonl")]
print(len(fails), "failing (probe, run) pairs")
for f in fails[:40]:
    print("=" * 90); print(f["probe_id"], f["run"], f["reasons"]); print(f["raw_output"][:700])
""")

md("""
## 6. E6: fairness (150 matched cases x {English, Spanish} x {45, 75} x 3 sampling seeds)
Only the structured header values `Primary language` and `Age` differ inside a case; every word of the notice
body and clinical summary is identical. The three seeds are sampling seeds of one adapter: they measure
**generation variability, not training-seed variability**.

Interpretation rule (fixed in advance): if the 95% CI of the paired slice difference excludes zero, evidence
of a difference; if it includes zero, unresolved / not statistically established, which is **not** evidence
that the groups perform equally. Point estimates are always shown with CIs and sample sizes.
Outputs: `results/generations/fairness_seed*.jsonl`, `results/fairness_*.csv`.
""")
code("""
!python src/generate_batched.py --stage fairness
""")
code("""
!python src/analyze_fairness.py --gen seed1001=results/generations/fairness_seed1001.jsonl --gen seed1002=results/generations/fairness_seed1002.jsonl --gen seed1003=results/generations/fairness_seed1003.jsonl
""")
code("""
dis = show_csv("results/fairness_disaggregated.csv")
""")
code("""
dif = pd.read_csv("results/fairness_slice_differences.csv")
display(dif[dif.stratum == "marginal"][["metric", "role", "slice", "contrast", "estimate_pp", "ci_low_pp", "ci_high_pp", "n_cases", "n_generations_a", "n_generations_b", "per_seed_estimates", "interpretation"]])
""")
code("""
display(dif[dif.stratum != "marginal"][["metric", "slice", "contrast", "stratum", "estimate_pp", "ci_low_pp", "ci_high_pp", "interpretation"]])
""")

md("## 7. Figures (drawn only from results/)")
code("""
!python src/make_figures.py
from IPython.display import Image, display
for f in ("contamination_flagged", "fairness_slice_differences", "probe_failures_by_family"):
    display(Image(f"results/figures/{f}.png"))
""")

md("""
## 8. Package the evidence
Download the zip and the executed notebook (File > Download > .ipynb, **with outputs**), and save the
notebook as `notebooks/week6_audit_executed.ipynb` in the repo.
""")
code("""
!zip -q -r week6_evidence.zip results logs data/dataset_audit.json MANIFEST.json -x "results/generations_smoke/*" "logs_smoke/*"
from google.colab import files
files.download("week6_evidence.zip")
""")

nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell{i:03d}"

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "week6_audit.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, f"({len(CELLS)} cells)")
