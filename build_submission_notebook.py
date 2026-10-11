"""Builds the single notebook to submit for Week 5.

Base: bol_experiment_cuda_run_v4_benchmark.ipynb, the Colab Tesla T4 run (Run all, one session). Cells 0-48 are kept exactly as
Colab saved them (sources and outputs), except that the markdown in cell 0 is replaced by a guide to the whole notebook.

Appended: the two sections written after that run (secondary memory diagnostics, scaling-law estimates), taken from
bol_experiment.ipynb. They are executed here, outside Colab, because they need no GPU and no training or benchmarking:
  * memory diagnostics read the real results/benchmark_meta.json produced by the Colab run;
  * the scaling-law cell needs four results JSON files that lived in the Colab session; they are rebuilt from the numbers
    printed in this notebook's own outputs (parameter counts, token counts, epochs, class counts, teacher-logit seconds).
The appended markdown says so. Nothing in cells 0-48 is re-run, edited or fabricated.
"""
import contextlib, copy, io, json, os, re, shutil, sys, tempfile
import nbformat

BASE = "bol_experiment_cuda_run_v4_benchmark.ipynb"
SRC = "bol_experiment.ipynb"
OUT = os.path.join("submission", "bol_experiment_cuda_run_v4_benchmark.ipynb")

base = json.load(open(BASE))
src = json.load(open(SRC))
cells = copy.deepcopy(base["cells"])
assert len(cells) == 49 and len(src["cells"]) == 55
text = lambda c: "".join(c["source"])
def out_text(i):
    return "\n".join("".join(o["text"]) if "text" in o else str(o.get("data", {}).get("text/plain", "")) for o in cells[i].get("outputs", []))
for i in range(49):                      # appended cells must continue the very same notebook
    assert text(cells[i]).strip() == text(src["cells"][i]).strip() or i == 0, i

# ---------------------------------------------------------------------------------------------------------------------
# 1. Guide in cell 0 (markdown only; cell positions do not move)
# ---------------------------------------------------------------------------------------------------------------------
GUIDE = """# Brightline Logistics model comparison: benchmarking notebook (Week 5)
One notebook, executed top to bottom (Runtime > Run all) in a single Google Colab session on a **Tesla T4** (CUDA 13.0, PyTorch 2.11.0+cu130, bitsandbytes 0.50.2, Python 3.13.15). The outputs shown for cells 0 to 48 are exactly what that run produced. Data is a synthetic bill-of-lading workload, and nothing here was measured on Android tablet hardware.

**What the four configurations are called**

| In this notebook | In the Excel workbook |
|---|---|
| FP32 baseline Transformer | FP32 Baseline Transformer |
| FP32 linear-attention variant | FP32 Linear Attention |
| INT8 hidden layers + FP32 head (bitsandbytes) / mixed-precision INT8 | Mixed-Precision INT8 Baseline |
| FP32 distilled student | Distilled Student |

**Where things are** (cell numbers count from 0; the workbook's source column uses the same numbering)

| Topic | Where | Workbook sheet |
|---|---|---|
| Dataset and setup: environment check, synthetic data, 70/10/20 split, label noise, SHA-256, tokenizer, fixed seeds | Part 1, headings 0 to 2 (cells 2 to 7) | Methodology |
| Baseline Transformer: model, training (seeds 0, 1, 2), report | Part 1, headings 3 to 5 (cells 8 to 14) | Quality by Seed, Per-Class Results |
| Linear attention variant: implementation, padding tests, training, comparison | Part 2 (cells 15 to 25) | Quality by Seed, Per-Class Results |
| INT8 quantization: bitsandbytes support check (the failed 128 -> 6 head), quantization, storage, logit check | Part 3 (cells 26 to 32) | Methodology, Trade-off Board |
| Distilled student: architecture, distillation loss (T = 2), training, report | Part 4 (cells 33 to 40) | Quality by Seed, Per-Class Results |
| Quality evaluation: three-seed tables and checks for every configuration | the report and check cells of Parts 1 to 4 (cells 14, 23, 31, 38) | Quality by Seed |
| Latency and memory benchmarking: batch 1, 15 warm-ups, 100 timed trials, CUDA synchronization, peak memory | Part 5 (cells 41 to 44) | Benchmark Detail, Raw Benchmark Runs |
| Benchmark repeatability: baseline run twice, 15% threshold | Part 5 results cell (cell 46) | Benchmark Detail |
| Final benchmark summary and verification checks | Part 5 (cells 45 to 48) | Trade-off Board |
| Secondary memory diagnostics | cells 49 and 50 | Benchmark Detail |
| Scaling inputs and notes (6ND, 20N, 10x) | Part 6 (cells 51 to 54) | Scaling Law |

**Things to know before reading the outputs**
- The per-trial latencies (500 rows) are written to `results/benchmark_raw_latency.csv` during Part 5 and are not printed; they are the workbook's Raw Benchmark Runs sheet. Checkpoints, JSON results and CSV files are written to the `results/` folder of the Colab session.
- Cells 49 to 54 were added after the Colab run and were executed outside Colab (no GPU, no training, no benchmarking); each has a note explaining its input. Everything else is the original Colab output.
- The baseline training cell prints a PyTorch warning that the softmax-attention backward pass is non-deterministic on CUDA. It is kept on purpose: GPU runs are not guaranteed to be bitwise identical.
- Quality values are means of three seeds. Latency and memory use the seed-0 models (same architecture, tensor shapes, precision and execution path across seeds).
- The workbook's 'bol_experiment.ipynb, Part N' references mean the Part N section of this notebook (same code)."""
cells[0]["source"] = GUIDE.splitlines(True)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Appended cells 49-54
# ---------------------------------------------------------------------------------------------------------------------
new = copy.deepcopy(src["cells"][49:55])
assert [c["cell_type"] for c in new] == ["markdown", "code", "markdown", "code", "markdown", "code"]
new[0]["source"] = (text(new[0]) + "\n\n**Provenance of the output below:** this cell reads the real `results/benchmark_meta.json` written by the Colab benchmark run in Part 5. "
                    "It was executed outside Colab after that run because it only reads that file. No benchmark was re-run.").splitlines(True)
new[2]["source"] = (text(new[2]) + "\n\n**Provenance of the output below:** this section was added after the Colab run and was executed outside Colab. "
                    "The four results JSON files it normally reads (`baseline_results_cuda.json`, `linear_results_cuda.json`, `int8_mixed_results_cuda.json`, `student_results_cuda.json`) "
                    "were no longer available, so they were rebuilt from numbers printed by this notebook's own cells: parameter counts (cell 34 output), "
                    "10,500 training documents and 457,666 non-padding tokens per epoch (cell 7 output), class counts (cell 5 output), 8 epochs and length 128 (cells 5 and 11), "
                    "and the teacher-logit caching seconds (cell 38 output, printed to one decimal). The arithmetic uses integers only, so the values do not depend on where the cell runs. "
                    "No model was trained or benchmarked. Running this cell at the end of a full Colab run recomputes the same values from the saved JSON files.").splitlines(True)

# --- rebuild the scaling-law input files from this notebook's printed outputs --------------------------------------------
t34, t7, t5, t38 = out_text(34), out_text(7), out_text(5), out_text(38)
num = lambda pat, t: int(re.search(pat, t).group(1).replace(",", ""))
base_total = num(r"Baseline total parameters\s*:\s*([\d,]+)", t34); base_non = num(r"Baseline non-embedding parameters\s*:\s*([\d,]+)", t34)
stu_total = num(r"Student total parameters\s*:\s*([\d,]+)", t34); stu_non = num(r"Student non-embedding parameters\s*:\s*([\d,]+)", t34)
counts = eval(re.search(r"class counts per split (\{.*\})", t5).group(1))
tok = json.loads(re.search(r"token stats: (\{.*?\n\})", t7, re.S).group(1))["train"]
epochs = int(re.search(r"EPOCHS, BATCH, LR, WD, WARMUP = (\d+),", text(cells[11])).group(1))
max_len = int(re.search(r"MAX_LEN = (\d+)", text(cells[5])).group(1))
teach = [float(x) for x in re.search(r"teacher-logit caching \[([\d., ]+)\]", t38).group(1).split(",")]
assert tok["n_docs"] == sum(counts["train"]) and len(teach) == 3
P = lambda tot, non: {"total": tot, "embedding": tot - non, "non_embedding": non}
bp, sp = P(base_total, base_non), P(stu_total, stu_non)
cfg = {"epochs": epochs, "max_len": max_len}
seeds = lambda p, **kw: {f"seed{s}": {"params": p, **{k: v[s] for k, v in kw.items()}} for s in range(3)}

work = tempfile.mkdtemp()
os.makedirs(os.path.join(work, "results"))
shutil.copy("results/benchmark_meta.json", os.path.join(work, "results", "benchmark_meta.json"))
dump = lambda name, obj: json.dump(obj, open(os.path.join(work, "results", name), "w"))
dump("baseline_results_cuda.json", {"config": cfg, "token_stats": {"train": {"n_docs": tok["n_docs"], "nonpad_tokens": tok["nonpad_tokens"]}}, "class_counts": counts, "baseline": seeds(bp)})
dump("linear_results_cuda.json", {"config": cfg, "linear": seeds(bp)})
dump("int8_mixed_results_cuda.json", {"int8_mixed": seeds(bp)})
dump("student_results_cuda.json", {"config": cfg, "student": seeds(sp, teacher_logit_seconds=teach)})

# --- execute the appended code cells (shared namespace, in order) and capture their printed output ------------------------
prelude = "import json, os\nimport numpy as np\nSEEDS = [0, 1, 2]\nOUT_DIR = 'results'\n"
ns = {}
cwd = os.getcwd(); os.chdir(work)
try:
    exec(prelude, ns)
    next_count = max(c.get("execution_count") or 0 for c in cells if c["cell_type"] == "code") + 1
    for c in new:
        if c["cell_type"] != "code":
            continue
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec("".join(c["source"]), ns)
        c["outputs"] = [nbformat.v4.new_output("stream", name="stdout", text=buf.getvalue())]
        c["execution_count"] = next_count; next_count += 1
finally:
    os.chdir(cwd)

# the generated files must be identical to the ones the workbook was built from
cmp = [("scaling_law_results.csv", "results/scaling_law_results.csv"), ("benchmark_memory_diagnostics.csv", "results/benchmark_memory_diagnostics.csv")]
for gen, ref in cmp:
    a, b = open(os.path.join(work, "results", gen), "rb").read(), open(ref, "rb").read()
    assert a == b, f"{gen} differs from {ref}"
sj = json.load(open(os.path.join(work, "results", "scaling_law_results.json"))); rj = json.load(open("results/scaling_law_results.json"))
assert sj["rows"] == rj["rows"] and sj["workload"] == rj["workload"] and sj["teacher_cost"] == rj["teacher_cost"]
print("appended outputs reproduce results/scaling_law_results.csv, .json rows, and benchmark_memory_diagnostics.csv exactly")

# ---------------------------------------------------------------------------------------------------------------------
nbdict = {"cells": cells + new, "metadata": base["metadata"], "nbformat": base["nbformat"], "nbformat_minor": base["nbformat_minor"]}
import uuid
used = {c["id"] for c in nbdict["cells"] if c.get("id")}
for c in nbdict["cells"]:                      # nbformat 4.5 expects a unique id on every cell (the appended cells had none)
    if not c.get("id"):
        c["id"] = uuid.uuid4().hex[:8]
        while c["id"] in used: c["id"] = uuid.uuid4().hex[:8]
        used.add(c["id"])
nb = nbformat.from_dict(nbdict)
nbformat.validate(nb)
os.makedirs("submission", exist_ok=True)
nbformat.write(nb, OUT)
print("wrote", OUT, "with", len(nb.cells), "cells")
shutil.rmtree(work)
