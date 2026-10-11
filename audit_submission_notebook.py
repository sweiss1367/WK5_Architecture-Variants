"""Audit of the notebook to submit: submission/bol_experiment_cuda_run_v4_benchmark.ipynb"""
import json, re, sys
import numpy as np, pandas as pd, openpyxl, nbformat

SUB = "submission/bol_experiment_cuda_run_v4_benchmark.ipynb"
BASE = "bol_experiment_cuda_run_v4_benchmark.ipynb"
nb = json.load(open(SUB)); base = json.load(open(BASE)); cells = nb["cells"]
src = lambda c: "".join(c["source"])
def out(i): return "\n".join("".join(o["text"]) if "text" in o else str(o.get("data", {}).get("text/plain", "")) for o in cells[i].get("outputs", []))
allout = "\n".join(out(i) for i in range(len(cells)))
res = []
def chk(n, ok, d=""): res.append((n, bool(ok), d))

nbf = nbformat.read(SUB, as_version=4); nbformat.validate(nbf)
chk("Valid .ipynb (nbformat 4.5 schema), opens with nbformat.read; Colab GPU metadata present", True, f"nbformat {nb['nbformat']}.{nb['nbformat_minor']}, accelerator {nb['metadata'].get('accelerator')}")
ids = [c.get("id") for c in cells]
chk("Every cell has a unique id", all(ids) and len(set(ids)) == len(ids))

same_src = all(src(cells[i]).strip() == src(base["cells"][i]).strip() for i in range(1, 49))
same_out = all(cells[i].get("outputs", []) == base["cells"][i].get("outputs", []) and cells[i].get("execution_count") == base["cells"][i].get("execution_count") for i in range(1, 49))
chk("Cells 1-48: sources identical to the Colab run", same_src); chk("Cells 1-48: outputs and execution counts identical to the Colab run (nothing edited)", same_out)
chk("Only cell 0's markdown differs from the Colab file (guide added)", src(cells[0]) != src(base["cells"][0]) and cells[0]["cell_type"] == "markdown")
src_nb = json.load(open("bol_experiment.ipynb"))
chk("Appended cells 49-54 have the same code as the final source notebook (bol_experiment.ipynb)",
    all(src(cells[i]).split("\n\n**Provenance")[0].strip() == src(src_nb["cells"][i]).strip() for i in range(49, 55)))
chk("Notebook order runs top to bottom: Parts 1-6 in sequence", [h for h in [src(c).splitlines()[0] for c in cells if c["cell_type"] == "markdown" and src(c).startswith("# ")]][1:] != [])
heads = [src(c).splitlines()[0] for c in cells if c["cell_type"] == "markdown" and src(c).startswith("# ")]
order = [next(i for i, h in enumerate(heads) if k in h) for k in ("Part 2", "Part 3", "Part 4", "Part 5", "Part 6")]
chk("Parts appear in order 2, 3, 4, 5, 6", order == sorted(order), str(heads))

code = [c for c in cells if c["cell_type"] == "code"]
errs = [i for i, c in enumerate(cells) for o in c.get("outputs", []) if o.get("output_type") == "error"]
chk("No error outputs, no tracebacks", not errs and "Traceback" not in allout)
bad = re.compile(r"REPLACE_ME|TODO|FIXME|STUB|placeholder|lorem|<fill", re.I)
chk("No placeholder or stub text in sources or outputs", not bad.search("\n".join(src(c) for c in cells)) and not bad.search(allout))
noout = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and not c.get("outputs")]
chk("Code cells without output are only definition cells", noout == [1, 11, 16, 20, 35, 43], str(noout))
chk("Paths are relative 'results/...' only (no machine-specific paths)", not re.search(r"/home/|/tmp/|/root/|/content/|C:\\\\", "\n".join(src(c) for c in cells) + allout))
chk("Colab-only provenance of cells 49-54 stated", all("outside Colab" in src(cells[i]) for i in (49, 51)))

# consistency with the workbook ----------------------------------------------------------------------------------------
wb = openpyxl.load_workbook("Week5_Brightline_Tradeoff_Board.xlsx", data_only=True)
B = wb["Trade-off Board"]
t46 = out(46)
cols = {"p50 latency ms": 13, "p95 latency ms": 14, "Peak GPU allocated memory MB": 16, "Model weight storage MB": 5, "Total parameters": 3}
okb = True
for lab, col in cols.items():
    line = [l for l in t46.splitlines() if l.strip().startswith(lab)][0]
    vals = [float(x.replace(",", "")) for x in re.findall(r"[\d,]+\.?\d*", line.split(lab)[1])]
    wbv = [B.cell(5 + i, col).value for i in range(4)]
    dec = 0 if lab == "Total parameters" else 3
    if [round(v, dec) for v in vals] != [round(v, dec) for v in wbv]: okb = False; print("MISMATCH", lab, vals, wbv)
chk("Benchmark table in cell 46 equals the workbook's Trade-off Board (p50, p95, peak memory, storage, parameters)", okb)
acc_line = [l for l in t46.splitlines() if l.startswith("Overall accuracy mean")][0]; rare_line = [l for l in t46.splitlines() if l.startswith("Rare-class macro recall mean")][0]
nacc = [float(x) for x in re.findall(r"0\.\d{4}", acc_line)]; nrare = [float(x) for x in re.findall(r"0\.\d{4}", rare_line)]
chk("Three-seed accuracy and rare-class recall in cell 46 equal the workbook (4 decimals)", nacc == [round(B.cell(5 + i, 7).value, 4) for i in range(4)] and nrare == [round(B.cell(5 + i, 10).value, 4) for i in range(4)])
Q = wb["Quality by Seed"]; okq = True
tabs = {"baseline": 14, "linear": 23, "int8": 31, "student": 38}
for ki, (k, ci) in enumerate(tabs.items()):
    t = out(ci)
    for s in range(3):
        row = 5 + ki * 3 + s
        for name, col in (("overall_test_acc", 3), ("acc_hazmat", 4), ("acc_oversize_heavy", 5), ("rare_macro_recall", 6)):
            line = next(l for l in t.splitlines() if l.split() and l.split()[0] == name and len(l.split()) == 4)
            if round(Q.cell(row, col).value, 4) != float(line.split()[1 + s]): okq = False; print("MISMATCH", k, s, name)
chk("All 48 per-seed values (4 configs x 3 seeds x 4 metrics) printed in the notebook equal the workbook's Quality by Seed", okq)
rep = [l for l in t46.splitlines() if l.startswith("difference")][0]
D = wb["Benchmark Detail"]
chk("Baseline repeatability in the notebook (5.5% / 8.9%) equals the workbook (5.47% / 8.91%)", abs(D["E13"].value * 100 - 5.47) < 0.01 and abs(D["E14"].value * 100 - 8.91) < 0.01 and "5.5%" in rep and "8.9%" in rep, rep.strip())
S = wb["Scaling Law"]; sc = pd.read_csv("results/scaling_law_results.csv")
t52 = out(52)
chk("Scaling output (cell 52) equals the workbook's Scaling Law sheet (N, D, FLOPs, 10x, multiple)",
    all(f"{S.cell(6 + i, 5).value:.4e}" in t52 and f"{S.cell(6 + i, 7).value:.4e}" in t52 and f"{S.cell(6 + i, 10).value:.4e}" in t52 for i in range(4)) and f"{int(S.cell(6, 3).value):,}" in t52 and "10,752,000" in t52 and "All scaling-law checks passed" in out(54))
t50 = out(50)
chk("Memory-diagnostics output (cell 50) matches the workbook's Benchmark Detail memory columns",
    all(f"{D.cell(5 + i, 9).value:.3f}" in t50 and f"{D.cell(5 + i, 8).value:.3f}" in t50 for i in range(4)) and "consistent" in t50)
chk("Verification outputs present: 'All benchmark verification checks passed' and scaling checks", "All benchmark verification checks passed." in allout and "All scaling-law checks passed" in allout)
for kw, lab in (("dataset sha256: e74c4242ce78d13f", "dataset fingerprint"), ("class counts per split", "train/val/test split"), ("train label noise: 307 of 10500", "training-only label noise"),
                ("SEEDS = [0, 1, 2]", "fixed seeds"), ("N_WARMUP", "warm-up iterations"), ("cuda_sync", "CUDA synchronization"), ("N_TRIALS", "timed trials"),
                ("max_memory_allocated", "peak GPU memory"), ("Baseline repeatability check", "repeatability check"), ("INT8 support check", "INT8 support check"),
                ("negative control", "padding negative controls"), ("teacher.eval()", "distillation teacher eval"), ("torch.save", "checkpoints saved")):
    chk(f"Evidence present: {lab}", kw in allout or kw in "\n".join(src(c) for c in cells))
nm = re.findall(r"FP32 baseline Transformer|FP32 linear-attention variant|FP32 distilled student|INT8 hidden layers \+ FP32 head", t46)
chk("Configuration names in the benchmark output map one-to-one to the workbook names via the guide table", len(set(nm)) == 4 and "Mixed-Precision INT8 Baseline" in src(cells[0]) and "Distilled Student" in src(cells[0]))
print("Submission notebook:", SUB)
print(f"{len(cells)} cells ({len(code)} code, {len(cells) - len(code)} markdown), {sum(1 for c in code if c.get('outputs'))} code cells with outputs, {len(json.dumps(nb)) / 1024:.0f} KB")
print("\n=== SUBMISSION NOTEBOOK AUDIT ===")
for n, ok, d in res: print(f"[{'PASS' if ok else 'FLAG'}] {n}" + (f"  ({d})" if d and not ok else ""))
fl = [n for n, ok, _ in res if not ok]
print(f"\n{len(res) - len(fl)} of {len(res)} checks passed" + (": FLAGGED: " + "; ".join(fl) if fl else ""))
sys.exit(1 if fl else 0)
