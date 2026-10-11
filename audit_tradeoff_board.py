"""Final requirement audit of Week5_Brightline_Tradeoff_Board.xlsx (reads the saved workbook, not the builder's variables)."""
import re, sys
import numpy as np, pandas as pd, openpyxl

PATH = "Week5_Brightline_Tradeoff_Board.xlsx"
wf = openpyxl.load_workbook(PATH)                 # formulas
wv = openpyxl.load_workbook(PATH, data_only=True) # cached values
res = []
def chk(name, ok, detail=""): res.append((name, bool(ok), detail))
def alltext(ws): return " ".join(str(c.value) for row in ws.iter_rows() for c in row if c.value is not None)

names = ["Trade-off Board", "Quality by Seed", "Per-Class Results", "Raw Benchmark Runs", "Benchmark Detail", "Scaling Law", "Methodology", "Recommendation"]
chk("Eight sheets with the requested names, in order", wf.sheetnames == names, ", ".join(wf.sheetnames))

B = wv["Trade-off Board"]
spec_cols = ["Configuration", "Precision / Compression", "Total Parameters", "Parameter Reduction vs Baseline", "Weight Storage MB", "Storage Reduction vs Baseline",
             "Overall Accuracy Mean", "Accuracy Std Dev", "Accuracy Change vs Baseline", "Rare-Class Macro Recall", "Rare-Class Std Dev", "Rare-Class Change vs Baseline",
             "p50 Latency ms", "p95 Latency ms", "p95 Change vs Baseline", "Peak GPU Memory MB", "Peak Memory Change vs Baseline", "Main Advantage", "Main Trade-off", "Residual Risk", "Final Rank"]
hdr = [B.cell(4, c).value for c in range(1, 22)]
chk("Trade-off Board: title, subtitle and the 21 requested columns", B["A1"].value == "Brightline Logistics Model Deployment Trade-Off Board"
    and B["A2"].value == "Measured on one Google Colab Tesla T4 using a controlled synthetic bill-of-lading classification workload" and hdr == spec_cols)
cfg = [B.cell(r, 1).value for r in range(5, 9)]
chk("Four configurations: baseline, architecture variant, quantized, distilled student",
    cfg == ["FP32 Baseline Transformer", "FP32 Linear Attention", "Mixed-Precision INT8 Baseline", "Distilled Student"], "; ".join(cfg))
chk("INT8 described accurately (INT8 hidden Linear layers, FP32 128->6 head)", "128 -> 6" in str(B.cell(7, 2).value) and "FP32" in str(B.cell(7, 2).value) and "INT8" in str(B.cell(7, 2).value))
ranks = [B.cell(r, 21).value for r in range(5, 9)]
chk("Final Rank filled with 1-4 once each", sorted(ranks) == [1, 2, 3, 4], str(ranks))
chk("Board freeze panes and conditional formatting present", wf["Trade-off Board"].freeze_panes == "B5" and len(wf["Trade-off Board"].conditional_formatting) > 0, wf["Trade-off Board"].freeze_panes)

Q = wv["Quality by Seed"]
seed_rows = [r for r in range(1, Q.max_row + 1) if Q.cell(r, 2).value in (0, 1, 2) and Q.cell(r, 1).value in cfg]
chk("Quality by Seed: 3 seeds x 4 configurations", len(seed_rows) == 12, f"{len(seed_rows)} rows")
stat_rows = [r for r in range(1, Q.max_row + 1) if Q.cell(r, 2).value in ("Mean",) or str(Q.cell(r, 2).value).startswith("Sample Std")]
chk("Mean and sample standard deviation rows below the seed rows", len(stat_rows) == 4 + 4 and min(stat_rows) > max(seed_rows))
qtxt = alltext(Q)
chk("Seed-spread comparison and 'no significance test' statement present", "Smaller than the seed spread" in qtxt and "no significance test" in qtxt.lower())

C = wv["Per-Class Results"]
rows_c = [r for r in range(5, 29)]
classes = {C.cell(r, 2).value for r in rows_c}
chk("Per-Class Results: all six classes x four configurations", classes == {"General Dry", "Retail Pallet", "Refrigerated", "Fragile / High-Value", "Hazmat", "Oversize / Heavy Haul"} and len(rows_c) == 24)
chk("Rare classes visually distinguished (shaded)", all(wf["Per-Class Results"].cell(r, 2).fill.fgColor.rgb not in (None, "00000000") and
    wf["Per-Class Results"].cell(r, 2).fill.fgColor.rgb.endswith("FFF2CC") for r in rows_c if C.cell(r, 2).value in ("Hazmat", "Oversize / Heavy Haul")))

R = wv["Raw Benchmark Runs"]
raw = pd.DataFrame([[R.cell(r, c).value for c in range(1, 7)] for r in range(5, R.max_row + 1)], columns=["cfg", "run", "trial", "idx", "lat", "pos"])
chk("Raw Benchmark Runs: 100 trials per configuration plus both baseline runs", len(raw) == 500 and raw.groupby(["cfg", "run"]).size().eq(100).all()
    and set(raw[raw.cfg == "FP32 Baseline Transformer"].run) == {1, 2})

D = wv["Benchmark Detail"]
dh = [D.cell(4, c).value for c in range(1, 16)]
need = ["p50 Latency ms", "p95 Latency ms", "Mean Latency ms", "Minimum Latency ms", "Maximum Latency ms", "Resident GPU Memory MB", "Peak GPU Memory MB (primary)",
        "Incremental Peak Memory MB", "Weight Storage MB", "Total Parameters", "Non-Embedding Parameters"]
dtxt = alltext(D)
chk("Benchmark Detail columns present", all(n in dh for n in need))
chk("Baseline repeatability shown with % differences and 15% statement", "stability threshold" in dtxt and "5.47%" in dtxt and "8.91%" in dtxt)

S = wv["Scaling Law"]
sh = [S.cell(5, c).value for c in range(1, 13)]
stxt = alltext(S)
chk("Scaling Law columns (N, tokens, FLOPs, 20N, 10x, multiple, notes)", all(any(k in str(h) for h in sh) for k in ["Non-Embedding Parameters N", "Actual Training Tokens", "Estimated Actual Training FLOPs",
    "Compute-Optimal Tokens", "Compute-Optimal Training FLOPs", "10x Non-Embedding", "10x Compute-Optimal Tokens", "10x Training FLOPs", "Compute Multiple", "Notes"]))
chk("Scaling Law distinguishes measured from estimated and carries all required notes", "NOT MEASUREMENTS" in stxt and all(k in stxt for k in [
    "does not reduce the original baseline training FLOPs", "does not include teacher-inference cost", "10x projection is an estimate", "GPU utilization", "single-GPU memory",
    "Gradient checkpointing", "attention cost", "communication overhead", "rule of thumb"]))

M = wv["Methodology"]; mtxt = alltext(M)
req_terms = {"15,000 documents": "dataset size", "10,500 train": "split", "Hazmat (4%)": "rare-class proportions", "Training-only label noise": "label noise", "OCR": "OCR corruption",
             "missing": "missing-field corruption", "SHA-256": "fingerprint", "Vocabulary size 5,565": "vocabulary", "Fixed sequence length 128": "length", "T = 2": "temperature",
             "0.5 x cross-entropy": "loss weights", "AdamW": "optimizer", "learning rate 1e-3": "learning rate", "weight decay 0.01": "weight decay", "batch size 64": "batch size",
             "8 epochs": "epochs", "seeds 0, 1, 2": "seeds", "Final-epoch": "evaluation policy", "Tesla T4": "GPU", "PyTorch 2.11.0+cu130": "PyTorch", "CUDA 13.0": "CUDA",
             "bitsandbytes 0.50.2": "bitsandbytes", "Batch size 1": "latency batch", "15 untimed warm-up": "warm-up", "100 timed trials": "timed trials",
             "synchronize": "CUDA sync", "reset_peak_memory_stats": "peak-memory method", "repeatability": "repeatability test", "Known limitations": "limitations section"}
missing_terms = [v for k, v in req_terms.items() if k.lower() not in mtxt.lower()]
chk("Methodology covers every requested item", not missing_terms, "missing: " + ", ".join(missing_terms) if missing_terms else f"{len(req_terms)} items found")
lims = ["Synthetic data may not represent real Brightline", "Only one GPU type", "Shared Colab hardware", "bitwise deterministic", "Sequence length was fixed at 128",
        "Linear attention may have different relative performance", "INT8 used a mixed-precision", "Rare classes had smaller test support", "Scaling-law results are formula-based estimates"]
chk("All nine required limitations stated", all(l.lower() in mtxt.lower() for l in lims))

X = wv["Recommendation"]
rk = [(X.cell(r, 1).value, X.cell(r, 2).value) for r in range(6, 10)]
chk("Recommendation: four ranked configurations with why / gains / gives up / risk", [a for a, _ in rk] == [1, 2, 3, 4] and all(X.cell(r, c).value for r in range(6, 10) for c in range(3, 7)), str(rk))
fin = next(X.cell(r, 1).value for r in range(1, X.max_row + 1) if str(X.cell(r, 1).value).startswith("Ship "))
sent = re.split(r"(?<=[.)])\s+(?=[A-Z])", fin)
chk("Final recommendation is at most three sentences and names model, evidence and residual risk", len(sent) <= 3 and "Distilled Student" in fin and "residual risk" in fin.lower(), f"{len(sent)} sentences")
chk("No claim of statistical significance", "statistically significant" not in (alltext(X) + alltext(B) + qtxt).lower().replace("no statistical significance test", "").replace("not a statistical", ""),
    "no significance test was run; stated explicitly")

# hand-typed numbers: numeric constants on the evidence sheets must be limited to source-derived inputs
def constants(ws, skip_cols=()):
    out = []
    for row in wf[ws].iter_rows():
        for c in row:
            if isinstance(c.value, (int, float)) and not isinstance(c.value, bool) and c.column not in skip_cols:
                out.append((c.coordinate, c.value))
    return out
chk("Trade-off Board: no typed numbers other than the rank column (everything links to the detail sheets)", all(coord.startswith("U") for coord, _ in constants("Trade-off Board")),
    f"{len(constants('Trade-off Board'))} constants, all in the rank column")
chk("Quality by Seed: only seed labels are typed numbers (accuracies are count / support formulas)", all(isinstance(v, int) and v in (0, 1, 2) and coord.startswith("B") for coord, v in constants("Quality by Seed")))

# user-quoted approximate figures vs workbook (rounded to the quoted precision)
quoted = {"FP32 Baseline Transformer": (97.63, 96.53, 5.23, 6.53, 75.40, 4.81), "FP32 Linear Attention": (98.19, 96.67, 6.93, 8.29, 75.40, 4.81),
          "Mixed-Precision INT8 Baseline": (97.64, 96.67, 11.47, 13.28, 73.91, 3.31), "Distilled Student": (97.98, 95.56, 2.85, 3.37, 69.53, 1.65)}
bad = []
for i, name in enumerate(cfg):
    r = 5 + i
    got = (B.cell(r, 7).value * 100, B.cell(r, 10).value * 100, B.cell(r, 13).value, B.cell(r, 14).value, B.cell(r, 16).value, B.cell(r, 5).value)
    for q, g, lab in zip(quoted[name], got, ["accuracy", "rare recall", "p50", "p95", "peak", "storage"]):
        if round(g, 2) != q: bad.append((name, lab, q, g))
chk("Quoted approximate values match the workbook's full-precision values when rounded to 2 decimals", not bad, str(bad) if bad else "all 24 match")
s = {k: B.cell(8, c).value for k, c in (("storage_red", 6), ("p95", 15), ("rare_pp", 12), ("acc_pp", 9), ("peak", 17))}
print()

print("Workbook file path :", PATH)
print("Sheet names        :", ", ".join(wf.sheetnames))
print("\n=== FINAL AUDIT ===")
for n, ok, d in res:
    print(f"[{'PASS' if ok else 'FLAG'}] {n}" + (f"  ({d})" if d else ""))
fails = [n for n, ok, _ in res if not ok]
print(f"\n{len(res) - len(fails)} of {len(res)} audit checks passed" + (": FLAGGED: " + "; ".join(fails) if fails else ""))
sys.exit(1 if fails else 0)
