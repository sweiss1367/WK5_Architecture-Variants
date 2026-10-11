"""Independent verification of Week5_Brightline_Tradeoff_Board.xlsx (run after build_tradeoff_board.py).

1. Every formula cell's cached value equals what LibreOffice computes when it recalculates the formula from scratch.
2. Key workbook values equal the full-precision values in the source CSV/JSON files.
3. Raw latency sheet equals benchmark_raw_latency.csv exactly.
"""
import sys, json
import numpy as np, pandas as pd, openpyxl

XLSX, RECALC = sys.argv[1], sys.argv[2]
wf = openpyxl.load_workbook(XLSX)                      # formulas
wc = openpyxl.load_workbook(XLSX, data_only=True)      # cached values written by the builder
wr = openpyxl.load_workbook(RECALC, data_only=True)    # values recomputed by LibreOffice
bad, n, maxrel = [], 0, 0.0
for ws in wf:
    for row in ws.iter_rows():
        for c in row:
            if isinstance(c.value, str) and c.value.startswith("="):
                n += 1
                a, b = wc[ws.title][c.coordinate].value, wr[ws.title][c.coordinate].value
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    rel = abs(a - b) / max(1e-300, abs(b)) if abs(b) > 1e-12 else abs(a - b)
                    maxrel = max(maxrel, rel)
                    if rel > 1e-9: bad.append((ws.title, c.coordinate, a, b))
                elif a != b:
                    bad.append((ws.title, c.coordinate, a, b))
print(f"1. formula cells checked: {n}; mismatches vs LibreOffice recalculation: {len(bad)}; max relative difference {maxrel:.2e}")
for x in bad[:10]: print("   MISMATCH", x)

summ = pd.read_csv("results/benchmark_summary.csv"); meta = json.load(open("results/benchmark_meta.json"))
B = wr["Trade-off Board"]; rows = {B.cell(r, 1).value: r for r in range(5, 9)}
order = ["FP32 Baseline Transformer", "FP32 Linear Attention", "Mixed-Precision INT8 Baseline", "Distilled Student"]
cols = {"Total Parameters": (3, "Total parameters"), "Weight Storage MB": (5, "Model weight storage MB"), "Overall Accuracy Mean": (7, "Overall accuracy mean (3 seeds)"),
        "Accuracy Std Dev": (8, "Overall accuracy std (3 seeds)"), "Rare-Class Macro Recall": (10, "Rare-class macro recall mean (3 seeds)"),
        "Rare-Class Std Dev": (11, "Rare-class macro recall std (3 seeds)"), "p50 Latency ms": (13, "p50 latency ms"), "p95 Latency ms": (14, "p95 latency ms"),
        "Peak GPU Memory MB": (16, "Peak GPU allocated memory MB")}
worst, fails = 0.0, []
for i, name in enumerate(order):
    for label, (col, scol) in cols.items():
        v = B.cell(rows[name], col).value
        s = summ.iloc[i][scol]
        if label.endswith("Std Dev"): v = v / 100            # board shows std in pp
        rel = abs(v - s) / abs(s)
        worst = max(worst, rel)
        if rel > 1e-12: fails.append((name, label, v, s))
print(f"2. Trade-off Board vs benchmark_summary.csv (9 metrics x 4 configs): worst relative difference {worst:.2e}; failures {len(fails)}")
for x in fails: print("   FAIL", x)

raw = pd.read_csv("results/benchmark_raw_latency.csv"); R = wc["Raw Benchmark Runs"]
vals = [R.cell(r, 5).value for r in range(5, 505)]
exact = np.array_equal(np.array(vals, dtype=float), raw["latency_ms"].values)
idx = [R.cell(r, 4).value for r in range(5, 505)] == raw["test_example_index"].tolist()
print(f"3. Raw Benchmark Runs: 500 latencies identical to CSV: {exact}; test-example indices identical: {idx}")

# detail sheet: recomputed p50/p95 vs meta
Dt = wr["Benchmark Detail"]; ok = True
for i, run in enumerate([r for r in meta["runs"]]):
    r = 5 + [(x["key"], x["run"]) for x in meta["runs"]].index((run["key"], run["run"]))
    # sheet order: baseline1, linear, int8, student, baseline2 == meta order
    p50, p95 = Dt.cell(5 + i, 3).value, Dt.cell(5 + i, 4).value
    if abs(p50 - run["p50"]) > 1e-9 or abs(p95 - run["p95"]) > 1e-9: ok = False
print("4. Benchmark Detail p50/p95 (recomputed from raw rows) equal benchmark_meta.json:", ok)
sc = pd.read_csv("results/scaling_law_results.csv"); S6 = wr["Scaling Law"]
sok = True
for i in range(4):
    for col, scol in ((5, "Estimated Actual Training FLOPs"), (7, "Compute-Optimal Training FLOPs"), (10, "10x Training FLOPs"), (11, "Compute Multiple")):
        if abs(S6.cell(6 + i, col).value - sc.iloc[i][scol]) > 1e-6 * abs(sc.iloc[i][scol]): sok = False
print("5. Scaling Law FLOPs and multiple (recomputed) equal scaling_law_results.csv:", sok)
