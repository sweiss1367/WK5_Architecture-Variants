"""Builds Week5_Brightline_Tradeoff_Board.xlsx from the completed experiment result files.

No model is trained and no benchmark is run here. Every number comes from tradeoff_data.load(), which reads existing
result files. Derived cells are written as Excel formulas (with the computed value cached) so they can be audited in Excel.
"""
import math, re
import numpy as np
import xlsxwriter
from xlsxwriter import xmlwriter as _xw
from xlsxwriter.utility import xl_rowcol_to_cell as rc, xl_col_to_name as cn


def _exact_number(self, number, attributes=[]):
    """xlsxwriter writes numbers with 16 significant digits, which can change the last digit of a double.
    Write the shortest string that round-trips exactly so raw measurements are preserved bit-for-bit."""
    attr = "".join(f' {k}="{self._escape_attributes(v)}"' for k, v in attributes)
    txt = f"{number}" if isinstance(number, int) and not isinstance(number, bool) else repr(float(number))
    self.fh.write(f"<c{attr}><v>{txt}</v></c>")


_xw.XMLwriter._xml_number_element = _exact_number

from tradeoff_data import (load, KEYS, NAMES, CLASSES, CLASS_LABEL, RARE, REPORT_CELL, NB)

OUT = "Week5_Brightline_Tradeoff_Board.xlsx"
D = load()
S, SUMM, META, RAW, MEM = D["seeds"], D["summ"], D["meta"], D["raw"], D["mem"]
SUPPORT, N_TEST = D["support"], D["n_test"]

# --------------------------------------------------------------------------------------------------------------
# Numbers (all derived from source files)
# --------------------------------------------------------------------------------------------------------------
METRICS = ["overall_test_acc", "acc_hazmat", "acc_oversize_heavy", "rare_macro_recall"]
SEEDVAL = {k: {m: [S[k][s]["value"][m] for s in range(3)] for m in METRICS + [f"acc_{c}" for c in CLASSES]} for k in KEYS}
MEAN = {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in SEEDVAL.items()}
STD = {k: {m: float(np.std(v, ddof=1)) for m, v in d.items()} for k, d in SEEDVAL.items()}

RUN1 = {r["key"]: r for r in META["runs"] if r["run"] == 1}
RUN2 = next(r for r in META["runs"] if r["key"] == "baseline" and r["run"] == 2)
BASE = RUN1["baseline"]
def rel(a, b): return a / b - 1.0

M = {}
for k in KEYS:
    r = RUN1[k]
    M[k] = dict(params=r["params"]["total"], nonemb=r["params"]["non_embedding"], storage=r["weight_storage_mb"], p50=r["p50"], p95=r["p95"],
                mean_lat=r["mean"], mn=r["min"], mx=r["max"], resident=r["resident_mb"], peak=r["peak_mb"], bg=r["background_mb"],
                acc=MEAN[k]["overall_test_acc"], acc_sd=STD[k]["overall_test_acc"], rare=MEAN[k]["rare_macro_recall"], rare_sd=STD[k]["rare_macro_recall"])
for k in KEYS:
    b = M["baseline"]
    M[k].update(d_params=1 - M[k]["params"] / b["params"], d_storage=1 - M[k]["storage"] / b["storage"],
                d_acc_pp=(M[k]["acc"] - b["acc"]) * 100, d_rare_pp=(M[k]["rare"] - b["rare"]) * 100,
                d_p50=rel(M[k]["p50"], b["p50"]), d_p95=rel(M[k]["p95"], b["p95"]), d_peak=rel(M[k]["peak"], b["peak"]))

# ratio of (mean difference vs baseline) to the larger seed std, per metric
DIFF = {}
for k in KEYS[1:]:
    for m in METRICS:
        d = MEAN[k][m] - MEAN["baseline"][m]
        larger = max(STD[k][m], STD["baseline"][m])
        seed_d = [SEEDVAL[k][m][s] - SEEDVAL["baseline"][m][s] for s in range(3)]
        DIFF[(k, m)] = dict(d=d, ratio=abs(d) / larger, seed_d=seed_d, below=sum(x < 0 for x in seed_d))
LABEL = {"overall_test_acc": "Overall Accuracy", "acc_hazmat": "Hazmat Recall", "acc_oversize_heavy": "Oversize / Heavy-Haul Recall",
         "rare_macro_recall": "Rare-Class Macro Recall"}
WITHIN = [(k, m) for (k, m), v in DIFF.items() if v["ratio"] < 1]
BEYOND = [(k, m) for (k, m), v in DIFF.items() if v["ratio"] >= 1]

# rare-class documents (hazmat + oversize correct) summed across seeds, student vs baseline
def rare_docs(k): return sum(S[k][s]["count"]["acc_hazmat"][0] + S[k][s]["count"]["acc_oversize_heavy"][0] for s in range(3))
RARE_DOCS_TOTAL = 3 * (SUPPORT["hazmat"] + SUPPORT["oversize_heavy"])
STUDENT_RARE_MISSED = rare_docs("baseline") - rare_docs("student")

# timing spikes in the raw data (>1.5x the run median), used only for a descriptive note
SPIKES = {}
for (key, run), g in RAW.groupby(["key", "run"], sort=False):
    l = g["latency_ms"].values; med = np.median(l); keep = l <= 1.5 * med
    SPIKES[(key, run)] = dict(n=int((~keep).sum()), p50=float(np.percentile(l, 50)), p95=float(np.percentile(l, 95)),
                              p50_wo=float(np.percentile(l[keep], 50)), p95_wo=float(np.percentile(l[keep], 95)))
REP_P50 = abs(RUN2["p50"] - BASE["p50"]) / BASE["p50"]
REP_P95 = abs(RUN2["p95"] - BASE["p95"]) / BASE["p95"]
THRESH = 0.15
STABLE = (REP_P50 <= THRESH) and (REP_P95 <= THRESH) and not META["repeatability"]["unstable"]
assert abs(REP_P50 * 100 - META["repeatability"]["p50_pct_diff"]) < 1e-9 and abs(REP_P95 * 100 - META["repeatability"]["p95_pct_diff"]) < 1e-9

def pct(x, d=1): return f"{x * 100:.{d}f}%"
def spct(x, d=1): return f"{x * 100:+.{d}f}%"
def pp(x, d=2): return f"{x:+.{d}f} pp"

# --------------------------------------------------------------------------------------------------------------
# Workbook + formats
# --------------------------------------------------------------------------------------------------------------
wb = xlsxwriter.Workbook(OUT)
wb.set_properties({"title": "Brightline Logistics Model Deployment Trade-Off Board", "subject": "Week 5 model trade-off assignment",
                   "comments": "Built from completed experiment result files; no values were retyped or invented."})
NAVY, GREY, LGREY, GREEN, AMBER, LBLUE = "#1F3A5F", "#595959", "#F2F2F2", "#E2F0D9", "#FFF2CC", "#DDEBF7"
_fc = {}
def F(**kw):
    key = tuple(sorted(kw.items()))
    if key not in _fc:
        _fc[key] = wb.add_format({"font_name": "Calibri", "font_size": 11, "valign": "vcenter", **kw})
    return _fc[key]
BORD = dict(border=1, border_color="#BFC7D1")
TITLE = F(bold=True, font_size=20, font_color=NAVY)
SUB = F(italic=True, font_size=11, font_color=GREY)
HDR = F(bold=True, font_color="white", bg_color=NAVY, text_wrap=True, align="center", **BORD)
SEC = F(bold=True, font_size=12, font_color="white", bg_color="#44546A")
NOTE = F(text_wrap=True, valign="top", font_color="#262626")
NOTEB = F(text_wrap=True, valign="top", bold=True, font_color="#262626")
TXT = F(text_wrap=True, valign="top", **BORD)
TXTB = F(text_wrap=True, valign="top", bold=True, **BORD)

def est_h(text, width, pt=11, min_h=18):
    """Estimated row height for wrapped text in a column `width` characters wide."""
    chars = max(1.0, width * 1.10 * 11 / pt)
    lines = sum(max(1, math.ceil(len(p) / chars)) for p in str(text).split("\n"))
    return max(min_h, lines * (pt * 1.32) + 10)

# Sheets in presentation order
ws_b = wb.add_worksheet("Trade-off Board")
ws_q = wb.add_worksheet("Quality by Seed")
ws_c = wb.add_worksheet("Per-Class Results")
ws_r = wb.add_worksheet("Raw Benchmark Runs")
ws_d = wb.add_worksheet("Benchmark Detail")
ws_s = wb.add_worksheet("Scaling Law")
ws_m = wb.add_worksheet("Methodology")
ws_x = wb.add_worksheet("Recommendation")
for w in (ws_b, ws_q, ws_c, ws_r, ws_d, ws_s, ws_m, ws_x):
    w.hide_gridlines(2)
    w.set_landscape(); w.fit_to_pages(1, 0); w.set_paper(9)

QN, CN_, RN, DN, SN, BN = "'Quality by Seed'", "'Per-Class Results'", "'Raw Benchmark Runs'", "'Benchmark Detail'", "'Scaling Law'", "'Trade-off Board'"

# ==============================================================================================================
# Sheet 2: Quality by Seed
# ==============================================================================================================
ws = ws_q
ws.set_column("A:A", 34); ws.set_column("B:B", 24); ws.set_column("C:I", 15); ws.set_column("J:J", 62)
ws.write(0, 0, "Quality by Seed: three seeds per configuration", TITLE); ws.set_row(0, 30)
ws.write(1, 0, f"Test set: {N_TEST:,} documents ({SUPPORT['hazmat']} Hazmat, {SUPPORT['oversize_heavy']} Oversize / Heavy Haul). "
               "Each seed value is correct documents / test support; click a cell to see the counts.", SUB)
QH = ["Configuration", "Seed", "Overall Accuracy", "Hazmat Recall", "Oversize / Heavy-Haul Recall", "Rare-Class Macro Recall"]
for c, h in enumerate(QH):
    ws.write(3, c, h, HDR)
ws.merge_range(3, 6, 3, 9, "Source (executed notebook output)", HDR); ws.set_row(3, 48)
QROW = {}
r = 4
for ki, k in enumerate(KEYS):
    for s in range(3):
        QROW[(k, s)] = r
        cnt = S[k][s]["count"]; val = S[k][s]["value"]
        band = F(bg_color=LGREY if ki % 2 else "white", **BORD)
        pcts = F(num_format="0.00%", bg_color=LGREY if ki % 2 else "white", **BORD)
        ws.write(r, 0, NAMES[k], band); ws.write(r, 1, s, F(align="center", bg_color=LGREY if ki % 2 else "white", **BORD))
        for col, name in ((2, "overall_test_acc"), (3, "acc_hazmat"), (4, "acc_oversize_heavy")):
            kk, n = cnt[name]
            ws.write_formula(r, col, f"={kk}/{n}", pcts, kk / n)
        ws.write_formula(r, 5, f"=({rc(r, 3)}+{rc(r, 4)})/2", pcts, val["rare_macro_recall"])
        ws.merge_range(r, 6, r, 9, f"{NB}, cell {REPORT_CELL[k]} output (per-seed table), seed {s}", F(font_color=GREY, font_size=10, **BORD))
        r += 1
# summary block
r += 1
ws.merge_range(r, 0, r, 9, "Summary over the three seeds: mean and sample standard deviation (n - 1)", SEC); r += 1
for c, h in enumerate(["Configuration", "Statistic (std dev in percentage points)"] + QH[2:]):
    ws.write(r, c, h, HDR)
ws.set_row(r, 48); r += 1
QMEAN, QSTD = {}, {}
for ki, k in enumerate(KEYS):
    first, last = QROW[(k, 0)], QROW[(k, 2)]
    for stat in ("Mean", "Sample Std Dev"):
        is_mean = stat == "Mean"
        (QMEAN if is_mean else QSTD)[k] = r
        bg = LGREY if ki % 2 else "white"
        ws.write(r, 0, NAMES[k], F(bg_color=bg, bold=is_mean, **BORD)); ws.write(r, 1, stat, F(bg_color=bg, bold=is_mean, **BORD))
        for col, m in zip((2, 3, 4, 5), METRICS):
            rng = f"{rc(first, col)}:{rc(last, col)}"
            if is_mean:
                ws.write_formula(r, col, f"=AVERAGE({rng})", F(num_format="0.00%", bg_color=bg, bold=True, **BORD), MEAN[k][m])
            else:
                ws.write_formula(r, col, f"=STDEV({rng})*100", F(num_format='0.000" pp"', bg_color=bg, **BORD), STD[k][m] * 100)
        r += 1
# differences vs baseline compared with seed spread
r += 1
ws.merge_range(r, 0, r, 9, "Difference from the FP32 baseline, compared with the observed seed spread", SEC); r += 1
DH = ["Configuration", "Metric", "Mean Difference vs Baseline (pp)", "Seed 0 Difference (pp)", "Seed 1 Difference (pp)", "Seed 2 Difference (pp)",
      "Baseline Seed Std Dev (pp)", "Config Seed Std Dev (pp)", "|Difference| / Larger Std Dev", "Reading"]
for c, h in enumerate(DH):
    ws.write(r, c, h, HDR)
ws.set_row(r, 62); r += 1
QDIFF = {}
for k in KEYS[1:]:
    for mi, m in enumerate(METRICS):
        col = 2 + mi
        d = DIFF[(k, m)]
        QDIFF[(k, m)] = r
        ws.write(r, 0, NAMES[k], F(**BORD)); ws.write(r, 1, LABEL[m], F(**BORD))
        ws.write_formula(r, 2, f"=({rc(QMEAN[k], col)}-{rc(QMEAN['baseline'], col)})*100", F(num_format='+0.00;-0.00;0.00', **BORD), d["d"] * 100)
        for s in range(3):
            ws.write_formula(r, 3 + s, f"=({rc(QROW[(k, s)], col)}-{rc(QROW[('baseline', s)], col)})*100", F(num_format='+0.00;-0.00;0.00', **BORD), d["seed_d"][s] * 100)
        ws.write_formula(r, 6, f"={rc(QSTD['baseline'], col)}", F(num_format="0.00", **BORD), STD["baseline"][m] * 100)
        ws.write_formula(r, 7, f"={rc(QSTD[k], col)}", F(num_format="0.00", **BORD), STD[k][m] * 100)
        ws.write_formula(r, 8, f"=ABS({rc(r, 2)})/MAX({rc(r, 6)},{rc(r, 7)})", F(num_format="0.00", bold=True, **BORD), d["ratio"])
        reading = ("Smaller than the seed spread: treat as similar" if d["ratio"] < 1 else
                   "Larger than the seed spread (only 3 seeds; no significance test was run)")
        ws.write_formula(r, 9, f'=IF({rc(r, 8)}<1,"Smaller than the seed spread: treat as similar","Larger than the seed spread (only 3 seeds; no significance test was run)")',
                         F(text_wrap=True, **BORD), reading)
        r += 1
r += 1
qnotes = [
    "How to read this sheet: a difference below 1.00 in the second-to-last column is smaller than the seed-to-seed spread of the larger of the two models, so it should be treated as similar, not as a real gain or loss.",
    "Only three seeds were run, so each standard deviation is itself uncertain. No statistical significance test was run; wording such as 'similar within observed seed variation' is descriptive.",
    f"Rare-class test support is {SUPPORT['hazmat']} documents per class, so one document = {100 / SUPPORT['hazmat']:.2f} percentage points of class recall.",
    "Seed pairing: the linear-attention model starts from the same initial weights as the baseline of the same seed; the INT8 model is quantized from the baseline of the same seed; the student of seed s is taught by the baseline of seed s. The per-seed difference columns are therefore paired comparisons.",
    f"Provenance: per-seed values are the 4-decimal tables printed in {NB}. Each equals correct documents / support; the counts were recovered exactly and the means and standard deviations were checked against benchmark_summary.csv (largest difference {max(max(a[2], a[3]) for a in D['crosscheck']):.1e}).",
]
for t in qnotes:
    ws.merge_range(r, 0, r, 9, t, NOTE); ws.set_row(r, est_h(t, 200)); r += 1
ws.freeze_panes(4, 2)

# ==============================================================================================================
# Sheet 3: Per-Class Results
# ==============================================================================================================
ws = ws_c
ws.set_column("A:A", 34); ws.set_column("B:B", 24); ws.set_column("C:F", 17); ws.set_column("G:G", 56); ws.set_column("H:H", 20)
ws.write(0, 0, "Per-Class Results (mean over seeds 0, 1, 2)", TITLE); ws.set_row(0, 30)
ws.write(1, 0, "Accuracy per class = recall of that class. Rare classes (Hazmat, Oversize / Heavy Haul) are shaded and each has only 120 test documents: "
               "a rare-class decline is shown here even when overall accuracy is unchanged.", SUB)
CH = ["Configuration", "Class", "Test Support", "Mean Accuracy / Recall", "Seed Std Dev", "Change vs FP32 Baseline (pp)", "Class Type"]
for c, h in enumerate(CH):
    ws.write(3, c, h, HDR)
ws.set_row(3, 48)
LOW0 = 4 + 24 + 4         # first data row of the source block (0-indexed)
CROW = {}
for ki, k in enumerate(KEYS):
    for ci, cl in enumerate(CLASSES):
        rr = 4 + ki * 6 + ci; CROW[(k, cl)] = rr
        lr = LOW0 + ki * 6 + ci
        rare = cl in RARE
        bg = AMBER if rare else ("white" if ki % 2 == 0 else LGREY)
        vals = SEEDVAL[k][f"acc_{cl}"]
        ws.write(rr, 0, NAMES[k], F(bg_color=bg, bold=rare, **BORD)); ws.write(rr, 1, CLASS_LABEL[cl], F(bg_color=bg, bold=rare, **BORD))
        ws.write(rr, 2, SUPPORT[cl], F(num_format="#,##0", bg_color=bg, **BORD))
        ws.write_formula(rr, 3, f"=AVERAGE({rc(lr, 3)}:{rc(lr, 5)})", F(num_format="0.00%", bg_color=bg, bold=rare, **BORD), MEAN[k][f"acc_{cl}"])
        ws.write_formula(rr, 4, f"=STDEV({rc(lr, 3)}:{rc(lr, 5)})*100", F(num_format='0.00" pp"', bg_color=bg, **BORD), STD[k][f"acc_{cl}"] * 100)
        brow = 4 + CLASSES.index(cl)
        ws.write_formula(rr, 5, f"=({rc(rr, 3)}-{rc(brow, 3)})*100", F(num_format='+0.00" pp";-0.00" pp";0.00" pp"', bg_color=bg, bold=rare, **BORD),
                         (MEAN[k][f"acc_{cl}"] - MEAN["baseline"][f"acc_{cl}"]) * 100)
        ws.write(rr, 6, "Rare class (4% of data)" if rare else "Common class", F(bg_color=bg, bold=rare, **BORD))
cnotes_r = 4 + 24 + 1
ws.merge_range(cnotes_r, 0, cnotes_r, 6,
               "Seed Std Dev is the sample standard deviation (pp) over three seeds. Compare each change with it, and with the 0.83 pp size of one rare-class document, before reading a difference as real.", NOTE)
ws.set_row(cnotes_r, 34)
ws.merge_range(LOW0 - 2, 0, LOW0 - 2, 6, "Source values per seed (correct documents / test support; click a cell to see the counts)", SEC)
for c, h in enumerate(["Configuration", "Class", "Test Support", "Seed 0", "Seed 1", "Seed 2", "Source"]):
    ws.write(LOW0 - 1, c, h, HDR)
for ki, k in enumerate(KEYS):
    for ci, cl in enumerate(CLASSES):
        lr = LOW0 + ki * 6 + ci; rare = cl in RARE
        bg = AMBER if rare else ("white" if ki % 2 == 0 else LGREY)
        ws.write(lr, 0, NAMES[k], F(bg_color=bg, **BORD)); ws.write(lr, 1, CLASS_LABEL[cl], F(bg_color=bg, **BORD))
        ws.write(lr, 2, SUPPORT[cl], F(num_format="#,##0", bg_color=bg, **BORD))
        for s in range(3):
            kk, n = S[k][s]["count"][f"acc_{cl}"]
            ws.write_formula(lr, 3 + s, f"={kk}/{n}", F(num_format="0.00%", bg_color=bg, **BORD), kk / n)
        ws.write(lr, 6, f"{NB}, cell {REPORT_CELL[k]}", F(font_color=GREY, font_size=10, bg_color=bg, **BORD))
ws.freeze_panes(4, 2)

# ==============================================================================================================
# Sheet 4: Raw Benchmark Runs
# ==============================================================================================================
ws = ws_r
ws.set_column("A:A", 34); ws.set_column("B:E", 17); ws.set_column("F:F", 20)
ws.write(0, 0, "Raw Benchmark Runs: every timed trial", TITLE); ws.set_row(0, 30)
ws.write(1, 0, "Source: results/benchmark_raw_latency.csv. Batch size 1, sequence length 128, 15 untimed warm-ups then 100 timed trials per run, CUDA synchronize before and after each forward. "
               "Run 2 is the baseline repeat used for the stability check.", SUB)
for c, h in enumerate(["Configuration", "Benchmark Run", "Trial Number", "Test Example Index", "Latency ms", "Document Position (0-99)"]):
    ws.write(3, c, h, HDR)
ws.set_row(3, 34)
RAW_RANGE = {}
rr = 4
for (key, run), g in RAW.groupby(["key", "run"], sort=False):
    RAW_RANGE[(key, int(run))] = (rr + 1, rr + len(g))          # 1-indexed Excel rows
    for _, row in g.iterrows():
        ws.write(rr, 0, NAMES[key]); ws.write_number(rr, 1, int(row["run"])); ws.write_number(rr, 2, int(row["trial"]))
        ws.write_number(rr, 3, int(row["test_example_index"])); ws.write_number(rr, 4, float(row["latency_ms"]), F(num_format="0.000"))
        ws.write_number(rr, 5, int(row["document_position"]))
        rr += 1
assert rr - 4 == len(RAW) == 500 and all(b - a + 1 == 100 for a, b in RAW_RANGE.values())
ws.autofilter(3, 0, rr - 1, 5)
ws.freeze_panes(4, 1)

# ==============================================================================================================
# Sheet 5: Benchmark Detail
# ==============================================================================================================
ws = ws_d
ws.set_column("A:A", 34); ws.set_column("B:B", 12); ws.set_column("C:K", 14); ws.set_column("L:M", 15); ws.set_column("N:O", 20)
ws.write(0, 0, "Benchmark Detail", TITLE); ws.set_row(0, 30)
ws.write(1, 0, f"{META['env']['gpu_name']} (compute capability {META['env']['gpu_compute_capability']}), batch size 1, sequence length 128, seed-0 models, 15 warm-up and 100 timed trials. "
               "Latency columns are formulas over the Raw Benchmark Runs sheet.", SUB)
DHDR = ["Configuration", "Benchmark Run", "p50 Latency ms", "p95 Latency ms", "Mean Latency ms", "Minimum Latency ms", "Maximum Latency ms",
        "Resident GPU Memory MB", "Peak GPU Memory MB (primary)", "Incremental Peak Memory MB", "Weight Storage MB", "Total Parameters",
        "Non-Embedding Parameters", "Background Allocated Before Load MB (diagnostic)", "Peak Minus Background MB (diagnostic)"]
for c, h in enumerate(DHDR):
    ws.write(3, c, h, HDR)
ws.set_row(3, 76)
DROW = {}
detail_runs = [("baseline", 1), ("linear", 1), ("int8", 1), ("student", 1), ("baseline", 2)]
for i, (k, run) in enumerate(detail_runs):
    r_ = 4 + i; DROW[(k, run)] = r_
    src = RUN1[k] if run == 1 else RUN2
    a, b = RAW_RANGE[(k, run)]
    rng = f"{RN}!$E${a}:$E${b}"
    lat = RAW[(RAW["key"] == k) & (RAW["run"] == run)]["latency_ms"].values
    bg = LGREY if run == 2 else "white"
    name = NAMES[k] + (" (repeat run for stability check)" if run == 2 else "")
    ws.write(r_, 0, name, F(bg_color=bg, text_wrap=True, **BORD)); ws.write(r_, 1, run, F(align="center", bg_color=bg, **BORD))
    lf = F(num_format="0.000", bg_color=bg, **BORD)
    ws.write_formula(r_, 2, f"=PERCENTILE({rng},0.5)", lf, float(np.percentile(lat, 50)))
    ws.write_formula(r_, 3, f"=PERCENTILE({rng},0.95)", lf, float(np.percentile(lat, 95)))
    ws.write_formula(r_, 4, f"=AVERAGE({rng})", lf, float(lat.mean()))
    ws.write_formula(r_, 5, f"=MIN({rng})", lf, float(lat.min()))
    ws.write_formula(r_, 6, f"=MAX({rng})", lf, float(lat.max()))
    mf = F(num_format="0.000", bg_color=bg, **BORD)
    ws.write_number(r_, 7, src["resident_mb"], mf)
    ws.write_number(r_, 8, src["peak_mb"], F(num_format="0.000", bold=True, bg_color=bg, **BORD))
    ws.write_formula(r_, 9, f"={rc(r_, 8)}-{rc(r_, 7)}", mf, src["peak_mb"] - src["resident_mb"])
    ws.write_number(r_, 10, src["weight_storage_mb"], F(num_format="0.000", bg_color=bg, **BORD))
    ws.write_number(r_, 11, src["params"]["total"], F(num_format="#,##0", bg_color=bg, **BORD))
    ws.write_number(r_, 12, src["params"]["non_embedding"], F(num_format="#,##0", bg_color=bg, **BORD))
    ws.write_number(r_, 13, src["background_mb"], mf)
    ws.write_formula(r_, 14, f"={rc(r_, 8)}-{rc(r_, 13)}", mf, src["peak_mb"] - src["background_mb"])
    ws.set_row(r_, 34)
r = 4 + len(detail_runs) + 1
ws.merge_range(r, 0, r, 14, "Baseline repeatability check (stability check only; Run 1 is the reported benchmark value)", SEC); r += 1
for c, h in enumerate(["Metric", "", "Run 1 (ms)", "Run 2 (ms)", "% Difference", "Threshold", "Within Threshold?"]):
    ws.write(r, c, h, HDR)
r += 1
REP_ROW = {}
for label, col, rel_ in (("p50 latency", 2, REP_P50), ("p95 latency", 3, REP_P95)):
    REP_ROW[label] = r
    ws.write(r, 0, label, F(bold=True, **BORD)); ws.write(r, 1, "", F(**BORD))
    ws.write_formula(r, 2, f"={rc(DROW[('baseline', 1)], col)}", F(num_format="0.000", **BORD), (BASE if col == 2 else BASE)["p50" if col == 2 else "p95"])
    ws.write_formula(r, 3, f"={rc(DROW[('baseline', 2)], col)}", F(num_format="0.000", **BORD), RUN2["p50" if col == 2 else "p95"])
    ws.write_formula(r, 4, f"=ABS({rc(r, 3)}-{rc(r, 2)})/{rc(r, 2)}", F(num_format="0.00%", bold=True, **BORD), rel_)
    ws.write_number(r, 5, THRESH, F(num_format="0%", **BORD))
    ws.write_formula(r, 6, f'=IF({rc(r, 4)}<={rc(r, 5)},"Yes","No: timing flagged unstable")', F(bold=True, **BORD), "Yes" if rel_ <= THRESH else "No: timing flagged unstable")
    r += 1
stmt = (f"Both baseline runs stayed within the predetermined 15% stability threshold (p50 differs by {REP_P50 * 100:.2f}%, p95 by {REP_P95 * 100:.2f}%), so the latency comparison is treated as final."
        if STABLE else f"WARNING: the baseline repeat exceeded the 15% threshold (p50 {REP_P50 * 100:.2f}%, p95 {REP_P95 * 100:.2f}%); the latency comparison is not final.")
ws.merge_range(r, 0, r, 14, stmt, F(bold=True, text_wrap=True, valign="top", bg_color=GREEN if STABLE else AMBER, **BORD)); ws.set_row(r, 22); r += 2
sp = SPIKES
sp_text = (f"Timing note: every run contains a few short bursts of trials at roughly 1.5 to 2 times its own median latency (more than 1.5x the run median: baseline run 1 = {sp[('baseline', 1)]['n']} trials, "
           f"linear attention = {sp[('linear', 1)]['n']}, INT8 = {sp[('int8', 1)]['n']}, student = {sp[('student', 1)]['n']}, baseline run 2 = {sp[('baseline', 2)]['n']}). They appear in consecutive trials, which is typical of transient interference on shared hardware. "
           f"They barely move p50 (the baseline run 1 p50 changes from {sp[('baseline', 1)]['p50']:.3f} to {sp[('baseline', 1)]['p50_wo']:.3f} ms if they are excluded) but do move p95 (baseline run 1: {sp[('baseline', 1)]['p95']:.3f} to {sp[('baseline', 1)]['p95_wo']:.3f} ms). "
           "No trial was removed from any reported number; this is a sensitivity note only. p50 is the more robust comparison metric.")
mem_text = (f"Memory note: Resident, Peak and Incremental values are PyTorch GPU allocator memory (1 MB = 1,048,576 bytes); they do not include the CUDA context or other driver-level memory. "
            f"A constant {META['runs'][0]['background_mb']:.4f} MB was already allocated before each model was loaded (identical for all five runs; its origin was not isolated, though it is consistent with library workspaces). "
            "Peak GPU Memory is the primary deployment-memory metric. Resident and Incremental Peak are supporting diagnostics, and the two background-subtracted columns are diagnostic only: they isolate the model-plus-inference footprint and are not a deployment-memory figure.")
for t in (sp_text, mem_text):
    ws.merge_range(r, 0, r, 14, t, NOTE); ws.set_row(r, est_h(t, 235)); r += 1
ws.freeze_panes(4, 1)

# ==============================================================================================================
# Sheet 6: Scaling Law
# ==============================================================================================================
ws = ws_s
SC = D["sc_csv"]; SCJ = D["sc_json"]
ws.set_column("A:A", 40); ws.set_column("B:D", 17); ws.set_column("E:K", 19); ws.set_column("L:L", 70)
ws.write(0, 0, "Scaling Law: compute estimates", TITLE); ws.set_row(0, 30)
ws.write(1, 0, "N = non-embedding parameters (total parameters shown for context only). Counted quantities come from the experiment; every FLOP figure and every projection is a formula-based estimate, not a measurement.", SUB)
ws.merge_range(3, 1, 3, 3, "MEASURED / COUNTED", F(bold=True, align="center", bg_color=LBLUE, font_color=NAVY, **BORD))
ws.merge_range(3, 4, 3, 10, "FORMULA-BASED ESTIMATES (6ND and D = 20N): NOT MEASUREMENTS", F(bold=True, align="center", bg_color=AMBER, font_color="#7F6000", **BORD))
SH = ["Configuration", "Total Parameters", "Non-Embedding Parameters N", "Actual Training Tokens", "Estimated Actual Training FLOPs (6ND)",
      "Compute-Optimal Tokens (D = 20N)", "Compute-Optimal Training FLOPs (C = 6ND)", "10x Non-Embedding Parameters", "10x Compute-Optimal Tokens",
      "10x Training FLOPs", "Compute Multiple", "Notes"]
for c, h in enumerate(SH):
    ws.write(4, c, h, HDR)
ws.set_row(4, 62)
sci = '0.000E+00'
SROW = {}
for i, k in enumerate(KEYS):
    r_ = 5 + i; SROW[k] = r_
    row = SC.iloc[i]
    assert row["Configuration"].startswith({"baseline": "FP32 baseline", "linear": "FP32 linear", "int8": "Mixed-precision INT8", "student": "FP32 distilled"}[k]), row["Configuration"]
    tot, N, Dtok = int(row["Total Parameters"]), int(row["Non-Embedding Parameters N"]), int(row["Actual Training Tokens"])
    assert N == M[k]["nonemb"] and tot == M[k]["params"]
    est = F(num_format=sci, bg_color=AMBER if False else "white", **BORD)
    ws.write(r_, 0, NAMES[k], F(bold=True, text_wrap=True, **BORD))
    ws.write_number(r_, 1, tot, F(num_format="#,##0", **BORD)); ws.write_number(r_, 2, N, F(num_format="#,##0", **BORD)); ws.write_number(r_, 3, Dtok, F(num_format="#,##0", **BORD))
    C = row["Estimated Actual Training FLOPs"]; Dopt = row["Compute-Optimal Tokens 20N"]; Copt = row["Compute-Optimal Training FLOPs"]
    N10, D10, C10, mult = row["10x Non-Embedding Parameters"], row["10x Compute-Optimal Tokens"], row["10x Training FLOPs"], row["Compute Multiple"]
    ws.write_formula(r_, 4, f"=6*{rc(r_, 2)}*{rc(r_, 3)}", est, float(C))
    ws.write_formula(r_, 5, f"=20*{rc(r_, 2)}", F(num_format="#,##0", **BORD), float(Dopt))
    ws.write_formula(r_, 6, f"=6*{rc(r_, 2)}*{rc(r_, 5)}", est, float(Copt))
    ws.write_formula(r_, 7, f"=10*{rc(r_, 2)}", F(num_format="#,##0", **BORD), float(N10))
    ws.write_formula(r_, 8, f"=20*{rc(r_, 7)}", F(num_format="#,##0", **BORD), float(D10))
    ws.write_formula(r_, 9, f"=6*{rc(r_, 7)}*{rc(r_, 8)}", est, float(C10))
    ws.write_formula(r_, 10, f"={rc(r_, 9)}/{rc(r_, 6)}", F(num_format="0.0000", bold=True, **BORD), float(mult))
    assert 6 * N * Dtok == C and 20 * N == Dopt and 6 * N * Dopt == Copt and 10 * N == N10 and 20 * N10 == D10 and 6 * N10 * D10 == C10 and abs(C10 / Copt - mult) < 1e-9
    ws.write(r_, 11, row["Notes"], F(text_wrap=True, valign="top", **BORD))
    ws.set_row(r_, max(64, est_h(row["Notes"], 70)))
r = 10
ws.merge_range(r, 0, r, 11, "Readable view (same estimates; TFLOP = 10^12, PFLOP = 10^15, GFLOP = 10^9 floating-point operations)", SEC); r += 1
def human(x):
    for scale, unit in ((1e15, "PFLOP"), (1e12, "TFLOP"), (1e9, "GFLOP"), (1e6, "MFLOP")):
        if x >= scale:
            return f"{x / scale:,.2f} {unit}"
    return f"{x:,.0f} FLOP"
for c, h in enumerate(["Configuration", "Actual Training FLOPs", "Compute-Optimal FLOPs", "10x Training FLOPs", "Actual / Compute-Optimal Tokens"]):
    ws.write(r, c, h, HDR)
ws.set_row(r, 34); r += 1
for i, k in enumerate(KEYS):
    row = SC.iloc[i]
    ws.write(r, 0, NAMES[k], F(**BORD))
    ws.write(r, 1, human(row["Estimated Actual Training FLOPs"]), F(align="right", **BORD))
    ws.write(r, 2, human(row["Compute-Optimal Training FLOPs"]), F(align="right", **BORD))
    ws.write(r, 3, human(row["10x Training FLOPs"]), F(align="right", **BORD))
    ws.write(r, 4, f"{row['Actual Training Tokens'] / row['Compute-Optimal Tokens 20N']:.2f}x", F(align="right", **BORD))
    r += 1
r += 1
ws.merge_range(r, 0, r, 11, "Method and notes", SEC); r += 1
wl = SCJ["workload"]
snotes = [
    f"Training workload (identical for the baseline, linear-attention and student runs): {wl['train_docs']:,} documents x {wl['seq_len']} positions x {wl['epochs']} epochs = {wl['padded_positions']:,} padded sequence positions. "
    f"D uses padded positions because the GPU computed on all {wl['seq_len']} positions of every document; non-padding tokens would be {wl['non_padding_tokens']:,} ({wl['padded_positions'] / wl['non_padding_tokens']:.2f}x fewer). D counts 8 repeated epochs, not unique tokens.",
    "Post-training INT8 does not reduce the original baseline training FLOPs: the INT8 model was produced from the already trained FP32 baseline, so its row shows the baseline's training compute. Quantization changes the inference representation, not the original training compute.",
    f"The distilled student's basic 6ND calculation does not include teacher-inference cost. Teacher-logit caching for the student (one forward pass over train and validation documents per seed) was measured at "
    f"{', '.join(f'{t:.1f}' for t in SCJ['teacher_cost']['measured_logit_cache_seconds_per_seed'])} s for seeds 0, 1, 2 (printed to one decimal); a formula-based forward-only estimate is {SCJ['teacher_cost']['forward_only_2ND_estimate_flops_per_seed']:.3e} FLOPs per seed. Neither is added to the student's 6ND figure, and the student also required the baseline teacher to be trained first.",
    "The 10x projection is an estimate (N_10x = 10N, D_10x = 20 N_10x, C_10x = 6 N_10x D_10x). The compute multiple is calculated as C_10x / C_compute-optimal, and it is about 100x because parameters and the compute-optimal token budget both grow 10x. These are not measured training runs.",
    "Scaling may change GPU utilization, and larger matrix multiplications may use the hardware more efficiently, so FLOPs would not translate linearly into wall-clock time.",
    "A larger model may exceed single-GPU memory. Gradient checkpointing or smaller batches could then change the wall-clock cost.",
    "Longer sequence lengths would change attention cost, which 6ND does not model. Multi-GPU scaling would introduce communication overhead.",
    "The 20-tokens-per-parameter relationship is a rule of thumb and may not be optimal for this synthetic classification workload. 6ND also ignores attention-score FLOPs, embeddings, LayerNorm, the optimizer and the per-epoch validation passes.",
    "Scaling-law figures are not used to rank the models.",
    "Provenance: results/scaling_law_results.csv and .json, produced by the notebook's Part 6 cell from the counts printed in the executed notebooks (parameter counts, 10,500 training documents, 8 epochs, length 128). The cells in this sheet are formulas that reproduce the CSV values exactly.",
]
for t in snotes:
    ws.merge_range(r, 0, r, 11, t, NOTE); ws.set_row(r, est_h(t, 250)); r += 1
ws.freeze_panes(5, 1)

# ==============================================================================================================
# Sheet 1: Trade-off Board
# ==============================================================================================================
ws = ws_b
widths = [27, 29, 12, 13, 11, 12, 11, 10, 12, 11, 10, 12, 10, 10, 11, 11, 12, 33, 33, 33, 8]
for c, w in enumerate(widths):
    ws.set_column(c, c, w)
ws.write(0, 0, "Brightline Logistics Model Deployment Trade-Off Board", F(bold=True, font_size=24, font_color=NAVY)); ws.set_row(0, 38)
ws.write(1, 0, "Measured on one Google Colab Tesla T4 using a controlled synthetic bill-of-lading classification workload", F(italic=True, font_size=13, font_color=GREY)); ws.set_row(1, 22)
ws.write(2, 0, "Quality = mean of 3 seeds on 3,000 test documents. Latency and memory = seed-0 model, batch size 1, sequence length 128, 100 timed trials. pp = percentage points. Prepared for Sandra Whitfield, VP Engineering.", F(font_size=10, font_color=GREY))
BH = ["Configuration", "Precision / Compression", "Total Parameters", "Parameter Reduction vs Baseline", "Weight Storage MB", "Storage Reduction vs Baseline",
      "Overall Accuracy Mean", "Accuracy Std Dev", "Accuracy Change vs Baseline", "Rare-Class Macro Recall", "Rare-Class Std Dev", "Rare-Class Change vs Baseline",
      "p50 Latency ms", "p95 Latency ms", "p95 Change vs Baseline", "Peak GPU Memory MB", "Peak Memory Change vs Baseline",
      "Main Advantage", "Main Trade-off", "Residual Risk", "Final Rank"]
for c, h in enumerate(BH):
    ws.write(3, c, h, F(bold=True, font_color="white", bg_color=NAVY, text_wrap=True, align="center", font_size=11, **BORD))
ws.set_row(3, 64)
PRECISION = {
    "baseline": "FP32 (reference model)",
    "linear": "FP32; linear (ELU+1) attention in place of softmax attention; same parameter count",
    "int8": "bitsandbytes INT8 on 16 hidden Linear layers; the 128 -> 6 classification head, embeddings and LayerNorm stay FP32",
    "student": "FP32; distilled student (d=64, 2 layers, 2 heads) taught by the FP32 baseline",
}
RANK = {"student": 1, "baseline": 2, "linear": 3, "int8": 4}
m = M
def hz(k): return MEAN[k]["acc_hazmat"]
ADV = {
    "baseline": f"Simplest, best-understood reference: no compression, custom attention or distillation. {pct(m['baseline']['acc'], 2)} overall accuracy at {m['baseline']['p50']:.2f} ms p50.",
    "linear": f"Highest mean overall accuracy ({pct(m['linear']['acc'], 2)}, {pp(m['linear']['d_acc_pp'])}), above every baseline seed; rare-class recall similar to the baseline.",
    "int8": f"Weight storage {pct(m['int8']['d_storage'])} smaller ({m['int8']['storage']:.2f} MB) with overall accuracy unchanged ({pct(m['int8']['acc'], 2)}).",
    "student": f"Smallest and fastest: weights {pct(m['student']['d_storage'])} smaller, p95 latency {pct(-m['student']['d_p95'])} lower, peak GPU memory {pct(-m['student']['d_peak'])} lower, with overall accuracy not lower than the baseline ({pct(m['student']['acc'], 2)}).",
}
TRD = {
    "baseline": f"Largest weights ({m['baseline']['storage']:.2f} MB) and about twice the student's p95 latency ({m['baseline']['p95']:.2f} ms vs {m['student']['p95']:.2f} ms).",
    "linear": f"Slower than the baseline at sequence length 128 (p95 {spct(m['linear']['d_p95'])}); no storage or peak-memory reduction.",
    "int8": f"Slowest model on the T4: p95 {spct(m['int8']['d_p95'])} vs baseline; peak GPU memory only {spct(m['int8']['d_peak'])}; the 6-class head could not be quantized.",
    "student": f"Mean rare-class macro recall {abs(m['student']['d_rare_pp']):.2f} pp lower than baseline ({pct(m['student']['rare'], 2)} vs {pct(m['baseline']['rare'], 2)}); needs a trained teacher and a distillation step.",
}
RSK = {
    "baseline": f"Rare-class recall varies {m['baseline']['rare_sd'] * 100:.2f} pp (std dev) across seeds; synthetic data only; one GPU type.",
    "linear": f"Accuracy edge is small ({pp(m['linear']['d_acc_pp'])}); behavior at longer sequences was not tested; custom attention code to maintain.",
    "int8": "bitsandbytes INT8 targets CUDA GPUs and failed on the 128 -> 6 head, so a tablet runtime would need a different path; no speed benefit was shown even on the T4.",
    "student": f"Possible rare-class trade-off: the decline is smaller than the baseline's seed spread, so it is not confirmed, but all three paired seeds were lower ({STUDENT_RARE_MISSED} more missed rare documents out of {RARE_DOCS_TOTAL} across seeds).",
}
BROW = {}
for i, k in enumerate(KEYS):
    r_ = 4 + i; BROW[k] = r_
    d1 = DROW[(k, 1)]
    rr_ = 5 + i              # 1-indexed row
    f_text = F(text_wrap=True, valign="top", **BORD)
    ws.write(r_, 0, NAMES[k], F(bold=True, text_wrap=True, valign="top", font_size=12, **BORD))
    ws.write(r_, 1, PRECISION[k], f_text)
    ws.write_formula(r_, 2, f"={DN}!{rc(d1, 11)}", F(num_format="#,##0", valign="top", **BORD), m[k]["params"])
    ws.write_formula(r_, 3, f"=1-{rc(r_, 2)}/{rc(4, 2)}", F(num_format="0.0%", valign="top", **BORD), m[k]["d_params"])
    ws.write_formula(r_, 4, f"={DN}!{rc(d1, 10)}", F(num_format="0.00", valign="top", **BORD), m[k]["storage"])
    ws.write_formula(r_, 5, f"=1-{rc(r_, 4)}/{rc(4, 4)}", F(num_format="0.0%", valign="top", **BORD), m[k]["d_storage"])
    ws.write_formula(r_, 6, f"={QN}!{rc(QMEAN[k], 2)}", F(num_format="0.00%", valign="top", bold=True, **BORD), m[k]["acc"])
    ws.write_formula(r_, 7, f"={QN}!{rc(QSTD[k], 2)}", F(num_format='0.00" pp"', valign="top", **BORD), m[k]["acc_sd"] * 100)
    ws.write_formula(r_, 8, f"=({rc(r_, 6)}-{rc(4, 6)})*100", F(num_format='+0.00" pp";-0.00" pp";0.00" pp"', valign="top", **BORD), m[k]["d_acc_pp"])
    ws.write_formula(r_, 9, f"={QN}!{rc(QMEAN[k], 5)}", F(num_format="0.00%", valign="top", bold=True, **BORD), m[k]["rare"])
    ws.write_formula(r_, 10, f"={QN}!{rc(QSTD[k], 5)}", F(num_format='0.00" pp"', valign="top", **BORD), m[k]["rare_sd"] * 100)
    ws.write_formula(r_, 11, f"=({rc(r_, 9)}-{rc(4, 9)})*100", F(num_format='+0.00" pp";-0.00" pp";0.00" pp"', valign="top", **BORD), m[k]["d_rare_pp"])
    ws.write_formula(r_, 12, f"={DN}!{rc(d1, 2)}", F(num_format="0.00", valign="top", bold=True, **BORD), m[k]["p50"])
    ws.write_formula(r_, 13, f"={DN}!{rc(d1, 3)}", F(num_format="0.00", valign="top", bold=True, **BORD), m[k]["p95"])
    ws.write_formula(r_, 14, f"={rc(r_, 13)}/{rc(4, 13)}-1", F(num_format="+0.0%;-0.0%;0.0%", valign="top", **BORD), m[k]["d_p95"])
    ws.write_formula(r_, 15, f"={DN}!{rc(d1, 8)}", F(num_format="0.00", valign="top", bold=True, **BORD), m[k]["peak"])
    ws.write_formula(r_, 16, f"={rc(r_, 15)}/{rc(4, 15)}-1", F(num_format="+0.0%;-0.0%;0.0%", valign="top", **BORD), m[k]["d_peak"])
    ws.write(r_, 17, ADV[k], f_text); ws.write(r_, 18, TRD[k], f_text); ws.write(r_, 19, RSK[k], f_text)
    ws.write_number(r_, 20, RANK[k], F(bold=True, font_size=18, align="center", valign="vcenter", font_color=NAVY, **BORD))
    ws.set_row(r_, max(est_h(ADV[k], 33), est_h(TRD[k], 33), est_h(RSK[k], 33), est_h(PRECISION[k], 29), 78))
best_fmt = wb.add_format({"bg_color": GREEN, "bold": True})
for col, fn in ((4, "MIN"), (6, "MAX"), (9, "MAX"), (12, "MIN"), (13, "MIN"), (15, "MIN")):
    L = cn(col)
    ws.conditional_format(4, col, 7, col, {"type": "formula", "criteria": f"=ROUND({L}5,9)=ROUND({fn}(${L}$5:${L}$8),9)", "format": best_fmt})
r = 9
ws.merge_range(r, 0, r, 16, "How to read this board", SEC); r += 1
def grp(pairs, with_ratio=False):
    out = {}
    for k, mm in pairs:
        out.setdefault(k, []).append(LABEL[mm] + (f" {DIFF[(k, mm)]['ratio']:.2f}x" if with_ratio else ""))
    return "; ".join(f"{NAMES[k]} ({', '.join(v)})" for k, v in out.items())
within_txt = grp(WITHIN)
beyond_txt = grp(BEYOND, with_ratio=True)
bnotes = [
    "Green fill marks the best value in a column (highest accuracy and recall; lowest storage, latency and memory). It is a guide, not a ranking; see the Recommendation sheet.",
    f"Differences smaller than the observed seed-to-seed spread should be read as similar. Within seed spread: {within_txt}. Larger than the seed spread (only 3 seeds, no significance test run): {beyond_txt}. Details: Quality by Seed sheet.",
    f"Rare classes (Hazmat, Oversize / Heavy Haul) have only {SUPPORT['hazmat']} test documents each, so one document is {100 / SUPPORT['hazmat']:.2f} pp. The Distilled Student's rare-class macro recall is {abs(m['student']['d_rare_pp']):.2f} pp lower than the baseline; see Per-Class Results for each class.",
    f"Peak GPU Memory is PyTorch allocator memory and includes a constant {M['baseline']['bg']:.2f} MB that was already allocated before any model was loaded, so differences between rows are smaller in percentage terms than the model sizes alone suggest. Details: Benchmark Detail sheet.",
    "Measured on a Tesla T4 GPU as a controlled proxy. Android tablet hardware and runtimes were not tested, and the data is synthetic.",
]
for t in bnotes:
    ws.merge_range(r, 0, r, 16, t, NOTE); ws.set_row(r, est_h(t, 200)); r += 1
ws.merge_range(r + 1, 0, r + 1, 16, f"Recommended to ship: {NAMES['student']} (Rank 1), subject to checking rare-class recall on real Brightline documents. Full reasoning: Recommendation sheet.",
               F(bold=True, font_size=13, font_color=NAVY, bg_color=GREEN, text_wrap=True, **BORD)); ws.set_row(r + 1, 30)
ws.freeze_panes(4, 1)
ws.set_zoom(80)

# ==============================================================================================================
# Sheet 7: Methodology  (assignment audit appended after the recommendation is defined below)
# ==============================================================================================================
cnt = D["counts"]
tok = D["tok"]
dsha = D["sha"]
envm = META["env"]
METH = [
    ("SECTION", "Dataset"),
    ("Dataset", "Synthetic bill-of-lading text generated by a seeded script (data seed 42), not Brightline production data.", "bol_experiment.ipynb, Part 1 data cell"),
    ("Size and split", f"15,000 documents: {sum(cnt['train']):,} train / {sum(cnt['val']):,} validation / {sum(cnt['test']):,} test (70% / 10% / 20%), stratified by class.", f"{NB}, cell 5 output"),
    ("Classes", "Six: General Dry (34%), Retail Pallet (26%), Refrigerated (20%), Fragile / High-Value (12%), Hazmat (4%), Oversize / Heavy Haul (4%).", "bol_experiment.ipynb, Part 1 data cell"),
    ("Rare classes", f"Hazmat and Oversize / Heavy Haul, 4% each. Test support {SUPPORT['hazmat']} documents each; training counts {cnt['train'][4]} and {cnt['train'][5]}.", f"{NB}, cell 5 output"),
    ("Training-only label noise", f"3% of training labels were flipped to a different random class ({D['noise'][0]} of {int(D['noise'][1]):,} = {D['noise'][2]}%). Validation and test labels are clean (verified by regenerating the data without noise).", f"{NB}, cell 5 and verification-check outputs"),
    ("Input-level corruption (train, validation and test)", "OCR character substitutions and drops (per-document error rate 0%, 1%, 3% or 6%), missing class-specific fields (25% each), missing commodity line (5%), conflicting instruction borrowed from another class (15%), truncated values (6%), irregular spacing. Corruption never changes the label.", "bol_experiment.ipynb, Part 1 constants"),
    ("Dataset SHA-256", dsha, f"{NB}, cell 5 output; verified identical in every configuration run"),
    ("SECTION", "Tokenizer and models"),
    ("Tokenizer", "Lowercase; split into alphanumeric runs and single punctuation marks; vocabulary built from training texts only (minimum frequency 2); padding id 0, unknown id 1.", "bol_experiment.ipynb, Part 1 tokenizer cell"),
    ("Vocabulary and length", f"Vocabulary size {D['vocab']:,}. Fixed sequence length 128. Mean document length {tok['train']['mean_len']:.1f} tokens (maximum {tok['test']['max_len_before_trunc_cap']}), nothing truncated, so about {100 * (1 - tok['train']['nonpad_tokens'] / (tok['train']['n_docs'] * 128)):.0f}% of positions are padding. Unknown-token rate {D['unk']['test'] * 100:.1f}% on test.", f"{NB}, cell 7 output"),
    ("FP32 Baseline Transformer", f"Token + learned position embeddings, 4 pre-LN blocks (d_model 128, 4 heads, feed-forward 256, GELU), padding-aware softmax attention, mean pooling over non-padding tokens, 6-class linear head. {M['baseline']['params']:,} parameters ({M['baseline']['nonemb']:,} non-embedding).", "bol_experiment.ipynb, Part 1 model cell"),
    ("FP32 Linear Attention", f"Same network with softmax attention replaced by non-causal linear attention using phi(x) = ELU(x) + 1; padded keys and values are zeroed; no T x T matrix is built; identical parameter count ({M['linear']['params']:,}). Verified against an explicit masked reference and a padding-invariance test.", "bol_experiment.ipynb, Part 2"),
    ("INT8 method", "Post-training INT8 with bitsandbytes LLM.int8 (Linear8bitLt, outlier threshold 6.0, no calibration data) applied to the trained FP32 baseline checkpoints. 16 hidden Linear layers are INT8 (99.85% of Linear weights); activations are cast to FP16 inside those layers.", "bol_experiment.ipynb, Part 3"),
    ("FP32 classification-head limitation", "The six-output head (128 -> 6) stayed FP32 because the bitsandbytes INT8 path failed for that shape on the Tesla T4 with a cuBLASLt runtime error (ldc = 6). The 128 -> 128, 128 -> 256, 128 -> 384 and 256 -> 128 shapes passed. The model is therefore a mixed-precision configuration, not fully INT8; embeddings and LayerNorm also stay FP32.", f"{NB}, cell 28 output"),
    ("Distilled student", f"d_model 64, 2 layers, 2 heads, feed-forward 128; same vocabulary, tokenizer, length 128, pre-LN structure, padding-aware attention and pooling. {M['student']['params']:,} parameters ({M['student']['nonemb']:,} non-embedding).", f"{NB}, cell 34 output"),
    ("Knowledge distillation", "Teacher = the FP32 baseline of the same seed. Loss = 0.5 x cross-entropy(student, noisy training labels) + 0.5 x T^2 x KL(teacher || student), T = 2, KL with batchmean reduction on temperature-softened distributions. Teacher logits were cached for train and validation only, with the teacher in eval mode, no gradients; the test set was never used for distillation.", "bol_experiment.ipynb, Part 4"),
    ("SECTION", "Training protocol"),
    ("Optimizer and schedule", "AdamW, learning rate 1e-3, weight decay 0.01, 100 warm-up steps then linear decay to zero, batch size 64, 8 epochs, FP32, plain cross-entropy (no class weights).", "bol_experiment.ipynb, Part 1 training cell"),
    ("Seeds", "Training seeds 0, 1, 2 for every trained model (INT8 is derived from each baseline seed). Dataset seed 42. Benchmark document-selection seed 2025.", "bol_experiment.ipynb"),
    ("Evaluation policy", "Final-epoch model is evaluated; validation accuracy is only monitored. The test set is evaluated once per model after training and never used for selection, calibration or distillation.", "bol_experiment.ipynb, training cells and verification checks"),
    ("SECTION", "Hardware and software"),
    ("GPU", f"{envm['gpu_name']}, compute capability {envm['gpu_compute_capability']}, one Google Colab session.", "results/benchmark_meta.json"),
    ("Software", f"PyTorch {envm['pytorch_version']}, CUDA {envm['cuda_version']}, bitsandbytes {envm['bitsandbytes_version']}, Python {envm['python_version']}.", "results/benchmark_meta.json"),
    ("Determinism", "Seeds fixed in Python, NumPy and PyTorch; cuDNN deterministic mode and torch.use_deterministic_algorithms(True, warn_only=True) requested. CUDA execution is not guaranteed to be bitwise deterministic: PyTorch warned that the softmax-attention backward pass uses a non-deterministic kernel, so exact GPU reruns may differ slightly.", f"{NB}, training cell warning output"),
    ("SECTION", "Benchmark protocol"),
    ("Workload", f"Batch size 1, sequence length 128 (input shape 1 x 128), the same 100 test documents in the same order for every model (seeded index list, sha256 {META['workload']['index_sha256']}). model.eval(), torch.no_grad(), no autocast beyond what the INT8 layers do internally. Seed-0 models.", "results/benchmark_meta.json"),
    ("Warm-up, trials, synchronization", "15 untimed warm-up forwards, then torch.cuda.synchronize(); 100 timed trials, each: synchronize, start time.perf_counter(), one forward pass, synchronize, stop. Model loading, tokenization, CPU-to-GPU transfer and warm-up were not timed. p50 and p95 use numpy percentile (linear interpolation).", "results/benchmark_raw_latency.csv, benchmark_meta.json"),
    ("Peak-memory method", "Before each model: previous model deleted, gc.collect(), torch.cuda.empty_cache(), synchronize. After loading the new model on CUDA: torch.cuda.reset_peak_memory_stats(), resident = torch.cuda.memory_allocated(), then warm-up and timed trials, then peak = torch.cuda.max_memory_allocated(). PyTorch allocator memory only; it excludes the CUDA context and driver memory. A constant "
     f"{M['baseline']['bg']:.4f} MB was allocated before every model was loaded.", "results/benchmark_meta.json, benchmark_memory_diagnostics.csv"),
    ("Baseline repeatability test", f"The FP32 baseline was benchmarked twice (first, and again after the other three models). p50 differed by {REP_P50 * 100:.2f}% and p95 by {REP_P95 * 100:.2f}% against a predetermined 15% threshold: {'both within the threshold' if STABLE else 'threshold exceeded'}.", "results/benchmark_meta.json"),
    ("Why latency uses one seed", "Latency and memory use seed 0. Across seeds the architecture, tensor shapes, precision and execution path are unchanged, so repeated latency trials on one representative trained seed are used while seed-to-seed variation is measured for quality (three seeds).", "Benchmark design"),
    ("SECTION", "Known limitations"),
    ("L1", "Synthetic data may not represent real Brightline traffic, document layouts or error patterns.", ""),
    ("L2", "Only one GPU type was benchmarked (Tesla T4). Android tablet hardware and runtimes were not tested; bitsandbytes INT8 is a CUDA-GPU method.", ""),
    ("L3", "Shared Colab hardware can introduce timing variation. Short bursts of slower trials appeared in every run and mainly affect p95, which is why p50 is the more robust comparison.", ""),
    ("L4", "CUDA execution is not guaranteed to be bitwise deterministic.", ""),
    ("L5", "Sequence length was fixed at 128 (documents average about 44 tokens, so most positions are padding).", ""),
    ("L6", "Linear attention may have different relative performance at longer sequences; that was not measured.", ""),
    ("L7", "INT8 used a mixed-precision configuration because the six-output classification head failed on the tested bitsandbytes / cuBLASLt path.", ""),
    ("L8", f"Rare classes had smaller test support ({SUPPORT['hazmat']} documents each), so rare-class results are noisier than overall accuracy.", ""),
    ("L9", "Scaling-law results are formula-based estimates, not measurements.", ""),
    ("L10", "Only three seeds were run and no statistical significance test was performed; standard deviations from three values are themselves uncertain.", ""),
    ("L11", "Memory figures are PyTorch allocator values and include a constant pre-existing allocation; they are not total GPU or device memory.", ""),
    ("L12", "The distilled student depends on a trained teacher; teacher training and teacher-inference cost are not part of the student's own 6ND estimate.", ""),
    ("L13", f"Per-seed quality values were read from the 4-decimal tables printed in the executed notebook and converted to exact fractions (correct documents / support); they reproduce the full-precision means and standard deviations in benchmark_summary.csv. The original per-seed JSON files from the Colab session were not available in this workspace.", ""),
]
ws = ws_m
ws.set_column("A:A", 34); ws.set_column("B:B", 120); ws.set_column("C:C", 52)
ws.write(0, 0, "Methodology and Known Limitations", TITLE); ws.set_row(0, 30)
ws.write(1, 0, "A concise record of how the results were produced. Source files are named for every item.", SUB)
r = 3
for item in METH:
    if item[0] == "SECTION":
        ws.merge_range(r, 0, r, 2, item[1], SEC); ws.set_row(r, 22); r += 1; continue
    topic, detail, src = item
    if topic.startswith("L") and topic[1:].isdigit():
        ws.write(r, 0, "Limitation " + topic[1:], TXTB)
    else:
        ws.write(r, 0, topic, TXTB)
    ws.write(r, 1, detail, TXT); ws.write(r, 2, src, F(text_wrap=True, valign="top", font_color=GREY, font_size=10, **BORD))
    ws.set_row(r, max(est_h(detail, 120), est_h(topic, 34), est_h(src, 52, pt=10)))
    r += 1
METH_END_ROW = r
ws.freeze_panes(3, 1)

# ==============================================================================================================
# Sheet 8: Recommendation
# ==============================================================================================================
ws = ws_x
ws.set_column("A:A", 8); ws.set_column("B:B", 30); ws.set_column("C:F", 44); ws.set_column("G:G", 44)
ws.write(0, 0, "Recommendation", TITLE); ws.set_row(0, 30)
ws.write(1, 0, "Evidence-based qualitative ranking. No hidden weighted score is used. Differences inside the observed seed variation are described as similar, not as wins or losses.", SUB)
ws.merge_range(3, 0, 3, 6, "Ranking", SEC)
for c, h in enumerate(["Rank", "Configuration", "Why it ranks here", "What Brightline gains", "What Brightline gives up", "Residual risk"]):
    ws.write(4, c, h, HDR)
ws.set_row(4, 30)
d_s = DIFF[("student", "rare_macro_recall")]
RK = {
    "student": (
        f"Best on every efficiency measure: p50 {spct(m['student']['d_p50'])}, p95 {spct(m['student']['d_p95'])}, peak GPU memory {spct(m['student']['d_peak'])}, weight storage {spct(-m['student']['d_storage'])}, parameters {spct(-m['student']['d_params'])} versus the baseline, with overall accuracy not lower ({pct(m['student']['acc'], 2)} vs {pct(m['baseline']['acc'], 2)}, {pp(m['student']['d_acc_pp'])}). "
        f"Its rare-class macro recall is {abs(m['student']['d_rare_pp']):.2f} pp lower, but that gap is {d_s['ratio']:.2f} of the larger seed spread, so it is possible but not confirmed.",
        "The smallest and fastest model, with headroom for slower tablet hardware and the same architecture family as the baseline (no special runtime).",
        f"About one percentage point of mean rare-class macro recall (hazmat {pct(hz('student'), 1)} vs {pct(hz('baseline'), 1)}); that is {STUDENT_RARE_MISSED} more missed rare documents out of {RARE_DOCS_TOTAL} across three seeds. It also needs a trained teacher and a distillation step.",
        "Rare-class (hazmat, oversize / heavy-haul) performance on real Brightline documents is unverified; the measured decline sits inside seed variation, but every paired seed was lower. Synthetic data, one GPU type, 120 test documents per rare class."),
    "baseline": (
        f"The reference configuration: simplest and established, with similar overall accuracy ({pct(m['baseline']['acc'], 2)}). It ranks above Linear Attention and INT8 because it is faster than both while their quality is similar to or only slightly better than its own; it ranks below the student because it needs about {m['baseline']['storage'] / m['student']['storage']:.1f}x the weight storage and {m['baseline']['p95'] / m['student']['p95']:.1f}x the p95 latency for no measured accuracy advantage.",
        "Simple, established behavior with no compression, custom attention or distillation; the natural fallback if rare-class validation of the student fails.",
        f"Larger weights ({m['baseline']['storage']:.2f} MB), higher p95 latency ({m['baseline']['p95']:.2f} ms) and higher peak GPU memory ({m['baseline']['peak']:.2f} MB, tied with Linear Attention) than the student.",
        f"Same synthetic-data and rare-class sampling limits; its rare-class recall varies by {m['baseline']['rare_sd'] * 100:.2f} pp (std dev) across seeds."),
    "linear": (
        f"Highest mean overall accuracy ({pct(m['linear']['acc'], 2)}, {pp(m['linear']['d_acc_pp'])}; above every baseline seed, {DIFF[('linear', 'overall_test_acc')]['ratio']:.1f}x the larger seed spread). But at sequence length 128 it was slower than the baseline (p50 {spct(m['linear']['d_p50'])}, p95 {spct(m['linear']['d_p95'])}) and did not reduce memory or storage; rare-class recall is similar within seed variation. "
        "Ranks 2 and 3 are a close judgment: Linear Attention is the better choice only if a half-point of accuracy is worth about a quarter more latency.",
        "The highest measured overall accuracy and an attention cost that grows linearly with length (benefit at longer sequences was not tested).",
        f"About {m['linear']['p95'] - m['baseline']['p95']:.1f} ms more p95 latency than the baseline and no savings in weights ({m['linear']['storage']:.2f} MB) or peak memory.",
        "Behavior at longer sequences was not measured; custom attention code to maintain; the accuracy edge is small and rare-class results did not improve beyond seed variation."),
    "int8": (
        f"Accuracy is unchanged ({pct(m['int8']['acc'], 2)}, {pp(m['int8']['d_acc_pp'])}) and weights are {pct(m['int8']['d_storage'])} smaller, but it was the slowest model on the T4 (p50 {spct(m['int8']['d_p50'])}, p95 {spct(m['int8']['d_p95'])}) and peak GPU memory fell only {pct(-m['int8']['d_peak'])}. The storage gain is limited because embeddings and the head stay FP32.",
        f"A smaller weight file ({m['int8']['storage']:.2f} MB) with no accuracy loss.",
        f"More than double the baseline's latency; a mixed-precision configuration because the 6-class head could not run in INT8; a dependency on bitsandbytes.",
        "bitsandbytes INT8 is a CUDA-GPU path; Android tablet inference would need a different runtime, and the speed benefit was not demonstrated even on the T4."),
}
order = ["student", "baseline", "linear", "int8"]
r = 5
RKROW = {}
for i, k in enumerate(order):
    why, gain, give, risk = RK[k]
    RKROW[k] = r
    ws.write_number(r, 0, i + 1, F(bold=True, font_size=18, align="center", valign="top", font_color=NAVY, bg_color=GREEN if i == 0 else "white", **BORD))
    ws.write(r, 1, NAMES[k], F(bold=True, text_wrap=True, valign="top", font_size=12, bg_color=GREEN if i == 0 else "white", **BORD))
    for c, t in enumerate((why, gain, give, risk)):
        ws.write(r, 2 + c, t, TXT)
    ws.set_row(r, max(est_h(why, 44), est_h(gain, 44), est_h(give, 44), est_h(risk, 44)))
    r += 1
r += 1
ws.merge_range(r, 0, r, 6, "Final recommendation for Sandra", SEC); r += 1
FINAL = (f"Ship the Distilled Student: on the Tesla T4 it cut model weight storage by {pct(m['student']['d_storage'])} and p95 latency by {pct(-m['student']['d_p95'])} versus the FP32 baseline "
         f"while overall accuracy stayed similar ({pct(m['student']['acc'], 2)} vs {pct(m['baseline']['acc'], 2)}, a small measured difference). "
         f"The main residual risk Sandra is accepting is a possible rare-class trade-off, because mean rare-class macro recall was {pct(m['student']['rare'], 2)} vs {pct(m['baseline']['rare'], 2)} (about one point lower, smaller than the baseline's seed-to-seed spread and therefore not confirmed). "
         "Before full rollout, check hazmat and oversize / heavy-haul recall on real Brightline documents and keep the FP32 baseline as the fallback.")
n_sent = len(re.split(r"(?<=[.)])\s+(?=[A-Z])", FINAL))
assert n_sent <= 3, (n_sent, FINAL)
ws.merge_range(r, 0, r, 6, FINAL, F(bold=True, font_size=13, text_wrap=True, valign="top", bg_color=GREEN, **BORD)); ws.set_row(r, est_h(FINAL, 215, pt=13) + 10); r += 2

ws.merge_range(r, 0, r, 6, "Evidence by criterion (values link to the Trade-off Board and Quality by Seed sheets; no weights are applied)", SEC); r += 1
for c, h in enumerate(["", "Criterion", "FP32 Baseline Transformer", "FP32 Linear Attention", "Mixed-Precision INT8 Baseline", "Distilled Student", "Reading"]):
    ws.write(r, c, h, HDR)
ws.set_row(r, 34); r += 1
def link(col, k): return f"={BN}!{rc(BROW[k], col)}"
EV = [
    ("Overall accuracy mean", 6, "0.00%", "acc", "Linear Attention highest; the others are within about half a point of one another (see seed-spread rows)."),
    ("Rare-class macro recall", 9, "0.00%", "rare", "Student lowest; the gap to the baseline is smaller than the baseline's seed spread."),
    ("p50 latency ms", 12, "0.00", "p50", "Student fastest; INT8 slowest."),
    ("p95 latency ms", 13, "0.00", "p95", "Same ordering as p50; p95 is more sensitive to transient timing spikes."),
    ("Peak GPU memory MB", 15, "0.00", "peak", "Includes a constant pre-existing allocation; the spread between rows is small."),
    ("Weight storage MB", 4, "0.00", "storage", "Student smallest; INT8 limited by FP32 embeddings."),
    ("Total parameters", 2, "#,##0", "params", "Quantization and linear attention do not change the parameter count."),
]
for label, col, nf, key, reading in EV:
    ws.write(r, 0, "", F(**BORD)); ws.write(r, 1, label, TXTB)
    for ci, k in enumerate(KEYS):
        ws.write_formula(r, 2 + ci, link(col, k), F(num_format=nf, align="center", **BORD), m[k][key])
    ws.write(r, 6, reading, TXT); ws.set_row(r, 34); r += 1
for label, metric, reading in (
        ("Overall accuracy difference vs baseline / larger seed std", "overall_test_acc", "Above 1.00 = larger than the seed spread (3 seeds, not a significance test)."),
        ("Rare-class recall difference vs baseline / larger seed std", "rare_macro_recall", "Below 1.00 = smaller than the seed spread: treat as similar, but keep watching rare classes.")):
    ws.write(r, 0, "", F(**BORD)); ws.write(r, 1, label, TXTB)
    ws.write(r, 2, "reference", F(align="center", font_color=GREY, **BORD))
    for ci, k in enumerate(KEYS[1:]):
        ws.write_formula(r, 3 + ci, f"={QN}!{rc(QDIFF[(k, metric)], 8)}", F(num_format="0.00", align="center", **BORD), DIFF[(k, metric)]["ratio"])
    ws.write(r, 6, reading, TXT); ws.set_row(r, 48); r += 1
cplx = {"baseline": "Simplest; no extra dependencies.", "linear": "Custom attention code; longer-sequence behavior untested.",
        "int8": "bitsandbytes dependency; head must stay FP32; CUDA-GPU only.", "student": "Needs a trained teacher and a distillation step."}
ws.write(r, 0, "", F(**BORD)); ws.write(r, 1, "Implementation complexity and limitations", TXTB)
for ci, k in enumerate(KEYS):
    ws.write(r, 2 + ci, cplx[k], F(text_wrap=True, align="center", **BORD))
ws.write(r, 6, "Qualitative; see Methodology for the INT8 head limitation.", TXT); ws.set_row(r, 48); r += 2

ws.merge_range(r, 0, r, 6, "Results to treat cautiously (difference from the baseline is smaller than the observed seed spread)", SEC); r += 1
for k, mm in WITHIN:
    v = DIFF[(k, mm)]
    t = f"{NAMES[k]}, {LABEL[mm]}: mean difference {v['d'] * 100:+.2f} pp vs baseline, {v['ratio']:.2f} of the larger seed std dev; per-seed paired differences {', '.join(f'{x * 100:+.2f}' for x in v['seed_d'])} pp."
    ws.merge_range(r, 0, r, 6, t, NOTE); ws.set_row(r, 20); r += 1
r += 1
ws.merge_range(r, 0, r, 6, "How strongly to read these results", SEC); r += 1
for t in ("No statistical significance test was run. With three seeds, wording such as 'similar within observed seed variation', 'small measured difference' and 'possible rare-class trade-off' is descriptive only.",
          "Two comparisons exceed the seed spread (Linear Attention overall accuracy, and the Student's overall accuracy at about 1.2x), but both are small (about half a percentage point or less) and rest on three seeds.",
          "All timing comes from one Colab Tesla T4 and one seed-0 model per configuration; Android tablet performance was not measured."):
    ws.merge_range(r, 0, r, 6, t, NOTE); ws.set_row(r, est_h(t, 215)); r += 1
ws.set_zoom(90)

# ==============================================================================================================
# Methodology sheet: assignment audit block (statuses filled in from build-time checks)
# ==============================================================================================================
AUDIT = [
    ("Four configurations: baseline, architecture variant, quantized, distilled student", True, "Trade-off Board"),
    ("Measured overall accuracy (three seeds)", True, "Quality by Seed; Trade-off Board"),
    ("Per-class results (all six classes)", True, "Per-Class Results"),
    ("Rare-class results (Hazmat, Oversize / Heavy Haul)", True, "Quality by Seed; Per-Class Results"),
    ("Three-seed quality measurements for every configuration", True, "Quality by Seed"),
    ("p50 and p95 latency", True, "Trade-off Board; Benchmark Detail"),
    ("Peak GPU memory", True, "Trade-off Board; Benchmark Detail"),
    ("Repeated benchmark trials (100 per configuration, plus a baseline repeat)", len(RAW) == 500, "Raw Benchmark Runs"),
    ("Methodology notes", True, "Methodology"),
    ("Fixed-seed documentation", True, "Methodology: Seeds, Determinism"),
    ("Warm-up documentation (15 iterations)", True, "Methodology: Warm-up, trials, synchronization"),
    ("CUDA synchronization documentation", True, "Methodology: Warm-up, trials, synchronization"),
    ("Parameter counts (total and non-embedding)", True, "Trade-off Board; Benchmark Detail; Scaling Law"),
    ("Weight storage", True, "Trade-off Board; Benchmark Detail"),
    ("6ND compute estimates", True, "Scaling Law (labeled as estimates)"),
    ("20N compute-optimal token estimates", True, "Scaling Law (labeled as estimates)"),
    ("10x scaling projection", True, "Scaling Law (labeled as estimates)"),
    ("Ranked recommendation", True, "Recommendation"),
    ("Residual risks", True, "Trade-off Board; Recommendation"),
    ("Quantization limitation detected and reported", True, "Methodology: FP32 classification-head limitation"),
    ("Baseline repeatability within the predetermined 15% threshold", STABLE, "Benchmark Detail"),
]
ws = ws_m
r = METH_END_ROW + 1
ws.merge_range(r, 0, r, 2, "Assignment coverage checklist (computed when the workbook was built)", SEC); r += 1
for c, h in enumerate(["Requirement", "Where in this workbook", "Status"]):
    ws.write(r, c, h, HDR)
r += 1
for req, ok, where in AUDIT:
    ws.write(r, 0, req, TXT); ws.write(r, 1, where, TXT)
    ws.write(r, 2, "Covered" if ok else "FLAG: not met", F(bold=True, valign="top", bg_color=GREEN if ok else AMBER, **BORD))
    ws.set_row(r, est_h(req, 34)); r += 1

ws_b.activate()
wb.close()
print("wrote", OUT)
