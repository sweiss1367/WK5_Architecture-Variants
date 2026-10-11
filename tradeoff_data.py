"""Data layer for the Week 5 trade-off workbook.

Everything here is read from existing result files. Nothing is estimated or typed in:
  * results/benchmark_summary.csv, benchmark_raw_latency.csv, benchmark_meta.json, benchmark_memory_diagnostics.csv
  * results/scaling_law_results.csv / .json
  * bol_experiment_cuda_run_v4_benchmark.ipynb (executed notebook): per-seed quality tables, class counts, token statistics

Per-seed accuracies are only available in that notebook as 4-decimal printed tables. Every one of them is
(correct documents) / (test support), so the exact fraction is recovered as round(printed * support) / support and
checked against the printed value. The recovered values are then cross-checked against the full-precision means and
standard deviations stored in benchmark_summary.csv.
"""
import json, re
import numpy as np
import pandas as pd

NB = "bol_experiment_cuda_run_v4_benchmark.ipynb"
KEYS = ["baseline", "linear", "int8", "student"]
NAMES = {"baseline": "FP32 Baseline Transformer", "linear": "FP32 Linear Attention",
         "int8": "Mixed-Precision INT8 Baseline", "student": "Distilled Student"}
CSV_NAME = {"FP32 baseline Transformer": "baseline", "FP32 linear-attention variant": "linear",
            "INT8 hidden layers + FP32 head (bitsandbytes)": "int8", "FP32 distilled student": "student"}
CLASSES = ["general_dry", "retail_pallet", "refrigerated", "fragile_high_value", "hazmat", "oversize_heavy"]
CLASS_LABEL = {"general_dry": "General Dry", "retail_pallet": "Retail Pallet", "refrigerated": "Refrigerated",
               "fragile_high_value": "Fragile / High-Value", "hazmat": "Hazmat", "oversize_heavy": "Oversize / Heavy Haul"}
RARE = {"hazmat", "oversize_heavy"}
REPORT_CELL = {"baseline": 14, "linear": 23, "int8": 31, "student": 38}     # cells of the executed notebook with the per-seed tables


def _cell_text(nb, i):
    out = []
    for o in nb["cells"][i].get("outputs", []):
        out.append("".join(o["text"]) if "text" in o else str(o.get("data", {}).get("text/plain", "")))
    return "\n".join(out)


def _seed_table(text):
    """First occurrence of each row name that carries exactly three numbers = the per-seed table."""
    rows = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] in ("overall_test_acc", "rare_macro_recall") + tuple(f"acc_{c}" for c in CLASSES):
            try:
                vals = [float(x) for x in parts[1:]]
            except ValueError:
                continue
            rows.setdefault(parts[0], vals)
    return rows


def load():
    nb = json.load(open(NB))
    t5, t7 = _cell_text(nb, 5), _cell_text(nb, 7)
    counts = eval(re.search(r"class counts per split (\{.*\})", t5).group(1))
    support = dict(zip(CLASSES, counts["test"]))
    n_test = sum(counts["test"])
    sha = re.search(r"dataset sha256: ([0-9a-f]{64})", t5).group(1)
    noise = re.search(r"train label noise: (\d+) of (\d+) labels flipped \(([\d.]+)%\)", t5)
    tok = json.loads(re.search(r"token stats: (\{.*?\n\})", t7, re.S).group(1))
    unk = eval(re.search(r"unk rate: (\{.*\})", t7).group(1))
    vocab = int(re.search(r"vocab size: (\d+)", t7).group(1))

    seeds = {}                                  # seeds[key][seed] = dict(metric -> exact fraction), plus counts
    audit = []
    for key in KEYS:
        tab = _seed_table(_cell_text(nb, REPORT_CELL[key]))
        assert set(tab) == {"overall_test_acc", "rare_macro_recall"} | {f"acc_{c}" for c in CLASSES}, (key, sorted(tab))
        seeds[key] = {}
        for s in range(3):
            d, k = {}, {}
            n_by = {"overall_test_acc": n_test, **{f"acc_{c}": support[c] for c in CLASSES}}
            for name, n in n_by.items():
                printed = tab[name][s]
                kk = round(printed * n)
                assert abs(kk / n - printed) <= 5.0001e-5, (key, s, name, printed, kk, n)
                k[name] = (kk, n)
                d[name] = kk / n
            rm = (d["acc_hazmat"] + d["acc_oversize_heavy"]) / 2
            assert abs(rm - tab["rare_macro_recall"][s]) <= 5.0001e-5, (key, s, rm, tab["rare_macro_recall"][s])
            d["rare_macro_recall"] = rm
            seeds[key][s] = {"value": d, "count": k, "printed": {n: tab[n][s] for n in tab}}

    summ = pd.read_csv("results/benchmark_summary.csv")
    summ["key"] = summ["Configuration"].map(CSV_NAME)
    summ = summ.set_index("key")
    for key in KEYS:                            # cross-check against the full-precision values in the summary CSV
        for metric, mcol, scol in (("overall_test_acc", "Overall accuracy mean (3 seeds)", "Overall accuracy std (3 seeds)"),
                                   ("rare_macro_recall", "Rare-class macro recall mean (3 seeds)", "Rare-class macro recall std (3 seeds)")):
            v = np.array([seeds[key][s]["value"][metric] for s in range(3)])
            dm, ds = abs(v.mean() - summ.loc[key, mcol]), abs(v.std(ddof=1) - summ.loc[key, scol])
            audit.append((key, metric, dm, ds))
            assert dm < 1e-12 and ds < 1e-12, (key, metric, dm, ds)

    raw = pd.read_csv("results/benchmark_raw_latency.csv")
    raw["key"] = raw["configuration"].map(CSV_NAME)
    meta = json.load(open("results/benchmark_meta.json"))
    mem = pd.read_csv("results/benchmark_memory_diagnostics.csv")
    sc_csv = pd.read_csv("results/scaling_law_results.csv")
    sc_json = json.load(open("results/scaling_law_results.json"))
    # seed-0 full test accuracy recorded by the benchmark (exact) must equal the recovered seed-0 value
    for r in meta["runs"]:
        if r["run"] == 1:
            assert abs(r["full_test_acc"] - seeds[r["key"]][0]["value"]["overall_test_acc"]) < 1e-12, r["key"]
    return dict(seeds=seeds, support=support, counts=counts, n_test=n_test, sha=sha, noise=noise.groups(), tok=tok, unk=unk,
                vocab=vocab, summ=summ, raw=raw, meta=meta, mem=mem, sc_csv=sc_csv, sc_json=sc_json, crosscheck=audit)


if __name__ == "__main__":
    D = load()
    print("support:", D["support"], "n_test", D["n_test"], "vocab", D["vocab"], "sha", D["sha"][:16], "noise", D["noise"])
    print("cross-check vs benchmark_summary.csv (max |diff| of mean / std):")
    for key, metric, dm, ds in D["crosscheck"]:
        print(f"  {key:8s} {metric:18s} mean diff {dm:.2e}  std diff {ds:.2e}")
    for key in KEYS:
        print(key, [f"{D['seeds'][key][s]['count']['overall_test_acc'][0]}/{D['n_test']}" for s in range(3)],
              "hazmat", [D['seeds'][key][s]['count']['acc_hazmat'][0] for s in range(3)],
              "oversize", [D['seeds'][key][s]['count']['acc_oversize_heavy'][0] for s in range(3)])
