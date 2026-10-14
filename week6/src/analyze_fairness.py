"""
Fairness analysis of the matched-case generations. CPU only.

    python src/analyze_fairness.py \
        --gen seed1001=results/generations/fairness_seed1001.jsonl \
        --gen seed1002=results/generations/fairness_seed1002.jsonl \
        --gen seed1003=results/generations/fairness_seed1003.jsonl

Writes (results/)
    fairness_row_metrics.csv       one row per generation with every metric
    fairness_disaggregated.csv     metric x {language, age, language x age cell, overall}: mean, 95% CI,
                                   n_cases, n_generations
    fairness_slice_differences.csv paired slice differences with 95% CI, n, per-seed estimates and the
                                   pre-registered interpretation (marginal + stratified)

Interpretation (analysis_plan.md): CI excludes 0 -> evidence of a difference; CI includes 0 -> unresolved,
NOT evidence of equal performance. The three seeds are sampling seeds of ONE adapter and measure
generation variability only.
"""
import argparse
import json
import os

import pandas as pd

from appeals_common import load_jsonl
from score_task import score_output
from stats import cell_summary, paired_slice_difference

METRICS = [
    ("rationale_omission", "primary"),
    ("unsupported_policy", "secondary (exploratory)"),
    ("completeness", "secondary (exploratory)"),
    ("invalid_or_refusal", "secondary (exploratory)"),
    ("rationale_recall", "secondary (exploratory)"),
    ("rationale_omission_valid_only", "sensitivity to the primary metric"),
]
CONTRASTS = [("language", "Spanish", "English"), ("age", 75, 45)]


def score_generation(gen, rec):
    finished = gen.get("finished", True)
    raw = gen["raw_output"] if finished else ""
    row = score_output(raw, rec["gold"], rec["notice"], rec["summary"])
    row["truncated"] = int(not finished)
    return row


def build_rows(data_path, gen_specs):
    recs = {r["id"]: r for r in load_jsonl(data_path)}
    rows = []
    for spec in gen_specs:
        name, path = spec.split("=", 1)
        seed = int(name.replace("seed", ""))
        gens = {g["id"]: g for g in load_jsonl(path)}
        missing = set(recs) - set(gens)
        if missing:
            raise SystemExit(f"{name}: {len(missing)} generations missing, e.g. {sorted(missing)[:3]}")
        for rid, rec in recs.items():
            row = score_generation(gens[rid], rec)
            row.update(id=rid, base_id=rec["base_id"], language=rec["language"], age=rec["age"], seed=seed)
            rows.append(row)
    return pd.DataFrame(rows)


def disaggregate(df, metric):
    out = []
    levels = [("overall", None, lambda d: d.index == d.index)]
    levels += [(f"language={l}", "language", lambda d, l=l: d["language"] == l) for l in ("English", "Spanish")]
    levels += [(f"age={a}", "age", lambda d, a=a: d["age"] == a) for a in (45, 75)]
    levels += [(f"language={l} & age={a}", "cell", lambda d, l=l, a=a: (d["language"] == l) & (d["age"] == a))
               for l in ("English", "Spanish") for a in (45, 75)]
    for label, _kind, fn in levels:
        s = cell_summary(df, metric, fn(df))
        out.append(dict(metric=metric, group=label, estimate=s["estimate"], ci_low=s["ci_low"], ci_high=s["ci_high"],
                        n_cases=s["n"], n_generations=s["n_generations"]))
    return out


def differences(df, metric, role):
    out = []
    for slice_col, a, b in CONTRASTS:
        other = "age" if slice_col == "language" else "language"
        strata = [("marginal", df)] + [(f"{other}={v}", df[df[other] == v]) for v in sorted(df[other].unique())]
        for label, sub in strata:
            r = paired_slice_difference(sub, metric, slice_col, a, b)
            out.append(dict(
                metric=metric, role=role, slice=slice_col, contrast=f"{a} minus {b}", stratum=label,
                estimate=r["estimate"], ci_low=r["ci_low"], ci_high=r["ci_high"],
                estimate_pp=round(100 * r["estimate"], 2), ci_low_pp=round(100 * r["ci_low"], 2),
                ci_high_pp=round(100 * r["ci_high"], 2),
                n_cases=r["n_cases"], n_generations_a=r["n_generations_a"], n_generations_b=r["n_generations_b"],
                per_seed_estimates=json.dumps({str(k): round(v, 4) for k, v in r["per_seed_estimates"].items()}),
                interpretation=r["interpretation"],
            ))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/fairness_matched.jsonl")
    ap.add_argument("--gen", action="append", required=True)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    df = build_rows(args.data, args.gen)
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "fairness_row_metrics.csv"), index=False)

    dis, dif = [], []
    for metric, role in METRICS:
        dis += disaggregate(df, metric)
        dif += differences(df, metric, role)
    pd.DataFrame(dis).to_csv(os.path.join(args.out_dir, "fairness_disaggregated.csv"), index=False)
    pd.DataFrame(dif).to_csv(os.path.join(args.out_dir, "fairness_slice_differences.csv"), index=False)

    marg = pd.DataFrame(dif)
    marg = marg[marg["stratum"] == "marginal"]
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60)
    print(marg[["metric", "slice", "contrast", "estimate_pp", "ci_low_pp", "ci_high_pp", "n_cases", "n_generations_a", "n_generations_b", "interpretation"]].to_string(index=False))


if __name__ == "__main__":
    main()
