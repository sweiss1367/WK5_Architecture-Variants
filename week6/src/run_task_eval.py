"""
Scores greedy task generations (val and eval_h) and writes the task-baseline table.

    python src/run_task_eval.py --item val=data/val.jsonl:results/generations/val_greedy.jsonl \
                                --item eval_h=data/eval_h.jsonl:results/generations/eval_h_greedy.jsonl

Writes results/task_scores_<name>.csv (per example) and results/task_baseline.csv
(metric, estimate, 95% CI, n). Binary rates use Wilson intervals over cases where the metric is
defined; rationale_recall uses a case-level bootstrap CI. This table is context for reading the
probe and fairness results, not a headline claim.
"""
import argparse
import os

import pandas as pd

from appeals_common import load_jsonl
from score_task import score_output
from stats import bootstrap_mean_ci, wilson

BINARY = [
    "invalid_or_refusal", "completeness", "rationale_omission", "rationale_omission_valid_only",
    "unsupported_policy", "policy_exact", "denial_reason_correct", "next_step_correct",
    "criteria_match", "missing_match", "any_unsupported_fact",
]


def score_file(data_path, gen_path):
    recs = {r["id"]: r for r in load_jsonl(data_path)}
    rows = []
    for g in load_jsonl(gen_path):
        rec = recs[g["id"]]
        finished = g.get("finished", True)
        row = score_output(g["raw_output"] if finished else "", rec["gold"], rec["notice"], rec["summary"])
        row["id"], row["truncated"] = g["id"], int(not finished)
        n_uns = row.get("n_unsupported_facts")
        row["any_unsupported_fact"] = None if n_uns is None else int(n_uns > 0)
        rows.append(row)
    if len(rows) != len(recs):
        raise SystemExit(f"{gen_path}: expected {len(recs)} generations, found {len(rows)}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--item", action="append", required=True, help="name=data.jsonl:generations.jsonl")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    table = []
    for spec in args.item:
        name, rest = spec.split("=", 1)
        data_path, gen_path = rest.split(":", 1)
        df = score_file(data_path, gen_path)
        df.to_csv(os.path.join(args.out_dir, f"task_scores_{name}.csv"), index=False)
        for m in BINARY:
            s = df[m].dropna()
            p, lo, hi = wilson(int(s.sum()), len(s))
            table.append(dict(dataset=name, metric=m, estimate=p, ci_low=lo, ci_high=hi, n=len(s), method="wilson"))
        b = bootstrap_mean_ci(df["rationale_recall"].dropna().values)
        table.append(dict(dataset=name, metric="rationale_recall", estimate=b["estimate"], ci_low=b["ci_low"],
                          ci_high=b["ci_high"], n=b["n"], method="bootstrap"))
        table.append(dict(dataset=name, metric="truncated_outputs", estimate=float(df["truncated"].mean()),
                          ci_low=None, ci_high=None, n=len(df), method="rate"))
    out = pd.DataFrame(table)
    out.to_csv(os.path.join(args.out_dir, "task_baseline.csv"), index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
