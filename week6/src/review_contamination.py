"""
Records the manual-inspection verdicts for the n-gram contamination matches.

Process: every DISTINCT matched span in results/contamination_matches.csv was read by a person
(here: the assistant, with the user able to re-read the same table in
results/contamination_unique_spans.csv). This script then applies the verdict rule that the
reading established and CHECKS that every row obeys it, so no row is labelled without being
verified. A row that does not fit a known explanation is left blank and the script exits
non-zero: unexplained overlap must be investigated, not waved through.

Known explanation (the only one found in the first sweep, see results/contamination_review.md):
    CLOSED_LIBRARY_STRING  The span is a policy library display string
    "MRP-#### (<Service> Coverage Policy)", optionally with ONE coincidentally adjacent word
    ("policy" before it, "we" after it). The policy library and service names are shared between
    training and evaluation by design (the model must be able to cite policy IDs), so this overlap
    is expected. It is a structural-overlap channel, reported as such, not a leaked evaluation item.

Outputs (results/)
    contamination_matches.csv        verdict + reason columns filled in (in place)
    contamination_unique_spans.csv   one row per distinct span with counts
    contamination_review.json        flagged vs. unexplained counts per dataset and n
"""
import argparse
import json
import os
import sys

import pandas as pd

from appeals_common import POLICIES, normalize_text

CORES = {normalize_text(f"{p['policy_id']} ({p['title']})"): p["policy_id"] for p in POLICIES}
NEIGHBOURS = {"policy": "before", "we": "after"}


def explain(span: str):
    """Returns (verdict, reason) or None if the span is not a known explained pattern."""
    toks = span.split()
    for core, pid in CORES.items():
        c = core.split()
        for pre in ([], ["policy"]):
            for post in ([], ["we"]):
                if toks == pre + c + post and len(pre) + len(post) <= 1:
                    extra = (pre + post)
                    nb = f"; plus one coincidentally adjacent word '{extra[0]}' ({NEIGHBOURS[extra[0]]})" if extra else ""
                    return ("closed_library_string",
                            f"Library display string for {pid} (shared by design between train and held-out text){nb}.")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    path = os.path.join(args.results, "contamination_matches.csv")
    m = pd.read_csv(path, keep_default_na=False)

    unexplained = []
    for i, row in m.iterrows():
        if str(row["is_synthetic_control"]) == "True":
            # Control rows: the copied summary is the expected hit; the host notice can still carry
            # the shared library string, which is labelled as such rather than as the control.
            if row["field"] == "summary":
                m.at[i, "verdict"] = "expected (synthetic positive control)"
                m.at[i, "reason"] = "summary copied verbatim from a training record by construction"
            else:
                res = explain(row["matched_span"])
                if res is None:
                    unexplained.append(i)
                else:
                    m.at[i, "verdict"], m.at[i, "reason"] = res
            continue
        res = explain(row["matched_span"])
        if res is None:
            unexplained.append(i)
        else:
            m.at[i, "verdict"], m.at[i, "reason"] = res
    m.to_csv(path, index=False)

    real = m[m["is_synthetic_control"].astype(str) == "False"]
    unexplained = [i for i in unexplained if str(m.at[i, "is_synthetic_control"]) == "False" or m.at[i, "verdict"] == ""]
    uniq = (real.groupby(["n", "matched_span", "verdict"]).agg(rows=("item_id", "size"), items=("item_id", "nunique"),
                                                               datasets=("dataset", lambda x: ",".join(sorted(set(x))))).reset_index())
    uniq.to_csv(os.path.join(args.results, "contamination_unique_spans.csv"), index=False)

    summ = pd.read_csv(os.path.join(args.results, "contamination_summary.csv"))
    out = []
    for _, s in summ[~summ["is_synthetic_control"]].iterrows():
        sub = real[(real["dataset"] == s["dataset"]) & (real["n"] == s["n"])]
        unexpl_items = sub[sub["verdict"] == ""]["item_id"].nunique()
        out.append(dict(dataset=s["dataset"], n=int(s["n"]), items=int(s["items"]), flagged_items=int(s["flagged_items"]),
                        pct_flagged=float(s["pct_flagged"]), flagged_items_unexplained_after_review=int(unexpl_items),
                        pct_unexplained=round(100 * unexpl_items / s["items"], 2)))
    with open(os.path.join(args.results, "contamination_review.json"), "w") as f:
        json.dump(dict(rows=out, n_distinct_spans_read=int(len(uniq)), n_rows_verified=int(len(real) - len(unexplained)),
                       n_rows_unexplained=len(unexplained)), f, indent=2)
    print(pd.DataFrame(out).to_string(index=False))
    print(f"distinct spans read: {len(uniq)}; rows verified: {len(real) - len(unexplained)}; unexplained rows: {len(unexplained)}")
    if unexplained:
        print("UNEXPLAINED MATCHES, investigate:", m.loc[unexplained[:5], ["dataset", "item_id", "n", "matched_span"]].to_string())
        sys.exit(1)


if __name__ == "__main__":
    main()
