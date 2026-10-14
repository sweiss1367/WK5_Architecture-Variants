"""
n-gram overlap contamination sweep: evaluation text vs training text.

Procedure (fixed in analysis_plan.md):
    1. The datasets are frozen and hashed (MANIFEST.json) before this runs; the sweep refuses to
       run if the manifest does not verify.
    2. Normalise: Unicode NFKD, strip accents, lowercase, every non-alphanumeric run -> one space,
       split on spaces (word-level tokens).
    3. Index every n-gram (n = 8, 13, 20) of every training + validation text field.
    4. For every evaluation item, check each of its text fields against the index. An item is
       FLAGGED at n if any field shares at least one n-gram with train/val.
    5. Report flagged items and percentages per evaluation set and n; write EVERY match, as a
       maximal contiguous span, to a CSV for manual inspection (verdict/reason columns start blank).

Fields compared
    train/val side : notice, summary, and every string in the gold labels (incl. appeal_summary)
    eval side      : notice, summary, extra_user_text (probes), and every string in the gold labels
Fields are compared one at a time: an n-gram never spans two fields. The fixed JSON key scaffold
and the system prompt are not compared (identical for every example by design).

What this test cannot show: a negative result does NOT rule out paraphrase, translation, or
structural/template contamination, or contamination of the base model's pretraining data.
Gold labels and fact phrases are mostly shorter than 8 words, so they cannot produce n-grams at
these n at all; they are counted in `docs_shorter_than_n` rather than silently ignored.

The positive control is a SYNTHETIC eval-style record with a summary copied verbatim from a
training record. It is reported separately, with is_synthetic_control=True, and is excluded from
every percentage for the real evaluation sets.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

from appeals_common import load_jsonl, tokens
import manifest

N_VALUES = (8, 13, 20)
EVAL_SETS = {
    "eval_h": "data/eval_h.jsonl",
    "fairness_matched": "data/fairness_matched.jsonl",
    "probe_suite": "probes/probe_suite.jsonl",
}
CONTROL = "data/positive_control.jsonl"


def _gold_strings(gold):
    out = []
    for k, v in (gold or {}).items():
        if k == "_ids":
            continue
        if isinstance(v, str):
            out.append((f"gold.{k}", v))
        elif isinstance(v, list):
            out.extend((f"gold.{k}", x) for x in v if isinstance(x, str))
    return out


def item_fields(rec, with_gold=True):
    fields = [("notice", rec["notice"]), ("summary", rec["summary"])]
    if rec.get("extra_user_text"):
        fields.append(("extra_user_text", rec["extra_user_text"]))
    if with_gold:
        gold = rec.get("gold") or (rec.get("meta") or {}).get("gold")
        fields.extend(_gold_strings(gold))
    return fields


def build_index(train_val, n):
    """n-gram tuple -> up to 5 'recordid:field' sources."""
    index, short = {}, 0
    for rec in train_val:
        for fname, text in item_fields(rec, with_gold=True):
            t = tokens(text)
            if len(t) < n:
                short += 1
                continue
            src = f"{rec['id']}:{fname}"
            for i in range(len(t) - n + 1):
                g = tuple(t[i:i + n])
                lst = index.setdefault(g, [])
                if len(lst) < 5 and src not in lst:
                    lst.append(src)
    return index, short


def spans_for(t, index, n):
    """Maximal contiguous token spans covered by at least one matching n-gram."""
    covered = [False] * len(t)
    srcs = defaultdict(list)
    for i in range(len(t) - n + 1):
        g = tuple(t[i:i + n])
        if g in index:
            for j in range(i, i + n):
                covered[j] = True
            for s in index[g]:
                if s not in srcs[i]:
                    srcs[i].append(s)
    out, i = [], 0
    while i < len(t):
        if covered[i]:
            j = i
            while j < len(t) and covered[j]:
                j += 1
            sources = []
            for k in range(i, j):
                for s in srcs.get(k, []):
                    if s not in sources:
                        sources.append(s)
            out.append((" ".join(t[i:j]), j - i, sources[:5]))
            i = j
        else:
            i += 1
    return out


def sweep_records(records, index, n, dataset, is_control, with_gold):
    rows, flagged_ids, short = [], set(), 0
    for rec in records:
        for fname, text in item_fields(rec, with_gold=with_gold):
            t = tokens(text)
            if len(t) < n:
                short += 1
                continue
            for span, length, sources in spans_for(t, index, n):
                flagged_ids.add(rec["id"])
                rows.append(dict(
                    dataset=dataset, is_synthetic_control=is_control, item_id=rec["id"], field=fname, n=n,
                    span_tokens=length, matched_span=span, train_val_sources=";".join(sources),
                    verdict="", reason="",
                ))
    return rows, flagged_ids, short


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    man = manifest.verify(quiet=True)
    base = manifest.root()
    os.chdir(base)

    train_val = load_jsonl("data/train.jsonl") + load_jsonl("data/val.jsonl")
    evals = {k: load_jsonl(p) for k, p in EVAL_SETS.items()}
    control = load_jsonl(CONTROL)

    summary, matches, short_counts = [], [], {}
    for n in N_VALUES:
        index, short_train = build_index(train_val, n)
        short_counts[f"train_val_docs_shorter_than_{n}"] = short_train
        for name, recs in evals.items():
            rows, flagged, short = sweep_records(recs, index, n, name, False, with_gold=True)
            matches.extend(rows)
            summary.append(dict(dataset=name, is_synthetic_control=False, n=n, items=len(recs),
                                flagged_items=len(flagged), pct_flagged=round(100 * len(flagged) / len(recs), 2),
                                match_spans=len(rows), eval_docs_shorter_than_n=short))
        rows, flagged, short = sweep_records(control, index, n, "positive_control", True, with_gold=False)
        for r in rows:
            r["verdict"] = "expected (synthetic positive control)"
            r["reason"] = "summary copied verbatim from a training record by construction"
        matches.extend(rows)
        summary.append(dict(dataset="positive_control", is_synthetic_control=True, n=n, items=len(control),
                            flagged_items=len(flagged), pct_flagged=round(100 * len(flagged) / len(control), 2),
                            match_spans=len(rows), eval_docs_shorter_than_n=short))

    os.makedirs(args.out_dir, exist_ok=True)
    for fname, rows in (("contamination_summary.csv", summary), ("contamination_matches.csv", matches)):
        fields = list(rows[0].keys()) if rows else ["dataset"]
        with open(os.path.join(args.out_dir, fname), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    meta = dict(
        manifest_sha256=manifest.manifest_sha256(),
        manifest_created_utc=man["created_utc"],
        frozen_input_hashes={k: man["files"][k]["sha256"] for k in
                             ("data/train.jsonl", "data/val.jsonl", "data/eval_h.jsonl",
                              "data/fairness_matched.jsonl", "probes/probe_suite.jsonl", "data/positive_control.jsonl")},
        normalization="NFKD, strip accents, lowercase, non-alphanumeric runs -> single space, word tokens",
        n_values=list(N_VALUES),
        train_val_records=len(train_val),
        n_real_matches_to_inspect=sum(1 for m in matches if not m["is_synthetic_control"]),
        **short_counts,
        limitation=("A negative n-gram result does not rule out paraphrase, translation, structural/template "
                    "contamination, or contamination of the base model's pretraining data."),
    )
    with open(os.path.join(args.out_dir, "contamination_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(summary, indent=1))
    print(f"real (non-control) matches to inspect: {meta['n_real_matches_to_inspect']}")


if __name__ == "__main__":
    main()
