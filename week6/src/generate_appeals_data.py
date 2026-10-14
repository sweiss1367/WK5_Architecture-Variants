"""
Generates the Week 6 synthetic appeals datasets (fully offline, deterministic from --seed).

    data/train.jsonl            400  training cases   (templates NT1-4 x CS1-3, vocabulary variant A)
    data/val.jsonl               40  validation cases (same templates/vocabulary as train, unseen cases)
    data/eval_h.jsonl           150  held-out base cases (templates NT5-6 x CS4-5, vocabulary variant B),
                                      rendered as Primary language = English, Age = 45
    data/fairness_matched.jsonl 600  the same 150 cases x {English, Spanish} x {45, 75}; only the two
                                      structured header values differ within a case
    data/positive_control.jsonl   1  SYNTHETIC positive control for the n-gram detector (see below)
    data/dataset_summary.json        counts and generation parameters

Everything is fictional: invented member names, random member IDs, a closed synthetic policy
library. Gold labels depend only on the latent case facts, never on age or language.

The positive control is NOT part of any evaluation set. It is an eval-style record whose
clinical summary is copied verbatim from one training record, so the contamination sweep can
confirm the detector fires when overlap really exists.
"""
import argparse
import json
import os
import random
from collections import Counter

from appeals_common import (
    FAIRNESS_AGES, LANGUAGES, target_json, tokens, write_jsonl,
)
from case_builder import content_key, gold_for, make_latent, make_member, render_case
from templates import HELDOUT_NOTICE, HELDOUT_SUMMARY, TRAIN_NOTICE, TRAIN_SUMMARY

SEED_DEFAULT = 627


def _statuses_for(rng, case_type, allow_empty_required):
    if case_type == "submit":
        return ["documented"] * rng.choice([1, 2])
    if case_type == "request":
        if allow_empty_required and rng.random() < 0.25:
            return ["absent"]
        s = ["documented", "absent"]
        rng.shuffle(s)
        return s
    s = ["documented", "contradicted"]
    rng.shuffle(s)
    return s


def _type_list(rng, counts):
    types = ["submit"] * counts[0] + ["request"] * counts[1] + ["escalate"] * counts[2]
    rng.shuffle(types)
    return types


def _template_cycle(rng, n, notice_ids, summary_ids):
    combos = [(a, b) for a in notice_ids for b in summary_ids]
    seq = [combos[i % len(combos)] for i in range(n)]
    rng.shuffle(seq)
    return seq


def build_cases(rng, n, counts, n_no_policy, notice_ids, summary_ids, variant, allow_empty_required, seen_keys):
    types = _type_list(rng, counts)
    combos = _template_cycle(rng, n, notice_ids, summary_ids)
    no_policy_idx = set(rng.sample(range(n), n_no_policy))
    cases = []
    for i in range(n):
        for _attempt in range(200):
            statuses = _statuses_for(rng, types[i], allow_empty_required)
            case = make_latent(rng, statuses=statuses, policy_in_notice=(i not in no_policy_idx))
            key = content_key(case, combos[i][0], combos[i][1], variant)
            if key not in seen_keys:
                seen_keys.add(key)
                break
        else:
            raise RuntimeError("could not draw a non-duplicate case")
        cases.append((case, combos[i]))
    return cases


def _record(rid, split, case, member, tids, variant, with_target):
    notice, summary = render_case(case, member, tids[0], tids[1], variant)
    gold = gold_for(case, variant, with_appeal_summary=with_target)
    rec = dict(
        id=rid,
        split=split,
        notice=notice,
        summary=summary,
        extra_user_text=None,
        gold=gold,
        meta=dict(
            notice_template=tids[0], summary_template=tids[1], vocabulary_variant=variant,
            member=member, policy_id=case["policy_id"], service=case["service"],
            case_type=case["case_type"], policy_in_notice=case["policy_in_notice"],
            age_relevant_to_criteria=False,  # by construction; asserted in tests
        ),
    )
    if with_target:
        rec["target_json"] = target_json(gold)
    return rec


def generate(seed):
    rng = random.Random(seed)
    used_ids, seen_keys = set(), set()

    AGE_BANDS = [(25, 44), (45, 64), (65, 90)]

    def lang_age_lists(cases):
        """Language and age band are assigned STRATIFIED by case type (balanced within each type), so
        the training gold labels cannot correlate with either attribute through sampling noise."""
        n = len(cases)
        langs, ages = [None] * n, [None] * n
        for ctype in ("submit", "request", "escalate"):
            idx = [i for i, (c, _t) in enumerate(cases) if c["case_type"] == ctype]
            rng.shuffle(idx)
            for j, i in enumerate(idx):
                langs[i] = "English" if j % 2 == 0 else "Spanish"
            rng.shuffle(idx)
            for j, i in enumerate(idx):
                lo, hi = AGE_BANDS[j % 3]
                ages[i] = rng.randint(lo, hi)
        return langs, ages

    # --- train / val (variant A, training templates) -----------------------------------
    out = {}
    for split, n, counts, n_np in (("train", 400, (200, 120, 80), 40), ("val", 40, (20, 12, 8), 4)):
        cases = build_cases(rng, n, counts, n_np, TRAIN_NOTICE, TRAIN_SUMMARY, "A", True, seen_keys)
        langs, ages = lang_age_lists(cases)
        recs = []
        for i, (case, tids) in enumerate(cases):
            member = make_member(rng, used_ids, langs[i], ages[i])
            recs.append(_record(f"{split[:2].upper()}-{i + 1:04d}", split, case, member, tids, "A", True))
        out[split] = recs

    # --- held-out base cases (variant B, held-out templates) ----------------------------
    eval_cases = build_cases(rng, 150, (83, 45, 22), 30, HELDOUT_NOTICE, HELDOUT_SUMMARY, "B", False, seen_keys)
    eval_h, fairness = [], []
    for i, (case, tids) in enumerate(eval_cases):
        base_id = f"EH-{i + 1:04d}"
        base_member = make_member(rng, used_ids, "English", 45)
        for lang in LANGUAGES:
            for age in FAIRNESS_AGES:
                member = dict(base_member, language=lang, age=age)
                rec = _record(f"{base_id}__{lang}_{age}", "fairness_matched", case, member, tids, "B", False)
                rec["base_id"], rec["language"], rec["age"] = base_id, lang, age
                fairness.append(rec)
                if lang == "English" and age == 45:
                    ev = _record(base_id, "eval_h", case, member, tids, "B", False)
                    ev["base_id"] = base_id
                    eval_h.append(ev)
    out["eval_h"], out["fairness_matched"] = eval_h, fairness

    # --- synthetic positive control ----------------------------------------------------
    donor = max(out["train"], key=lambda r: len(tokens(r["summary"])))
    host = eval_h[0]
    out["positive_control"] = [dict(
        id="POSCTRL-001",
        split="positive_control",
        is_synthetic_positive_control=True,
        description=("SYNTHETIC POSITIVE CONTROL, not naturally occurring contamination. Notice is "
                     f"{host['id']}'s notice; the clinical summary is copied verbatim from training "
                     f"record {donor['id']}."),
        copied_summary_from=donor["id"],
        notice=host["notice"],
        summary=donor["summary"],
        extra_user_text=None,
    )]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    data = generate(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    files = {
        "train": "train.jsonl", "val": "val.jsonl", "eval_h": "eval_h.jsonl",
        "fairness_matched": "fairness_matched.jsonl", "positive_control": "positive_control.jsonl",
    }
    for key, fname in files.items():
        write_jsonl(os.path.join(args.out_dir, fname), data[key])

    summary = dict(
        seed=args.seed,
        counts={k: len(v) for k, v in data.items()},
        case_type_counts={k: dict(Counter(r["meta"]["case_type"] for r in data[k]))
                          for k in ("train", "val", "eval_h")},
        no_policy_in_notice={k: sum(1 for r in data[k] if not r["meta"]["policy_in_notice"])
                             for k in ("train", "val", "eval_h")},
        notice_templates={k: dict(Counter(r["meta"]["notice_template"] for r in data[k]))
                          for k in ("train", "val", "eval_h")},
        summary_templates={k: dict(Counter(r["meta"]["summary_template"] for r in data[k]))
                           for k in ("train", "val", "eval_h")},
        train_language_counts=dict(Counter(r["meta"]["member"]["language"] for r in data["train"])),
        fairness_design="150 base cases x {English,Spanish} x {45,75}; only header values differ within a case",
        positive_control_note="synthetic; excluded from every reported contamination percentage",
    )
    with open(os.path.join(args.out_dir, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
