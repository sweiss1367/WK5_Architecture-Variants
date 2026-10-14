"""
Structural audit of the generated Week 6 datasets and probe suite. CPU only.

Hard checks (the script exits non-zero if any fails):
    * sizes are 400 / 40 / 150 / 600 / 100
    * train+val use only templates NT1-4/CS1-3 and vocabulary A; eval/fairness/probes use only
      NT5-6/CS4-5 and vocabulary B; no template signature or vocabulary-B string appears in
      train/val text and no vocabulary-A string appears in held-out text
    * no duplicate (notice, summary) pair anywhere; member IDs unique across train/val/eval
    * gold labels are internally consistent with the rendered text
    * fairness pairs: within each base case the 4 variants have byte-identical clinical summaries,
      identical gold, and notices that are identical once the structured header is masked
    * no age or language wording in any criterion/fact string, summary, or masked notice body
    * probe suite: 100 unique IDs, 20 per family, every probe has hypothesis/expected/rule

Descriptive (reported, not pass/fail): case-type and next-step rates by language and age band
in the training data, as evidence that gold labels were not generated from those attributes.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

from appeals_common import CRITERIA, EXTRA_SERVICES, POLICIES, load_jsonl, normalize_text
from templates import _HEADERS

SIGNATURES = {
    "NT1": "DENIAL NOTICE", "NT2": "Coverage decision letter", "NT3": "RE: denial of",
    "NT4": "MEDICAL REVIEW RESULT", "NT5": "Notice of adverse determination", "NT6": "Determination regarding",
    "CS1": "Clinical summary. Diagnosis:", "CS2": "According to the chart", "CS3": "Summary of record",
    "CS4": "Referral letter:", "CS5": "Provider note (condensed)",
}
TRAIN_T = {"NT1", "NT2", "NT3", "NT4", "CS1", "CS2", "CS3"}
HELD_T = {"NT5", "NT6", "CS4", "CS5"}
AGE_RE = re.compile(r"\b(age|aged|ages|year|years|old|older|elderly|young|younger|senior|geriatric|pediatric|adult|adults|minor|infant|child|children|teen\w*)\b", re.I)
LANG_RE = re.compile(r"\b(language|languages|interpreter|spanish|english|bilingual|translat\w*|native speaker)\b", re.I)


def mask_header(rec):
    m = rec["meta"]["member"]
    hdr = _HEADERS[rec["meta"]["notice_template"]].format(name=m["name"], mid=m["member_id"], age=m["age"], lang=m["language"])
    assert hdr in rec["notice"], f"header not found verbatim in {rec['id']}"
    return rec["notice"].replace(hdr, "<HEADER>")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--probes", default="probes/probe_suite.jsonl")
    ap.add_argument("--out", default="data/dataset_audit.json")
    args = ap.parse_args()

    d = {k: load_jsonl(os.path.join(args.data_dir, f"{k}.jsonl")) for k in ("train", "val", "eval_h", "fairness_matched")}
    probes = load_jsonl(args.probes)
    fails, report = [], {}

    def check(name, ok, detail=None):
        report[name] = dict(passed=bool(ok), **({"detail": detail} if detail is not None else {}))
        if not ok:
            fails.append(name)

    check("sizes", [len(d[k]) for k in ("train", "val", "eval_h", "fairness_matched")] == [400, 40, 150, 600] and len(probes) == 100,
          {k: len(v) for k, v in d.items()} | {"probes": len(probes)})

    # -- template + vocabulary separation ---------------------------------------------
    used_t = {k: Counter() for k in d}
    for k, rows in d.items():
        for r in rows:
            used_t[k][r["meta"]["notice_template"]] += 1
            used_t[k][r["meta"]["summary_template"]] += 1
    check("train_val_use_only_train_templates",
          set(used_t["train"]) | set(used_t["val"]) <= TRAIN_T and set(used_t["train"]) | set(used_t["val"]) == TRAIN_T)
    check("heldout_sets_use_only_heldout_templates",
          set(used_t["eval_h"]) | set(used_t["fairness_matched"]) == HELD_T)
    check("variants", all(r["meta"]["vocabulary_variant"] == "A" for r in d["train"] + d["val"])
          and all(r["meta"]["vocabulary_variant"] == "B" for r in d["eval_h"] + d["fairness_matched"]))

    def text_of(r):
        return r["notice"] + "\n" + r["summary"] + "\n" + (r.get("extra_user_text") or "")

    tv_text = "\n".join(text_of(r) for r in d["train"] + d["val"])
    held_text = "\n".join(text_of(r) for r in d["eval_h"] + d["fairness_matched"] + probes)
    sig_leaks = [t for t in HELD_T if SIGNATURES[t] in tv_text] + [t for t in TRAIN_T if SIGNATURES[t] in held_text]
    check("no_template_signature_crosses_split", not sig_leaks, sig_leaks)
    tv_n, held_n = normalize_text(tv_text), normalize_text(held_text)
    vocab_leaks = []
    for cid, c in CRITERIA.items():
        for kind in ("label", "pres", "cont"):
            if normalize_text(c[f"{kind}_B"]) in tv_n:
                vocab_leaks.append(f"{cid}.{kind}_B in train/val")
            if normalize_text(c[f"{kind}_A"]) in held_n:
                vocab_leaks.append(f"{cid}.{kind}_A in held-out")
    check("no_criterion_phrase_crosses_split", not vocab_leaks, vocab_leaks)

    # -- uniqueness -------------------------------------------------------------------
    all_pairs = [(r["notice"], r["summary"]) for k in ("train", "val", "eval_h", "fairness_matched") for r in d[k]]
    # fairness_matched contains the eval_h notice/summary again by design, so dedupe on ids
    keyed = Counter((r["notice"], r["summary"]) for k in ("train", "val") for r in d[k])
    keyed.update((r["notice"], r["summary"]) for r in d["fairness_matched"])
    check("no_duplicate_notice_summary_pairs", max(keyed.values()) == 1, [k for k, v in keyed.items() if v > 1][:3])
    ids = [r["meta"]["member"]["member_id"] for k in ("train", "val", "eval_h") for r in d[k]]
    check("member_ids_unique_across_splits", len(ids) == len(set(ids)))
    check("eval_h_equals_english_45_variant", all(
        e["notice"] == f["notice"] and e["summary"] == f["summary"]
        for e in d["eval_h"] for f in d["fairness_matched"]
        if f["base_id"] == e["id"] and f["language"] == "English" and f["age"] == 45))

    # -- gold consistency -------------------------------------------------------------
    problems = []
    for k in ("train", "val", "eval_h", "fairness_matched"):
        for r in d[k]:
            g = r["gold"]
            sm_n, nt_n = normalize_text(r["summary"]), normalize_text(r["notice"])
            for f in g["supporting_clinical_facts"]:
                if normalize_text(f) not in sm_n:
                    problems.append((r["id"], "fact_not_in_summary", f))
            for lab in g["criteria_not_met"]:
                if normalize_text(lab) not in nt_n:
                    problems.append((r["id"], "label_not_in_notice", lab))
            if g["policy_cited"] is not None and normalize_text(g["policy_cited"]) not in nt_n:
                problems.append((r["id"], "policy_not_in_notice", g["policy_cited"]))
            if g["policy_cited"] is None and re.search(r"MRP-\d+", r["notice"]):
                problems.append((r["id"], "policy_in_notice_but_gold_null", ""))
            st = g["_ids"]["statuses"]
            exp = "escalate_to_clinical_reviewer" if "contradicted" in st.values() else (
                "request_more_information" if "absent" in st.values() else "submit_appeal")
            if g["recommended_next_step"] != exp:
                problems.append((r["id"], "next_step_inconsistent", exp))
            if k in ("train", "val") and json.loads(r["target_json"])["recommended_next_step"] != exp:
                problems.append((r["id"], "target_next_step_inconsistent", exp))
            if k in ("eval_h", "fairness_matched") and not g["_ids"]["required"]:
                problems.append((r["id"], "no_required_fact_in_eval_case", ""))
    check("gold_consistent_with_text", not problems, problems[:5])

    # -- fairness pair invariance -----------------------------------------------------
    by_base = defaultdict(list)
    for r in d["fairness_matched"]:
        by_base[r["base_id"]].append(r)
    bad = []
    for b, rows in by_base.items():
        if sorted((x["language"], x["age"]) for x in rows) != [("English", 45), ("English", 75), ("Spanish", 45), ("Spanish", 75)]:
            bad.append((b, "variant_set"))
            continue
        if len({x["summary"] for x in rows}) != 1:
            bad.append((b, "summary_differs"))
        if len({mask_header(x) for x in rows}) != 1:
            bad.append((b, "notice_body_differs"))
        if len({json.dumps(x["gold"], sort_keys=True) for x in rows}) != 1:
            bad.append((b, "gold_differs"))
        if len({(x["meta"]["member"]["name"], x["meta"]["member"]["member_id"]) for x in rows}) != 1:
            bad.append((b, "member_identity_differs"))
    check("fairness_pairs_identical_except_two_header_values", len(by_base) == 150 and not bad, bad[:5])

    # -- age / language wording -------------------------------------------------------
    hits = []
    strings = [s for c in CRITERIA.values() for s in c.values()]
    strings += [p["service"] for p in POLICIES] + [p["dx"] for p in POLICIES] + [p["title"] for p in POLICIES]
    strings += [e["service"] for e in EXTRA_SERVICES] + [e["dx"] for e in EXTRA_SERVICES]
    for s in strings:
        if AGE_RE.search(s) or LANG_RE.search(s):
            hits.append(("vocab", s))
    for r in d["train"] + d["val"] + d["eval_h"]:
        body = mask_header(r) + "\n" + r["summary"]
        body = body.replace("<HEADER>", "")
        if AGE_RE.search(body) or LANG_RE.search(body):
            hits.append((r["id"], AGE_RE.search(body) or LANG_RE.search(body)))
    check("no_age_or_language_wording_outside_header", not hits, [str(h) for h in hits[:5]])

    # -- probe suite ------------------------------------------------------------------
    pid = [p["id"] for p in probes]
    fam = Counter(p["family"] for p in probes)
    fields_ok = all(p["failure_hypothesis"] and p["expected_behavior"] and p["scoring_rule"] and p["scoring_rule_id"] for p in probes)
    check("probe_suite_structure", len(pid) == len(set(pid)) == 100 and set(fam.values()) == {20} and len(fam) == 5 and fields_ok, dict(fam))

    # -- descriptive: gold vs attributes in training ----------------------------------
    desc = defaultdict(Counter)
    for r in d["train"]:
        m = r["meta"]["member"]
        band = "age<45" if m["age"] < 45 else ("age45-64" if m["age"] < 65 else "age65+")
        desc[f"language={m['language']}"][r["gold"]["recommended_next_step"]] += 1
        desc[band][r["gold"]["recommended_next_step"]] += 1
    report["descriptive_next_step_by_attribute_in_train"] = {k: dict(v) for k, v in sorted(desc.items())}
    report["descriptive_train_language_counts"] = dict(Counter(r["meta"]["member"]["language"] for r in d["train"]))

    out = dict(all_hard_checks_passed=not fails, failed=fails, checks=report)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps({k: (v["passed"] if isinstance(v, dict) and "passed" in v else v) for k, v in report.items()}, indent=1, default=str))
    print("ALL HARD CHECKS PASSED" if not fails else f"FAILED: {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
