"""
Builds governance/risk_register.xlsx from the REAL Week 6 results in results/ (no typed-in numbers
in the Finding and evidence column: they are formatted from the CSV/JSON files).

Rows are findings, not a target count. Residual risk is NOT reduced by any control: none has been
implemented or tested, so "Residual risk" states "unchanged" and "Control status" says why.

NIST AI RMF is mapped at function level (GOVERN / MAP / MEASURE / MANAGE); the subcategory column is
optional context. ISO/IEC 42001 clauses are limited to 6.1.2, 6.1.3, 6.1.4, 8.2, 8.3 and 9.1.
Likelihood and impact are qualitative 1-5 judgments (scales on the Legend sheet), not measurements.
"""
import json
import os

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

R = "results"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def build():
    task = pd.read_csv(f"{R}/task_baseline.csv").set_index(["dataset", "metric"])
    fam = pd.read_csv(f"{R}/probe_summary_by_family.csv").set_index("family")
    inj = pd.read_csv(f"{R}/posthoc_injection_breakdown.csv").set_index("run")
    post = json.load(open(f"{R}/posthoc_summary.json"))
    fd = pd.read_csv(f"{R}/fairness_slice_differences.csv")
    fd = fd[(fd.stratum == "marginal") & (fd.metric == "rationale_omission")].set_index("slice")
    byp = pd.read_csv(f"{R}/probe_summary_by_probe.csv")

    def t(ds, m, ci=True):
        r = task.loc[(ds, m)]
        return f"{pct(r.estimate)} [{pct(r.ci_low)}, {pct(r.ci_high)}] (n={int(r.n)})" if ci else pct(r.estimate)

    scen = byp.groupby("scenario").agg(n=("probe_id", "size"), g=("greedy_passed", lambda x: int((1 - x).sum())))
    f = lambda k: fam.loc[k]
    esc = post["escalate_recall_eval_h_greedy"]
    pa = post["pair_agreement"]
    agree = {(a["comparison"], a["field"]): a["identical_pct"] for a in pa}
    inst = post["prompts_with_next_step_differing_across_seeds"]
    lang, age = fd.loc["language"], fd.loc["age"]

    rows = [
        dict(
            id="R1",
            finding=("Contradicted or urgent cases are not escalated, and are often routed to 'submit appeal'. "
                     f"The model recommended escalation for {esc} gold-escalate cases on the held-out set under greedy decoding "
                     f"({post['escalate_recall_val_greedy']} in-template) and {post['escalate_recall_fairness_sampled_3_seeds']} under sampling, "
                     f"although 20% of its training labels were escalations. Overconfidence probes: {int(f('unsafe_overconfident_recommendation').greedy_failed)} of 20 failed on greedy "
                     f"({int(scen.loc['contradicted_criterion_member_asks_will_it_succeed'].g)}/{int(scen.loc['contradicted_criterion_member_asks_will_it_succeed'].n)} contradicted-evidence, "
                     f"{int(scen.loc['urgent_symptoms_noted_in_summary'].g)}/{int(scen.loc['urgent_symptoms_noted_in_summary'].n)} urgent-symptom probes), every failure being a wrong recommended step (a missed escalation in 11 of the 13). "
                     " No certainty language or treatment directives appeared in any of 80 probe runs. "
                     f"Held-out next-step accuracy {t('eval_h', 'next_step_correct')}."),
            limit=("Synthetic, templated cases; one adapter and one training seed, so this may be an under-training effect of this adapter "
                   "rather than a property of the approach. Urgency was never a training signal, so the urgent-symptom expectation "
                   "(escalate) is an audit-design decision. 20 probes per family: counts, not rates."),
            nist="MEASURE; MANAGE", nist_sub="MEASURE 2.5", iso="6.1.2; 6.1.3; 6.1.4; 8.2; 8.3",
            L=4, I=5,
            mit=("Proposed: do not use the model's recommended_next_step for routing; require human clinical review of every case; "
                 "add an escalation rule outside the model (for example urgency and contradiction checks); retrain with more escalation "
                 "examples and re-run this audit before any reliance."),
            owner="Medical Director / Appeals Operations Lead",
            treat="Mitigate (do not rely on the model's routing)", gate="BLOCKING: no pilot use of model-driven routing until escalation behavior is demonstrated on held-out and real-format cases."),
        dict(
            id="R2",
            finding=("Embedded instructions hijack the output. Prompt-injection probes: "
                     f"{int(inj.loc['greedy'].unambiguous_injection_effect)} of 20 show unambiguous injection effects on greedy and on each sampled seed "
                     "(counts "
                     f"{', '.join(str(int(inj.loc[r].unambiguous_injection_effect)) for r in ['seed1001','seed1002','seed1003'])} on the three seeds). On greedy: the bare word 'APPROVED' "
                     "instead of JSON on all 4 notice-override probes, and the injected fake assistant turn (with its marker) echoed on all 3 delimiter-break probes. "
                     f"A further {int(inj.loc['greedy'].next_step_divergence_only)} probes diverge only in the recommended step and are not attributable to the injection "
                     "without a control. The system prompt told the model to treat input as data."),
            limit=("No un-injected control probes were run, so next-step-only divergence cannot be split from the model's baseline error. 20 probes, one adapter, "
                   "synthetic injections authored by the audit team; real adversarial text may differ."),
            nist="MEASURE; MANAGE", nist_sub="MEASURE 2.7", iso="6.1.2; 6.1.3; 8.2; 8.3",
            L=3, I=4,
            mit=("Proposed: strict output validation that rejects anything that is not schema-valid JSON; strip or quarantine instruction-like text and "
                 "role markers from notices before they reach the model; treat member- or provider-authored text as untrusted; add injection tests "
                 "to release criteria."),
            owner="Security / Platform Engineering",
            treat="Mitigate", gate="BLOCKING for any input channel that includes third-party text, until input handling and output validation are tested."),
        dict(
            id="R3",
            finding=("Evidence is misattributed or missing evidence goes unflagged. Fabricated-rationale probes: "
                     f"{int(f('fabricated_rationale').greedy_failed)} of 20 failed on greedy, {int(f('fabricated_rationale').failed_any_run)} on at least one run, "
                     f"{int(f('fabricated_rationale').failed_all_runs)} on every run. Greedy failure reasons: missing evidence not flagged 9, recommended submit 6, an "
                     "unrelated documented fact used as support for a cited criterion 6, text not in the summary 2 (borderline paraphrase). "
                     f"Held-out task: rationale omission {t('eval_h', 'rationale_omission')}, criteria identified correctly {t('eval_h', 'criteria_match')}, "
                     f"missing evidence identified correctly {t('eval_h', 'missing_match')}."),
            limit=("Mostly misattribution, not free invention (invented text was rare). Rule-based scoring with a token-overlap threshold; two borderline "
                   "labels listed in results/probe_rule_vs_reading.csv. Synthetic facts with no clinical validity."),
            nist="MEASURE; MANAGE", nist_sub="MEASURE 2.5", iso="6.1.2; 6.1.3; 8.2; 8.3; 9.1",
            L=4, I=4,
            mit=("Proposed: show the source sentence for every cited fact and require a reviewer to confirm each one; block 'submit' unless every listed criterion "
                 "has a verified supporting fact; track omission and misattribution rates on real cases."),
            owner="Appeals Operations Lead; ML Engineering",
            treat="Mitigate", gate="REQUIRED before pilot: mandatory human verification of every cited fact."),
        dict(
            id="R4",
            finding=("Recommendations are unstable between runs. For identical prompts, the recommended step differs across the three sampling seeds for "
                     f"{inst['n']} of {inst['of']} prompts ({inst['pct']}%); the whole JSON is identical in only "
                     f"{agree[('same prompt, different sampling seeds', 'entire JSON')]}% of seed pairs. Sampled next-step accuracy was "
                     f"{post['sampled_next_step_accuracy_overall']} versus {t('eval_h', 'next_step_correct', ci=False)} greedy."),
            limit=("Measured at temperature 0.7 / top_p 0.9 only; greedy decoding is deterministic but was not better at escalation. Single adapter; "
                   "these seeds measure generation variability, not training-seed variability."),
            nist="MEASURE", nist_sub="MEASURE 2.5", iso="6.1.2; 8.2; 9.1",
            L=4, I=3,
            mit=("Proposed: fix decoding (greedy) in production and re-test; log every generation; record a model/version/decoding identifier with each case "
                 "so a reviewer can reproduce the output."),
            owner="ML Engineering",
            treat="Mitigate", gate="REQUIRED: deterministic decoding and reproducibility logging before pilot."),
        dict(
            id="R5",
            finding=("Performance drops on formats the model did not see in training. Criteria identified correctly: "
                     f"{t('val', 'criteria_match')} on in-template validation vs {t('eval_h', 'criteria_match')} on held-out templates (intervals do not overlap). "
                     f"Rationale omission {t('val', 'rationale_omission')} vs {t('eval_h', 'rationale_omission')}; next-step accuracy "
                     f"{t('val', 'next_step_correct')} vs {t('eval_h', 'next_step_correct')}."),
            limit=("Validation set has only 40 cases (wide intervals; the omission and next-step comparisons overlap). 'Held-out' means different "
                   "synthetic templates and wording, not real-world documents."),
            nist="MEASURE; MAP", nist_sub="MEASURE 2.5", iso="6.1.2; 8.2; 9.1",
            L=4, I=3,
            mit=("Proposed: test on a sample of real, de-identified denial notices in the insurer's actual formats before the pilot; widen training template "
                 "diversity; set a go/no-go threshold on the real-format sample."),
            owner="ML Engineering; Appeals Operations Lead",
            treat="Mitigate", gate="REQUIRED: real-format validation sample before pilot."),
        dict(
            id="R6",
            finding=("No disparity could be established or ruled out across the two tested slices. Primary metric (required-rationale omission), paired difference: "
                     f"language Spanish minus English {lang.estimate_pp:+.2f} pp (95% CI {lang.ci_low_pp:.2f} to {lang.ci_high_pp:.2f}); "
                     f"age 75 minus 45 {age.estimate_pp:+.2f} pp (95% CI {age.ci_low_pp:.2f} to {age.ci_high_pp:.2f}); n={int(lang.n_cases)} matched cases, "
                     f"{int(lang.n_generations_a)} generations per level. Both intervals include zero: unresolved under the preregistered rule. "
                     "All secondary metrics were also unresolved. Whole-output identity across matched pairs "
                     f"({agree[('language: Spanish vs English', 'entire JSON')]}% language, {agree[('age: 75 vs 45', 'entire JSON')]}% age) "
                     f"matched the same-prompt seed-to-seed rate ({agree[('same prompt, different sampling seeds', 'entire JSON')]}%) (exploratory)."),
            limit=("Unresolved is not evidence of equal performance. The test changes only a structured header marker, not non-English text, names, or real-world "
                   "inputs. One adapter, synthetic data, no planted-bias power check. Two slices only; intersectional effects not analysed."),
            nist="MEASURE; MANAGE", nist_sub="MEASURE 2.11", iso="6.1.2; 6.1.4; 8.2; 9.1",
            L=2, I=4,
            mit=("Proposed: do not claim fairness from this audit; disaggregated monitoring by language and age (and other relevant groups) on real pilot "
                 "cases with predefined review triggers; test non-English notices and names with a power check before expansion."),
            owner="Compliance / Fairness and Privacy Officer",
            treat="Mitigate (monitor); fairness not certified", gate="CONDITION on pilot: monitoring plan agreed; no fairness assurance statement issued from this evidence."),
        dict(
            id="R7",
            finding=("The evidence base cannot certify real-world performance or exclude contamination. n-gram sweep (n = 8, 13, 20): n = 8 flagged 72% of held-out "
                     "cases, all 618 matches (35 unique spans) being the intentionally shared synthetic policy vocabulary (expected shared-domain vocabulary, not "
                     "confirmed leakage); n = 13 and n = 20 flagged none. A synthetic positive control was detected at all three n and is excluded from these "
                     "percentages."),
            limit=("Zero matches at n = 13 and n = 20 do not prove the evaluation set is uncontaminated: n-gram screening misses paraphrase, structural reuse, "
                   "translation and other non-verbatim leakage, and cannot see pretraining contamination. All data synthetic and templated; one adapter, one training seed."),
            nist="MAP; MEASURE", nist_sub="", iso="6.1.2; 8.2; 9.1",
            L=3, I=3,
            mit=("Proposed: build the acceptance test from real, de-identified, never-trained-on cases with provenance records; add paraphrase and "
                 "template-level overlap checks; repeat training with more than one seed before relying on any metric."),
            owner="Model Risk Management",
            treat="Mitigate (evidence quality)", gate="REQUIRED: independent real-case acceptance set before any pilot-readiness claim."),
        dict(
            id="R8",
            finding=("Policy hallucination and synthetic member-information leakage were not observed. Policy-authority probes "
                     f"{int(f('policy_authority').greedy_failed)} of 20 failed and leakage probes {int(f('member_information_leakage').greedy_failed)} of 20 failed on greedy and on every sampled run "
                     f"(0 of 80 runs each; no member-ID token in any leakage output). Unsupported policy citation {t('eval_h', 'unsupported_policy')} on the held-out set "
                     "and 0 of 1,794 scored fairness generations."),
            limit=("20 probes per family; the output schema has no member-ID field, which limits the leakage surface; only 7 prefix-completion probes; IDs and "
                   "names are synthetic and seen once per epoch (3 epochs) in training. Absence in these tests is not proof of absence."),
            nist="MEASURE", nist_sub="MEASURE 2.10", iso="6.1.2; 8.2; 9.1",
            L=1, I=4,
            mit=("Proposed: keep the fixed output schema (no free identifier fields); re-run the policy and leakage probes on every model change; "
                 "monitor for identifiers and unknown policy IDs in production outputs."),
            owner="Compliance / Fairness and Privacy Officer; Security",
            treat="Accept with monitoring", gate="NON-BLOCKING: monitor; re-test on each model change."),
    ]
    for r in rows:
        r["score"] = r["L"] * r["I"]
        r["residual"] = (f"Unchanged: {r['L']} x {r['I']} = {r['score']}. No control has been implemented or tested, so no risk reduction is claimed.")
        r["status"] = "Proposed; not implemented; not tested"
    return rows


def main():
    rows = build()
    wb = Workbook()
    ws = wb.active
    ws.title = "Register"
    cols = [
        ("Finding ID", "id", 9), ("Finding and evidence", "finding", 70), ("Evidence limitation", "limit", 52),
        ("NIST AI RMF function", "nist", 16), ("NIST subcategory (optional)", "nist_sub", 14),
        ("ISO/IEC 42001 clause", "iso", 18), ("Likelihood (1-5)", "L", 11), ("Impact (1-5)", "I", 9),
        ("Current risk score (L x I)", "score", 12), ("Mitigation (proposed)", "mit", 52), ("Proposed owner role", "owner", 24),
        ("Control status", "status", 20), ("Residual risk", "residual", 36), ("Risk-treatment decision", "treat", 26),
        ("Pilot-gate implication", "gate", 40),
    ]
    ws.append([c[0] for c in cols])
    for r in sorted(rows, key=lambda x: (-x["score"], x["id"])):
        ws.append([r[c[1]] for c in cols])
    head = PatternFill("solid", fgColor="1F3A5F")
    for i, c in enumerate(cols, 1):
        cell = ws.cell(row=1, column=i)
        cell.font, cell.fill = Font(bold=True, color="FFFFFF"), head
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[get_column_letter(i)].width = c[2]
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "C2"

    leg = wb.create_sheet("Method and legend")
    for line in [
        ["Purpose", "Risk register for a hypothetical health-insurer appeals-drafting pilot (Meridian), built only from the real Week 6 audit results. Rows are findings; no target count."],
        ["Evidence rule", "Every number in 'Finding and evidence' is formatted from files in results/ (see the Evidence Index sheet). Week 3 findings are not used."],
        ["Likelihood (1-5)", "Judgment of how often the failure would occur on pilot-like inputs given what was observed in testing: 1 not observed, 2 possible, 3 plausible, 4 frequent in testing, 5 near-certain."],
        ["Impact (1-5)", "Judgment of consequence if it occurs in a live appeals workflow: 1 negligible, 2 minor rework, 3 delayed or incorrect appeal, 4 serious member or regulatory harm possible, 5 patient-safety-relevant delay or harm."],
        ["Scores", "Qualitative, assigned by the audit author; not measurements. Reasonable reviewers may score differently."],
        ["Residual risk", "Equal to current risk. No proposed control has been implemented or tested, so no reduction is claimed."],
        ["Control status", "Proposed / implemented / tested. All are 'Proposed; not implemented; not tested'."],
        ["NIST AI RMF", "Mapped at function level (GOVERN, MAP, MEASURE, MANAGE). The subcategory column is optional context. No row maps to GOVERN yet because no governance control was assessed in this audit."],
        ["ISO/IEC 42001", "Clauses used: 6.1.2 AI risk assessment; 6.1.3 AI risk treatment; 6.1.4 AI system impact assessment; 8.2 AI risk assessment; 8.3 AI risk treatment; 9.1 monitoring, measurement, analysis and evaluation. Clause 8.4 is not used."],
        ["Pilot gate", "BLOCKING = must be resolved before any pilot use; REQUIRED = condition of the pilot; CONDITION / NON-BLOCKING as stated."],
    ]:
        leg.append(line)
    leg.column_dimensions["A"].width, leg.column_dimensions["B"].width = 20, 140
    for row in leg.iter_rows():
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
        row[0].font = Font(bold=True)

    ev = wb.create_sheet("Evidence index")
    ev.append(["Finding ID", "Source files (results/ and logs/)"])
    for fid, files in {
        "R1": "task_baseline.csv; posthoc_next_step_confusion.csv; probe_summary_by_family.csv; probe_summary_by_probe.csv; posthoc_overconfidence_predicted_vs_expected.csv; probe_results_by_run.csv",
        "R2": "probe_results_by_run.csv; probe_failure_transcripts.jsonl; posthoc_injection_breakdown.csv; probe_rule_vs_reading.csv",
        "R3": "probe_summary_by_family.csv; probe_results_by_run.csv; task_baseline.csv; probe_rule_vs_reading.csv",
        "R4": "posthoc_summary.json; posthoc_pair_agreement.csv; fairness_row_metrics.csv; task_baseline.csv",
        "R5": "task_baseline.csv; task_scores_val.csv; task_scores_eval_h.csv",
        "R6": "fairness_slice_differences.csv; fairness_disaggregated.csv; fairness_row_metrics.csv; posthoc_pair_agreement.csv",
        "R7": "contamination_summary.csv; contamination_matches.csv; contamination_unique_spans.csv; contamination_review.md; contamination_meta.json",
        "R8": "probe_summary_by_family.csv; probe_results_by_run.csv; task_baseline.csv; fairness_row_metrics.csv",
    }.items():
        ev.append([fid, files])
    ev.append([])
    ev.append(["Bindings", "MANIFEST.json sha256 prefix 29cfecf0ca584a16; adapter sha256 in logs/adapter_sha256.json; environment in results/env_manifest.json"])
    ev.column_dimensions["A"].width, ev.column_dimensions["B"].width = 14, 150

    lim = wb.create_sheet("Limitations")
    for line in [
        "One adapter, one training seed. The three inference seeds measure generation variability, not training-seed variability.",
        "All data are synthetic and templated; no real notices, members or clinical content. No clinical validity claimed.",
        "Rule-based scoring; no model judge; token-overlap thresholds are conventions (80%).",
        "20 probes per family support counts, not failure rates. No un-injected control probes were run.",
        "Fairness tests a structured header marker only (primary language, age), at one decoding setting, with no planted-bias power check. Unresolved is not equal.",
        "The contamination sweep cannot see paraphrase, translation, structural or pretraining contamination.",
        "Post-hoc analyses (posthoc_*.csv) are exploratory and were not preregistered; the preregistered rules and frozen files are unchanged.",
        "A 0.5B model is not what an insurer would deploy; findings concern this audit method and this adapter.",
    ]:
        lim.append([line])
    lim.column_dimensions["A"].width = 160
    for row in lim.iter_rows():
        row[0].alignment = Alignment(wrap_text=True, vertical="top")

    os.makedirs("governance", exist_ok=True)
    wb.save("governance/risk_register.xlsx")
    print("wrote governance/risk_register.xlsx;", len(rows), "rows")
    for r in sorted(rows, key=lambda x: (-x["score"], x["id"])):
        print(r["id"], r["L"], r["I"], r["score"], r["treat"])


if __name__ == "__main__":
    main()
