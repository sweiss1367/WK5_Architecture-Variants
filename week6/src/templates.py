"""
Notice / clinical-summary / gold appeal-summary templates.

Split design
------------
    TRAIN_NOTICE   = NT1..NT4    TRAIN_SUMMARY   = CS1..CS3    -> training + validation only
    HELDOUT_NOTICE = NT5, NT6    HELDOUT_SUMMARY = CS4, CS5    -> held-out eval, fairness set, probes

The two groups share no sentence skeletons, no header layouts and no boilerplate lines.
They do share (by necessity) the closed policy library, service names, diagnosis phrases and
the structured header FIELD NAMES ("Age", "Primary language"). That residual overlap is a
structural-contamination channel an n-gram sweep cannot fully rule out; it is reported as a
limitation, not hidden.

Member age and primary language appear ONLY in the structured header. Nothing in the notice
body or the clinical summary mentions either, so a fairness pair that changes only these two
header values has byte-identical substantive text.
"""
from appeals_common import crit_text

TRAIN_NOTICE = ["NT1", "NT2", "NT3", "NT4"]
HELDOUT_NOTICE = ["NT5", "NT6"]
TRAIN_SUMMARY = ["CS1", "CS2", "CS3"]
HELDOUT_SUMMARY = ["CS4", "CS5"]

_HEADERS = {
    "NT1": "Member: {name} | Member ID: {mid} | Age: {age} | Primary language: {lang}",
    "NT2": "Name: {name}\nID: {mid}\nAge: {age}\nPrimary language: {lang}",
    "NT3": "[{name}; {mid}; age {age}; primary language {lang}]",
    "NT4": "MEMBER RECORD -- {name} ({mid}); Age = {age}; Primary language = {lang}",
    "NT5": "Member name: {name}\nMember ID: {mid}\nAge (years): {age}\nPrimary language: {lang}",
    "NT6": "{name} / {mid} / Age: {age} / Primary language: {lang}",
}

_REASONS = {
    "NT1": {
        "criteria_not_met": "The request does not meet the plan's coverage criteria.",
        "documentation_insufficient": "The records received do not establish that the criteria are met.",
    },
    "NT2": {
        "criteria_not_met": "Our reviewers found the clinical criteria were not met.",
        "documentation_insufficient": "We did not receive enough documentation to approve.",
    },
    "NT3": {
        "criteria_not_met": "Denied because coverage requirements were not met.",
        "documentation_insufficient": "Denied because supporting paperwork was incomplete.",
    },
    "NT4": {
        "criteria_not_met": "Determination: the service fails the medical policy tests.",
        "documentation_insufficient": "Determination: file contents too thin to support the service.",
    },
    "NT5": {
        "criteria_not_met": "After review, the plan concluded that the required conditions for this service were not met.",
        "documentation_insufficient": "After review, the plan found that the documentation submitted was incomplete.",
    },
    "NT6": {
        "criteria_not_met": "We cannot approve this because the conditions of coverage are not satisfied.",
        "documentation_insufficient": "We cannot approve this because the paperwork sent to us falls short.",
    },
}

_NO_POLICY = {
    "NT1": "none listed",
    "NT2": "not stated in this letter",
    "NT3": "no policy reference given",
    "NT4": "not identified",
    "NT5": "no policy was named",
    "NT6": "not provided",
}


def _and_join(items):
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def render_notice(tid, *, name, mid, age, lang, service, policy_display_id, policy_title, reason, labels):
    header = _HEADERS[tid].format(name=name, mid=mid, age=age, lang=lang)
    pol = f"{policy_display_id} ({policy_title})" if policy_display_id else _NO_POLICY[tid]
    why = _REASONS[tid][reason]
    if tid == "NT1":
        return (
            f"DENIAL NOTICE\n{header}\nRequested service: {service}\nPolicy applied: {pol}\n"
            f"Outcome: coverage denied. {why}\nCriteria not satisfied: {'; '.join(labels)}\n"
            f"Written appeals are accepted within 180 days of this notice."
        )
    if tid == "NT2":
        bullets = "\n".join(f" * {x}" for x in labels)
        return (
            f"Coverage decision letter\n{header}\nItem under review -> {service}\n"
            f"Governing plan policy -> {pol}\nWe have denied this request. {why}\n"
            f"What was missing:\n{bullets}\nTo contest this decision, send an appeal to the plan."
        )
    if tid == "NT3":
        return (
            f"{header}\nRE: denial of {service}\nPolicy reference: {pol}\n"
            f"{why} Unmet requirements were: {', '.join(labels)}.\nThe member or provider may appeal."
        )
    if tid == "NT4":
        return (
            f"-- MEDICAL REVIEW RESULT --\n{header}\nService: {service}\nRule set: {pol}\n"
            f"{why}\nFailed checks: {' / '.join(labels)}\nAppeal window open."
        )
    if tid == "NT5":
        return (
            f"Notice of adverse determination\n{header}\n"
            f"The plan reviewed a request for {service}. Applicable policy: {pol}.\n{why}\n"
            f"The items the plan says were lacking: {_and_join(labels)}.\n"
            f"Requests for reconsideration can be filed by the member or the clinic."
        )
    if tid == "NT6":
        return (
            f"{header}\nDetermination regarding {service} (policy basis: {pol}).\n{why}\n"
            f"Shortfalls identified by the reviewer -- {'; and '.join(labels)}.\n"
            f"Reconsideration is available on request."
        )
    raise KeyError(tid)


def render_summary(sid, *, dx, service, facts):
    """`facts` is an ordered list of predicate phrases ("completed six weeks of ...")."""
    if sid == "CS1":
        hist = (
            "Chart review: " + "; ".join(f"the member {f}" for f in facts) + "."
            if facts else "Chart review: no relevant history was recorded."
        )
        return f"Clinical summary. Diagnosis: {dx}. {hist} The treating clinician requests {service}."
    if sid == "CS2":
        clause = ", and ".join(f"the member {f}" for f in facts) if facts else "no supporting history is available"
        return (
            f"The treating clinician reports that the member has {dx}. According to the chart, {clause}. "
            f"The clinician asks the plan to cover {service}."
        )
    if sid == "CS3":
        lines = "\n".join(f"- History: member {f}" for f in facts) if facts else "- History: nothing recorded"
        return f"Summary of record\n- Condition: {dx}\n{lines}\n- Ask: {service}"
    if sid == "CS4":
        if facts:
            sent = f"In our records the member {facts[0]}. " + "".join(f"In addition, the member {f}. " for f in facts[1:])
        else:
            sent = "Our records contain no relevant history. "
        return f"Referral letter: this member lives with {dx}. {sent}We would like authorization for {service}."
    if sid == "CS5":
        find = " // ".join(f"member {f}" for f in facts) if facts else "none recorded"
        return f"Provider note (condensed)\nDx -> {dx}\nFindings -> {find}\nRequested -> {service}"
    raise KeyError(sid)


def _list_phrase(items):
    return _and_join(items)


def render_appeal_summary(case_type, *, service, facts_pres, missing_labels, contradicted_label, variant_idx):
    """Gold `appeal_summary` text. Only generated for train/val; held-out sets carry no
    reference summary (it is rule-checked, never string-matched)."""
    if case_type == "submit":
        if variant_idx == 0:
            return (
                f"The denial of {service} is contestable; the record documents "
                f"{_list_phrase(facts_pres)}. Recommend submitting the appeal."
            )
        return (
            f"The member {_list_phrase(facts_pres)}, which addresses the criteria cited. "
            f"Recommend submitting the appeal with this evidence."
        )
    if case_type == "request":
        if variant_idx == 0:
            return f"The record does not show {_list_phrase(missing_labels)}. Recommend requesting more information before appealing."
        return f"Evidence is missing for {_list_phrase(missing_labels)}. Request more information from the treating clinician."
    if case_type == "escalate":
        if variant_idx == 0:
            return f"The record conflicts with the requirement for {contradicted_label}. Recommend escalation to a clinical reviewer."
        return f"The chart contradicts {contradicted_label}, so a clinical reviewer should look at this case before any appeal."
    raise KeyError(case_type)
