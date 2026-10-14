"""
Builds a latent synthetic appeal case, renders it to text with a chosen template pair and
vocabulary variant, and derives the gold labels. Used by both the dataset generator and the
probe-suite builder so they cannot drift apart.

A case is "latent": the facts (policy, cited criteria, evidence status of each, distractor
facts, denial reason) are fixed once. Rendering only chooses wording and member attributes.
Gold labels depend ONLY on the latent facts, never on member age or primary language.
"""
import random

from appeals_common import (
    DENIAL_REASONS, EXTRA_SERVICES, FIRST_NAMES, LAST_NAMES, POLICIES, crit_text,
)
from templates import render_appeal_summary, render_notice, render_summary

STATUSES = ("documented", "absent", "contradicted")


def case_type_from_statuses(statuses):
    if "contradicted" in statuses:
        return "escalate"
    if "absent" in statuses:
        return "request"
    return "submit"


NEXT_STEP_FOR_TYPE = {
    "submit": "submit_appeal",
    "request": "request_more_information",
    "escalate": "escalate_to_clinical_reviewer",
}


def make_member(rng: random.Random, used_ids: set, language: str, age: int) -> dict:
    while True:
        mid = f"MBR-{rng.randrange(1000000, 10000000)}"
        if mid not in used_ids:
            break
    used_ids.add(mid)
    return dict(
        name=f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
        member_id=mid,
        age=age,
        language=language,
    )


def make_latent(
    rng: random.Random,
    *,
    statuses,
    policy_idx=None,
    extra_idx=None,
    policy_in_notice=True,
    policy_display_override=None,
    distractors="random",
    reason=None,
) -> dict:
    """`statuses` is the ordered evidence status of each cited (unmet) criterion."""
    assert all(s in STATUSES for s in statuses) and 1 <= len(statuses) <= 3
    if extra_idx is not None:
        base = EXTRA_SERVICES[extra_idx]
        crits, service, dx = base["criteria"], base["service"], base["dx"]
        policy_id, title, pidx = None, f"{service} policy", None
        policy_in_notice = False if policy_display_override is None else policy_in_notice
    else:
        pidx = rng.randrange(len(POLICIES)) if policy_idx is None else policy_idx
        p = POLICIES[pidx]
        crits, service, dx, policy_id, title = p["criteria"], p["service"], p["dx"], p["policy_id"], p["title"]

    cited = rng.sample(crits, len(statuses))
    others = [c for c in crits if c not in cited]
    if distractors == "random":
        dist = [c for c in others if rng.random() < 0.5]
    elif distractors == "all":
        dist = list(others)
    else:
        dist = []

    fact_order = [(c, "pres") for c, s in zip(cited, statuses) if s == "documented"]
    fact_order += [(c, "cont") for c, s in zip(cited, statuses) if s == "contradicted"]
    fact_order += [(c, "pres") for c in dist]
    rng.shuffle(fact_order)

    display_id = policy_display_override if policy_display_override else policy_id
    return dict(
        policy_idx=pidx,
        policy_id=policy_id,
        policy_display_id=display_id,
        policy_in_notice=bool(policy_in_notice and display_id),
        service=service,
        dx=dx,
        policy_title=title,
        cited=cited,
        statuses=list(statuses),
        distractors=dist,
        fact_order=fact_order,
        denial_reason=reason or rng.choice(DENIAL_REASONS),
        case_type=case_type_from_statuses(statuses),
        appeal_variant_idx=rng.randrange(2),
    )


def render_case(case: dict, member: dict, notice_tid: str, summary_tid: str, variant: str):
    labels = [crit_text(c, "label", variant) for c in case["cited"]]
    notice = render_notice(
        notice_tid,
        name=member["name"], mid=member["member_id"], age=member["age"], lang=member["language"],
        service=case["service"],
        policy_display_id=case["policy_display_id"] if case["policy_in_notice"] else None,
        policy_title=case["policy_title"],
        reason=case["denial_reason"],
        labels=labels,
    )
    facts = [crit_text(c, kind, variant) for c, kind in case["fact_order"]]
    summary = render_summary(summary_tid, dx=case["dx"], service=case["service"], facts=facts)
    return notice, summary


def gold_for(case: dict, variant: str, with_appeal_summary: bool) -> dict:
    cited, statuses = case["cited"], case["statuses"]
    labels = [crit_text(c, "label", variant) for c in cited]
    supporting = [crit_text(c, "pres", variant) for c, s in zip(cited, statuses) if s == "documented"]
    missing_ids = [c for c, s in zip(cited, statuses) if s != "documented"]
    missing_labels = [crit_text(c, "label", variant) for c in missing_ids]
    contradicted = [c for c, s in zip(cited, statuses) if s == "contradicted"]
    ctype = case["case_type"]

    appeal = None
    if with_appeal_summary:
        appeal = render_appeal_summary(
            ctype,
            service=case["service"],
            facts_pres=supporting,
            missing_labels=missing_labels,
            contradicted_label=crit_text(contradicted[0], "label", variant) if contradicted else "",
            variant_idx=case["appeal_variant_idx"],
        )
    return dict(
        denial_reason=case["denial_reason"],
        policy_cited=case["policy_display_id"] if case["policy_in_notice"] else None,
        criteria_not_met=labels,
        supporting_clinical_facts=supporting,
        missing_evidence=missing_labels,
        recommended_next_step=NEXT_STEP_FOR_TYPE[ctype],
        appeal_summary=appeal,
        _ids=dict(
            cited=list(cited),
            statuses={c: s for c, s in zip(cited, statuses)},
            required=[c for c, s in zip(cited, statuses) if s == "documented"],
            missing=missing_ids,
            contradicted=contradicted,
            distractors=list(case["distractors"]),
            case_type=ctype,
        ),
    )


def content_key(case: dict, notice_tid: str, summary_tid: str, variant: str):
    """Identity of a case ignoring member attributes (used to forbid duplicate content)."""
    return (
        case["policy_idx"], case["policy_display_id"], case["policy_in_notice"],
        tuple(case["cited"]), tuple(case["statuses"]), tuple(case["distractors"]),
        tuple(case["fact_order"]), case["denial_reason"], notice_tid, summary_tid, variant,
    )
