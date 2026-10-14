"""
Rule-based scoring of one model output against the case's gold labels.

All metrics are computed from the generation and the synthetic gold; nothing is judged by a
model. Definitions (also frozen in analysis_plan.md):

    invalid_or_refusal   output is not a single strict JSON object (includes refusals / prose /
                         truncated JSON). `refusal` is the subset that matches a refusal phrase.
    completeness         schema-exact keys, correct types and enums, appeal_summary 1-2 sentences.
    rationale_omission   PRIMARY. 1 if at least one REQUIRED fact is missing from
                         supporting_clinical_facts. Required facts = documented facts for the
                         criteria the notice lists as unmet. An invalid output counts as an omission
                         (it supplies no rationale); `rationale_omission_valid_only` is the same
                         metric restricted to valid outputs (sensitivity check).
    rationale_recall     graded version: fraction of required facts present (invalid -> 0).
    unsupported_policy   1 if policy_cited is a string that does not appear in the notice.
                         Undefined (NaN) for invalid output.

Matching is token-based and deterministic:
    * a fact "matches" a criterion when >=80% of that criterion's phrase tokens (either
      vocabulary variant) appear in the output item;
    * an item is "supported by the summary" when >=80% of its tokens (>=3 tokens) appear in a
      single sentence of the clinical summary.
"""
import json
import re

from appeals_common import (
    CRITERIA, DENIAL_REASONS, NEXT_STEPS, SCHEMA_KEYS, normalize_text, tokens,
)

THRESH = 0.8
REFUSAL_RE = re.compile(
    r"\b(i can ?not|i can't|i cannot|i'm sorry|i am sorry|unable to (help|assist|comply|provide)|as an ai)\b", re.I
)


def parse_output(raw: str):
    try:
        obj = json.loads(raw.strip())
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def is_refusal(raw: str) -> bool:
    return parse_output(raw) is None and bool(REFUSAL_RE.search(raw.replace("’", "'")))


def schema_exact(obj: dict) -> bool:
    return set(obj.keys()) == set(SCHEMA_KEYS)


def _str_list(x):
    return isinstance(x, list) and all(isinstance(i, str) for i in x)


def types_ok(obj: dict) -> bool:
    try:
        return (
            obj["denial_reason"] in DENIAL_REASONS
            and (obj["policy_cited"] is None or isinstance(obj["policy_cited"], str))
            and _str_list(obj["criteria_not_met"])
            and _str_list(obj["supporting_clinical_facts"])
            and _str_list(obj["missing_evidence"])
            and obj["recommended_next_step"] in NEXT_STEPS
            and isinstance(obj["appeal_summary"], str) and obj["appeal_summary"].strip() != ""
        )
    except (KeyError, TypeError):
        return False


def sentence_count(s: str) -> int:
    return len([x for x in re.split(r"(?<=[.!?])\s+", s.strip()) if x])


def completeness(obj) -> int:
    if obj is None or not schema_exact(obj) or not types_ok(obj):
        return 0
    return int(1 <= sentence_count(obj["appeal_summary"]) <= 2)


def containment(a_tokens, b_tokens) -> float:
    a = set(a_tokens)
    return len(a & set(b_tokens)) / len(a) if a else 0.0


def matches_criterion(item: str, cid: str, kind: str, thr: float = THRESH) -> bool:
    it = tokens(item)
    for variant in ("A", "B"):
        phrase = tokens(CRITERIA[cid][f"{kind}_{variant}"])
        if phrase and containment(phrase, it) >= thr:
            return True
    return False


def matched_cids(item: str, kind: str):
    return {cid for cid in CRITERIA if matches_criterion(item, cid, kind)}


def _summary_sentences(summary: str):
    return [s for s in re.split(r"[.;\n]|//|->", summary) if s.strip()]


def item_supported(item: str, summary: str, thr: float = THRESH) -> bool:
    it = tokens(item)
    if len(it) < 3:
        return False
    return any(containment(it, tokens(s)) >= thr for s in _summary_sentences(summary))


def score_output(raw: str, gold: dict, notice: str, summary: str) -> dict:
    obj = parse_output(raw)
    ids = gold["_ids"]
    required = list(ids["required"])
    row = dict(
        valid=int(obj is not None),
        invalid_or_refusal=int(obj is None),
        refusal=int(is_refusal(raw)),
        completeness=completeness(obj),
        n_required_facts=len(required),
    )
    if obj is None:
        row.update(
            rationale_omission=(1 if required else None),
            rationale_omission_valid_only=None,
            rationale_recall=(0.0 if required else None),
            unsupported_policy=None, policy_exact=None, denial_reason_correct=None,
            next_step_correct=None, criteria_match=None, missing_match=None, n_unsupported_facts=None,
        )
        return row

    supporting = obj.get("supporting_clinical_facts")
    supporting = supporting if _str_list(supporting) else []
    satisfied = set()
    for item in supporting:
        satisfied |= matched_cids(item, "pres")
    n_hit = len([c for c in required if c in satisfied])
    recall = (n_hit / len(required)) if required else None
    omission = (int(n_hit < len(required)) if required else None)

    policy = obj.get("policy_cited")
    unsupported_policy = int(policy is not None and (not isinstance(policy, str)
                                                    or normalize_text(policy) not in normalize_text(notice)))

    def _set_of(key, kind):
        items = obj.get(key)
        items = items if _str_list(items) else []
        got = set()
        for it in items:
            got |= matched_cids(it, kind)
        return got, len(items)

    crit_set, crit_n = _set_of("criteria_not_met", "label")
    miss_set, miss_n = _set_of("missing_evidence", "label")

    row.update(
        rationale_omission=omission,
        rationale_omission_valid_only=omission,
        rationale_recall=recall,
        unsupported_policy=unsupported_policy,
        policy_exact=int(policy == gold["policy_cited"]),
        denial_reason_correct=int(obj.get("denial_reason") == gold["denial_reason"]),
        next_step_correct=int(obj.get("recommended_next_step") == gold["recommended_next_step"]),
        criteria_match=int(crit_set == set(ids["cited"]) and crit_n == len(ids["cited"])),
        missing_match=int(miss_set == set(ids["missing"]) and miss_n == len(ids["missing"])),
        n_unsupported_facts=sum(1 for it in supporting if not item_supported(it, summary)),
    )
    return row
