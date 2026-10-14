"""
FIXTURES FOR TESTING THE SCORING AND STATISTICS CODE. NOT RESULTS.

Everything built here is hand-made or randomly simulated so the scorers, probe rules and
bootstrap code can be exercised before any real model has run. None of it is model output, none
of it may be quoted as evidence, and no file in results/ is ever written from it.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from appeals_common import SCHEMA_KEYS, load_jsonl  # noqa: E402

NEUTRAL_SUMMARY = "The record needs review by the appeals team."


def data(name):
    return load_jsonl(os.path.join(ROOT, "data", name))


def probes():
    return load_jsonl(os.path.join(ROOT, "probes", "probe_suite.jsonl"))


def ideal_output(gold, next_step=None, policy="__gold__", summary=NEUTRAL_SUMMARY, **overrides):
    """FAKE_IDEAL: the output a perfect model would give for this gold."""
    obj = dict(
        denial_reason=gold["denial_reason"],
        policy_cited=gold["policy_cited"] if policy == "__gold__" else policy,
        criteria_not_met=list(gold["criteria_not_met"]),
        supporting_clinical_facts=list(gold["supporting_clinical_facts"]),
        missing_evidence=list(gold["missing_evidence"]),
        recommended_next_step=next_step or gold["recommended_next_step"],
        appeal_summary=summary,
    )
    obj.update(overrides)
    return json.dumps({k: obj[k] for k in obj})


def ideal_for_probe(p):
    g = p["meta"]["gold"]
    params = p["meta"]["params"]
    step = params.get("expected_step") or g["recommended_next_step"]
    policy = params.get("expected_policy_cited", "__gold__") if p["scoring_rule_id"] == "POL" else "__gold__"
    return ideal_output(g, next_step=step, policy=policy)
