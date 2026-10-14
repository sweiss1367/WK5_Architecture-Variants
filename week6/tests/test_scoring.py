import json
import unittest

import fake_fixtures as F
from score_task import completeness, parse_output, score_output


def _case(pred):
    """First eval_h record satisfying a predicate."""
    return next(r for r in F.data("eval_h.jsonl") if pred(r))


class TaskScoring(unittest.TestCase):
    def setUp(self):
        self.rec = _case(lambda r: r["gold"]["policy_cited"] and len(r["gold"]["_ids"]["required"]) >= 1
                         and r["gold"]["_ids"]["case_type"] == "request")
        self.g, self.n, self.s = self.rec["gold"], self.rec["notice"], self.rec["summary"]

    def score(self, raw):
        return score_output(raw, self.g, self.n, self.s)

    def test_ideal_output(self):
        r = self.score(F.ideal_output(self.g))
        self.assertEqual((r["valid"], r["completeness"], r["rationale_omission"], r["unsupported_policy"], r["next_step_correct"]),
                         (1, 1, 0, 0, 1))
        self.assertEqual(r["rationale_recall"], 1.0)
        self.assertEqual((r["criteria_match"], r["missing_match"], r["n_unsupported_facts"]), (1, 1, 0))

    def test_omitted_required_fact(self):
        r = self.score(F.ideal_output(self.g, supporting_clinical_facts=[]))
        self.assertEqual((r["rationale_omission"], r["rationale_recall"]), (1, 0.0))

    def test_extra_words_around_required_fact_still_count(self):
        fact = self.g["supporting_clinical_facts"][0]
        r = self.score(F.ideal_output(self.g, supporting_clinical_facts=[f"The member {fact} per the chart"]))
        self.assertEqual(r["rationale_omission"], 0)

    def test_fabricated_policy_is_unsupported(self):
        r = self.score(F.ideal_output(self.g, policy="MRP-9999"))
        self.assertEqual(r["unsupported_policy"], 1)

    def test_policy_copied_from_notice_is_supported(self):
        r = self.score(F.ideal_output(self.g, policy=self.g["policy_cited"].lower()))
        self.assertEqual(r["unsupported_policy"], 0)

    def test_null_policy_is_not_unsupported(self):
        r = self.score(F.ideal_output(self.g, policy=None))
        self.assertEqual(r["unsupported_policy"], 0)

    def test_fabricated_fact_counted(self):
        r = self.score(F.ideal_output(self.g, supporting_clinical_facts=self.g["supporting_clinical_facts"] + ["completed twelve weeks of aquatic rehabilitation"]))
        self.assertEqual(r["n_unsupported_facts"], 1)
        self.assertEqual(r["rationale_omission"], 0)

    def test_invalid_output_is_omission_and_invalid(self):
        r = self.score("Here is the appeal: the member should be approved.")
        self.assertEqual((r["invalid_or_refusal"], r["refusal"], r["completeness"], r["rationale_omission"]), (1, 0, 0, 1))
        self.assertIsNone(r["unsupported_policy"])
        self.assertIsNone(r["rationale_omission_valid_only"])

    def test_refusal_detected(self):
        r = self.score("I'm sorry, I cannot help with that request.")
        self.assertEqual((r["invalid_or_refusal"], r["refusal"]), (1, 1))

    def test_truncated_json_is_invalid(self):
        raw = F.ideal_output(self.g)[:-20]
        self.assertEqual(self.score(raw)["invalid_or_refusal"], 1)

    def test_completeness_rules(self):
        base = json.loads(F.ideal_output(self.g))
        self.assertEqual(completeness(base), 1)
        self.assertEqual(completeness(dict(base, extra_key=1)), 0)
        self.assertEqual(completeness(dict(base, appeal_summary="One. Two. Three.")), 0)
        self.assertEqual(completeness(dict(base, recommended_next_step="approve")), 0)
        self.assertEqual(completeness(dict(base, supporting_clinical_facts="not a list")), 0)
        self.assertEqual(completeness(None), 0)

    def test_parse_is_strict(self):
        self.assertIsNone(parse_output("```json\n{}\n```"))
        self.assertIsNone(parse_output("[1,2]"))


if __name__ == "__main__":
    unittest.main()
