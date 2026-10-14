import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import fake_fixtures as F  # noqa: F401  (sets sys.path)
import generate_appeals_data as G
from appeals_common import CRITERIA, normalize_text


class DataGeneration(unittest.TestCase):
    def test_generation_is_deterministic(self):
        a, b = G.generate(627), G.generate(627)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        c = G.generate(628)
        self.assertNotEqual(a["train"][0]["notice"], c["train"][0]["notice"])

    def test_disk_files_match_regeneration(self):
        fresh = G.generate(627)
        for key, fname in (("train", "train.jsonl"), ("eval_h", "eval_h.jsonl"), ("fairness_matched", "fairness_matched.jsonl")):
            self.assertEqual(F.data(fname), json.loads(json.dumps(fresh[key])), fname)

    def test_sizes(self):
        self.assertEqual([len(F.data(f)) for f in ("train.jsonl", "val.jsonl", "eval_h.jsonl", "fairness_matched.jsonl")],
                         [400, 40, 150, 600])
        self.assertEqual(len(F.probes()), 100)

    def test_gold_independent_of_language_and_age_in_matched_set(self):
        by = {}
        for r in F.data("fairness_matched.jsonl"):
            by.setdefault(r["base_id"], set()).add(json.dumps(r["gold"], sort_keys=True))
        self.assertTrue(all(len(v) == 1 for v in by.values()))

    def test_every_eval_case_has_a_required_fact(self):
        self.assertTrue(all(r["gold"]["_ids"]["required"] for r in F.data("eval_h.jsonl")))

    def test_balanced_attributes_within_case_type_in_train(self):
        from collections import Counter
        c = Counter((r["meta"]["case_type"], r["meta"]["member"]["language"]) for r in F.data("train.jsonl"))
        for t in ("submit", "request", "escalate"):
            self.assertLessEqual(abs(c[(t, "English")] - c[(t, "Spanish")]), 1)

    def test_no_heldout_vocabulary_in_train(self):
        tv = normalize_text(" ".join(r["notice"] + " " + r["summary"] for r in F.data("train.jsonl") + F.data("val.jsonl")))
        for c in CRITERIA.values():
            for k in ("label_B", "pres_B", "cont_B"):
                self.assertNotIn(normalize_text(c[k]), tv)


class AuditCatchesProblems(unittest.TestCase):
    """The audit must FAIL when the data is wrong, not just pass on good data."""

    def _run_audit(self, mutate):
        tmp = tempfile.mkdtemp()
        try:
            shutil.copytree(os.path.join(F.ROOT, "data"), os.path.join(tmp, "data"))
            shutil.copytree(os.path.join(F.ROOT, "probes"), os.path.join(tmp, "probes"))
            mutate(tmp)
            out = os.path.join(tmp, "audit.json")
            p = subprocess.run([sys.executable, os.path.join(F.ROOT, "src", "audit_dataset.py"), "--data-dir", os.path.join(tmp, "data"),
                                "--probes", os.path.join(tmp, "probes", "probe_suite.jsonl"), "--out", out],
                               capture_output=True, text=True)
            with open(out) as fh:
                return p.returncode, json.load(fh)["failed"]
        finally:
            shutil.rmtree(tmp)

    def test_clean_data_passes(self):
        rc, failed = self._run_audit(lambda t: None)
        self.assertEqual((rc, failed), (0, []))

    def test_changed_summary_in_matched_pair_is_caught(self):
        def mut(t):
            p = os.path.join(t, "data", "fairness_matched.jsonl")
            with open(p) as fh:
                rows = [json.loads(l) for l in fh]
            rows[1]["summary"] += " The member is elderly."
            with open(p, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
        rc, failed = self._run_audit(mut)
        self.assertEqual(rc, 1)
        self.assertIn("fairness_pairs_identical_except_two_header_values", failed)

    def test_train_template_in_heldout_is_caught(self):
        def mut(t):
            p = os.path.join(t, "data", "eval_h.jsonl")
            with open(p) as fh:
                rows = [json.loads(l) for l in fh]
            rows[0]["notice"] = "DENIAL NOTICE\n" + rows[0]["notice"]
            with open(p, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
        rc, failed = self._run_audit(mut)
        self.assertEqual(rc, 1)
        self.assertIn("no_template_signature_crosses_split", failed)

    def test_wrong_gold_is_caught(self):
        def mut(t):
            p = os.path.join(t, "data", "train.jsonl")
            with open(p) as fh:
                rows = [json.loads(l) for l in fh]
            rows[0]["gold"]["recommended_next_step"] = "submit_appeal" if rows[0]["gold"]["recommended_next_step"] != "submit_appeal" else "request_more_information"
            with open(p, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
        rc, failed = self._run_audit(mut)
        self.assertEqual(rc, 1)
        self.assertIn("gold_consistent_with_text", failed)


if __name__ == "__main__":
    unittest.main()
