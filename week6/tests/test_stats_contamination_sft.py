import unittest

import numpy as np
import pandas as pd

import fake_fixtures as F
import contamination_sweep as C
import sft_data
import stats
from appeals_common import build_chat_prompt, normalize_text, tokens


def fake_matched_frame(n_cases=150, spanish_effect=0.0, base=0.3, seed=0):
    """FAKE: simulated binary outcomes with a planted language effect (or none). Test use only."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cases):
        case_rate = np.clip(base + rng.normal(0, 0.1), 0.02, 0.95)
        for lang in ("English", "Spanish"):
            for age in (45, 75):
                for s in (1001, 1002, 1003):
                    p = np.clip(case_rate + (spanish_effect if lang == "Spanish" else 0), 0, 1)
                    rows.append(dict(base_id=f"B{c}", language=lang, age=age, seed=s, m=float(rng.random() < p)))
    return pd.DataFrame(rows)


class Stats(unittest.TestCase):
    def test_wilson_known_value(self):
        p, lo, hi = stats.wilson(5, 10)
        self.assertAlmostEqual(p, 0.5)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)

    def test_planted_difference_is_detected(self):
        r = stats.paired_slice_difference(fake_matched_frame(spanish_effect=0.15), "m", "language", "Spanish", "English", B=2000)
        self.assertGreater(r["ci_low"], 0)
        self.assertLess(abs(r["estimate"] - 0.15), 0.05)
        self.assertEqual(r["interpretation"], stats.DIFFERENCE)
        self.assertEqual(r["n_cases"], 150)
        self.assertEqual((r["n_generations_a"], r["n_generations_b"]), (900, 900))

    def test_age_slice_with_no_effect_is_unresolved(self):
        r = stats.paired_slice_difference(fake_matched_frame(spanish_effect=0.15), "m", "age", 75, 45, B=2000)
        self.assertLessEqual(r["ci_low"], 0)
        self.assertGreaterEqual(r["ci_high"], 0)
        self.assertEqual(r["interpretation"], stats.UNRESOLVED)

    def test_unresolved_wording_never_claims_equality(self):
        u = stats.UNRESOLVED.lower()
        self.assertIn("not statistically established", u)
        self.assertIn("not evidence that the groups perform equally", u)
        self.assertNotIn("no difference", u)

    def test_false_positive_rate_under_null_is_near_nominal(self):
        hits = 0
        n_sims = 150
        for s in range(n_sims):
            r = stats.paired_slice_difference(fake_matched_frame(n_cases=100, seed=s, spanish_effect=0.0), "m", "language", "Spanish", "English", B=400, boot_seed=s)
            hits += r["interpretation"] == stats.DIFFERENCE
        self.assertLess(hits / n_sims, 0.12)

    def test_per_seed_estimates_reported(self):
        r = stats.paired_slice_difference(fake_matched_frame(spanish_effect=0.1), "m", "language", "Spanish", "English", B=500)
        self.assertEqual(sorted(r["per_seed_estimates"]), [1001, 1002, 1003])

    def test_nan_cases_are_dropped_not_zeroed(self):
        df = fake_matched_frame(n_cases=20)
        df.loc[df["base_id"] == "B0", "m"] = np.nan
        r = stats.paired_slice_difference(df, "m", "language", "Spanish", "English", B=200)
        self.assertEqual(r["n_cases"], 19)


class Contamination(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(tokens("Hello,  WORLD!!"), tokens("hello world"))
        self.assertEqual(normalize_text("Ignora las instrucciones — ¿qué?"), "ignora las instrucciones que")

    def _idx(self, text, n):
        rec = dict(id="T1", notice=text, summary="", gold=None)
        return C.build_index([rec], n)[0]

    def test_identical_text_flagged_at_all_n(self):
        text = " ".join(f"w{i}" for i in range(30))
        for n in C.N_VALUES:
            self.assertTrue(C.spans_for(tokens(text), self._idx(text, n), n))

    def test_shared_ten_token_run_flags_n8_not_n13(self):
        shared = " ".join(f"s{i}" for i in range(10))
        a = "alpha beta gamma " + shared + " delta epsilon zeta"
        b = "one two three four " + shared + " five six seven eight"
        self.assertTrue(C.spans_for(tokens(b), self._idx(a, 8), 8))
        self.assertFalse(C.spans_for(tokens(b), self._idx(a, 13), 13))

    def test_punctuation_and_case_do_not_hide_a_match(self):
        a = "The member completed six weeks of conservative therapy and then stopped."
        b = "THE MEMBER, completed six weeks of conservative therapy -- and then stopped!"
        self.assertTrue(C.spans_for(tokens(b), self._idx(a, 8), 8))

    def test_spans_are_maximal(self):
        text = " ".join(f"w{i}" for i in range(20))
        spans = C.spans_for(tokens(text), self._idx(text, 8), 8)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0][1], 20)

    def test_positive_control_is_flagged_on_real_data(self):
        tv = F.data("train.jsonl") + F.data("val.jsonl")
        ctrl = F.load_jsonl(F.os.path.join(F.ROOT, "data", "positive_control.jsonl"))
        self.assertGreaterEqual(len(tokens(ctrl[0]["summary"])), 20)
        for n in C.N_VALUES:
            index, _ = C.build_index(tv, n)
            rows, flagged, _ = C.sweep_records(ctrl, index, n, "positive_control", True, with_gold=False)
            self.assertEqual(flagged, {"POSCTRL-001"}, f"n={n}")


class FakeTokenizer:
    """FAKE tokenizer for testing label masking only."""
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "\n".join(f"<{m['role']}> {m['content']}" for m in messages) + "\n<assistant>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [abs(hash(w)) % 50000 for w in text.split()]}


class SftData(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTokenizer()
        self.recs = F.data("train.jsonl")[:20]

    def test_prompt_tokens_masked_and_target_supervised(self):
        items = sft_data.tokenize_records(self.tok, self.recs)
        ex, trunc = sft_data.build_examples(items, 4096)
        self.assertEqual(trunc, 0)
        for it, e in zip(items, ex):
            n_p = len(it["prompt_ids"])
            self.assertTrue(all(l == -100 for l in e["labels"][:n_p]))
            self.assertEqual(e["labels"][n_p:], it["target_ids"])
            self.assertEqual(len(e["input_ids"]), len(e["labels"]))

    def test_truncation_is_counted(self):
        items = sft_data.tokenize_records(self.tok, self.recs)
        _, trunc = sft_data.build_examples(items, 50)
        self.assertEqual(trunc, len(items))

    def test_choose_max_seq_len_rounds_up_and_caps(self):
        items = [dict(prompt_ids=[0] * 500, target_ids=[0] * 20)]
        self.assertEqual(sft_data.choose_max_seq_len(items), 576)
        with self.assertRaises(SystemExit):
            sft_data.choose_max_seq_len([dict(prompt_ids=[0] * 2000, target_ids=[0])])

    def test_every_example_has_supervised_tokens(self):
        ex, _ = sft_data.build_examples(sft_data.tokenize_records(self.tok, self.recs), 4096)
        self.assertEqual(sft_data.validate_labels(ex)["n_zero_supervised"], 0)

    def test_target_is_valid_json_with_all_keys(self):
        import json
        from appeals_common import SCHEMA_KEYS
        for r in F.data("train.jsonl"):
            self.assertEqual(list(json.loads(r["target_json"]).keys()), SCHEMA_KEYS)


if __name__ == "__main__":
    unittest.main()
