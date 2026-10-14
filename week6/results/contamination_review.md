# Contamination sweep: review of every flagged match

Produced by `src/contamination_sweep.py` then `src/review_contamination.py` on the frozen data
(manifest sha256 `29cfecf0ca584a16...`). Normalisation: NFKD, strip accents, lowercase, non-alphanumeric runs -> single space, word tokens.
Evaluation fields are compared against train + val text, one field at a time (no n-gram spans two fields).

## Result

| dataset | n | items | flagged | % flagged | unexplained after review |
|---|---|---|---|---|---|
| eval_h | 8 | 150 | 108 | 72.0 | 0 (0.0%) |
| fairness_matched | 8 | 600 | 432 | 72.0 | 0 (0.0%) |
| probe_suite | 8 | 100 | 78 | 78.0 | 0 (0.0%) |
| eval_h | 13 | 150 | 0 | 0.0 | 0 (0.0%) |
| fairness_matched | 13 | 600 | 0 | 0.0 | 0 (0.0%) |
| probe_suite | 13 | 100 | 0 | 0.0 | 0 (0.0%) |
| eval_h | 20 | 150 | 0 | 0.0 | 0 (0.0%) |
| fairness_matched | 20 | 600 | 0 | 0.0 | 0 (0.0%) |
| probe_suite | 20 | 100 | 0 | 0.0 | 0 (0.0%) |

* The detector works: the synthetic positive control (a held-out-style record whose clinical summary is
  copied verbatim from a training record) is flagged at n = 8, 13 and 20. It is **not** naturally
  occurring contamination and is excluded from the percentages above.
* n = 13 and n = 20 flag nothing.
* n = 8 flags 618 item-rows across the three evaluation files. All 618 match rows
  collapse to 35 distinct spans, **all of which were read**. Every one is a policy-library display string,
  "MRP-#### (<Service> Coverage Policy)", plus at most one coincidentally adjacent word ("policy" before it or
  "we" after it). `review_contamination.py` verifies that each row fits this pattern and fails if any does not.
  Unexplained rows: 0.

## What this does and does not show

* The closed policy library (policy IDs, titles, service names) is shared between training and evaluation **by
  design**: the model has to be able to cite the same policy IDs it was trained on. The n = 8 hits are that
  shared identifier string, not a leaked evaluation case. It is a real structural overlap channel and is
  reported as one.
* Nothing else at n >= 8 was shared: no sentence skeleton, notice boilerplate, clinical-summary wording or
  criterion wording overlaps between training and held-out text at these n.
* **A negative n-gram result does not rule out paraphrase, translation, structural/template contamination, or
  contamination of the base model's pretraining data.** Fact phrases and criterion labels are mostly shorter
  than 8 words, so they cannot produce n-grams at these n at all; they are kept separate between training
  (vocabulary A) and held-out (vocabulary B) wording by construction and checked by `audit_dataset.py`, not by
  this sweep.
* The held-out sets are held out by construction, so a near-zero result is partly a property of the generator.
  The sweep guards against accidental overlap; it is not independent proof that nothing is shared.
* Probes SAF-LEAK-14..20 intentionally reuse training members' names and ID prefixes. They did not produce an
  8-gram match.
