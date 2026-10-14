# Contamination sweep: results and review of every flagged match

Produced by `src/contamination_sweep.py` and `src/review_contamination.py` on the frozen data
(manifest sha256 `29cfecf0ca584a16...`). Normalisation: NFKD, strip accents, lowercase, non-alphanumeric runs -> single space, word tokens.
Evaluation fields are compared against train + val text, one field at a time. The datasets were not
changed in response to these results (decision recorded in `logs/freeze_history.md`).

## Results

| dataset | n | items | flagged | % flagged | not explained after review |
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

* **n = 8 produced substantial lexical overlap** (72% of held-out cases, 72% of matched cases, 78% of probes).
* **Manual review showed every n = 8 hit came from the intentionally shared synthetic policy vocabulary**:
  the `MRP-####` policy identifier and the "<Service> Coverage Policy" wording, plus neighbouring text (at most
  one coincidentally adjacent word, "policy" before or "we" after). All 618 reviewed matches
  collapse to 35 unique spans, every one of which was read. Unexplained rows: 0.
* **Classification: expected shared-domain vocabulary / boilerplate, not confirmed train/evaluation leakage.**
  The policy library is shared by design so the model can cite the policy IDs it was trained on.
* **n = 13 and n = 20 produced zero real evaluation matches.**
* **The synthetic positive control was detected at n = 8, 13 and 20.** It is a record built with a training
  summary copied in, is not naturally occurring contamination, and is excluded from every natural-overlap
  percentage above.
* The 618 reviewed n = 8 matches (`results/contamination_matches.csv`) and the 35 unique spans
  (`results/contamination_unique_spans.csv`) are preserved as audit evidence.

## What this does not show

**Zero matches at n = 13 and n = 20 do not prove the evaluation set is uncontaminated.** N-gram screening
misses paraphrase, structural or template reuse, translation, and other non-verbatim leakage, and it cannot
see contamination of the base model's pretraining data. Fact phrases and criterion labels are mostly shorter
than 8 words, so they cannot produce n-grams at these n; their separation between training (vocabulary A) and
held-out (vocabulary B) wording is by construction and was checked by `audit_dataset.py`, not by this sweep.
The held-out sets are held out by construction, so a near-zero result at larger n is partly a property of the
generator. Probes SAF-LEAK-14..20 intentionally reuse training members' names and ID prefixes; they produced
no 8-gram match.
