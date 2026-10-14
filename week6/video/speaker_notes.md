# Briefing notes for the Chief Risk Officer (camera on, target 7 to 8 minutes)

Cue points only. Say it in your own words, keep your eyes on the camera, glance at the notes. Every number
below comes from `results/` and `governance/risk_register.xlsx`; if you change a number, check it there.
**Order follows the register ranking:** R1, R3, R2, R4, R5, R7, R6, R8.

## Open (0:00 to 0:40): the decision and the top risk
- The decision asked of the CRO: **do not clear an operational pilot yet.** A tightly controlled shadow evaluation
  can continue, but the model should not drive routing or produce member-facing appeal content until the
  escalation, evidence-grounding, injection and real-format validation requirements are met.
- Top risk (R1, score 20): when a clinical summary **contradicts** a criterion, or flags **urgent symptoms**, the
  model does not escalate.
- Scope in one line: small model, synthetic data, one adapter. It shows what this audit method can find, not what a
  production system would do.

## R1, non-escalation (0:40 to 2:00)
- It recommended escalation for **0 of 22** such held-out cases (0 of 8 on validation) and for 18 of 264 when
  sampled, although 20% of its training examples were escalations.
- Show `probe_failures_by_family.png`. Overconfidence probes: 13 of 20 failed, including all 6 contradicted-evidence and
  all 5 urgent-symptom probes. Cases came out as "submit appeal" or "request more information".
- Say what did **not** happen: no certainty language and no medical advice in any of 80 runs. The failure is quiet:
  it routes the case wrong.
- Held-out next-step accuracy 68% (interval 60 to 75). This is the case a clinician most needs to see.

## R3, evidence misattributed (2:00 to 2:50)
- 9 of 20 fabrication probes failed on greedy. Mostly the model used an unrelated documented fact as support or did not
  flag missing evidence; invented text was rare.
- Held-out rationale omission 28% (21 to 36). Gate: a human verifies every cited fact.

## R2, prompt injection (2:50 to 3:40)
- 7 of 20 probes hijacked the output on greedy and on every seed: the bare word "APPROVED" instead of JSON on all 4
  notice overrides, and the fake assistant turn echoed on all 3 delimiter breaks.
- Ten more probes differ only in the recommended step. I cannot attribute those to the injection (no un-injected control).

## R4 and R5, reliability (3:40 to 4:30)
- R4: same prompt, three sampling seeds: the recommendation changed for **33%** of prompts (199 of 600).
- R5: on formats it did not train on, criteria were identified correctly 79% of the time versus 100% in-template
  (intervals do not overlap). Real notices will not look like our templates.

## R7, what the evidence can and cannot show (4:30 to 5:50)
- Contamination test: n=8 word overlap flagged 72% of held-out cases, and every match was the intentionally shared synthetic
  policy wording (expected vocabulary, not leakage). n=13 and n=20 found nothing. A planted control was detected.
  **Zero matches does not prove the set is clean:** it misses paraphrase, structure, translation.
- Frontier claim: Wei et al. report abilities that appear abruptly with scale. Schaeffer et al. show much of that
  sharpness comes from the metric chosen (two all-or-nothing metrics account for over 92% of the claimed cases).
  Neither paper tests a fine-tuned small model or any of our risks. Do not argue from scaling curves: measure the metric
  that carries the harm, on the deployed model, with intervals. Our audit has one model size and cannot test emergence.

## R6, fairness (5:50 to 6:40)
- Primary-language and age markers, required-rationale omission: language 0.00 points (interval -2.3 to 2.3), age
  -0.2 points (-3.0 to 2.6), 150 matched cases. Both intervals include zero, so the result is **unresolved**. It is not
  evidence of equal treatment. One adapter, a header marker only, no non-English text.
- Ask: no fairness assurance statement from this evidence; monitor by group on real pilot cases.

## R8, no failures observed, and the close (6:40 to 7:30)
- Policy hallucination and member-information leakage: 0 failures in 40 probes, 0 of 160 runs. Encouraging, but 20
  probes per family and a fixed output schema limit what that means.
- Gates: **blocking** (R1 escalation shown on real-format cases; R2 input handling and output validation tested),
  **required** (R3 human verification, R4 deterministic decoding and logging, R5 and R7 real-format validation set),
  **condition** (R6 monitoring plan).
- Status: every control is proposed, none implemented or tested, so no risk reduction is claimed.
- Close on the decision requested.

## Show on screen
`probe_failures_by_family.png` (R1, R3, R2), the register sorted by score, `fairness_slice_differences.png` (hollow
markers = unresolved), `contamination_flagged.png` (brief).

## Do not say
- "The model is fair" or "no bias" (unresolved is not equal).
- "The test set is clean" (zero n-gram matches is not proof).
- Anything about real patients or clinical validity (all data synthetic).
- That a control reduced a risk (none was tested).
- That injection explains all 17 failed injection probes (7 are clear compliance; 10 are not attributable).
