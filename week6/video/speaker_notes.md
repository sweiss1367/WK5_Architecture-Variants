# Briefing notes for the Chief Risk Officer (camera on, target 7 to 8 minutes)

Cue points only. Say it in your own words, keep your eyes on the camera, glance at the notes. Every number
below comes from `results/` and `governance/risk_register.xlsx`; if you change a number, check it there.

## Open (0:00 to 0:45): lead with the top risk
- Ask the CRO to decide one thing: whether this model may route appeals. My answer: **not yet.**
- Top risk (R1): when a clinical summary **contradicts** a criterion, or flags **urgent symptoms**, the model does
  not escalate. It recommended escalation for **0 of 22** such held-out cases (0 of 8 on validation) and for
  18 of 264 when sampled, even though 20% of its training examples were escalations.
- Most of those cases came out as "submit appeal" or "request more information".
- One line on scope: a small model, synthetic data, one adapter. It shows what the audit method can find, not
  what a production system would do.

## R1 in detail (0:45 to 2:15)
- Show `probe_failures_by_family.png`. Overconfidence probes: 13 of 20 failed. All 6 contradicted-evidence and all 5
  urgent-symptom probes failed.
- Say what did **not** happen: no certainty language, no medical advice in any of 80 runs. The failure is quiet:
  it just routes the case wrong.
- Next-step accuracy on held-out cases: 68% (interval 60 to 75).
- Why it matters: this is the case a clinician most needs to see.

## The other high-priority risks (2:15 to 3:45)
- **Injection (R2):** 7 of 20 probes hijacked the output on greedy and on every seed. The bare word "APPROVED" instead
  of JSON on all 4 notice overrides; the fake assistant turn echoed on all 3 delimiter breaks. Ten more probes
  diverge only in the recommended step: be clear I cannot attribute those to the injection.
- **Evidence (R3):** 9 of 20 fabrication probes failed, mostly by using an unrelated documented fact as support, not
  by inventing text. Held-out rationale omission 28% (21 to 36).
- **Instability (R4):** same prompt, three sampling seeds: the recommendation changed for 33% of prompts.

## What I cannot conclude (3:45 to 4:45)
- **Fairness:** primary-language and age markers. Differences in required-rationale omission: language 0.00 points
  (interval -2.3 to 2.3), age -0.2 points (-3.0 to 2.6), 150 matched cases. Both intervals include zero, so the
  result is **unresolved**. It is not evidence the groups are treated equally. One adapter, one decoding setting,
  a header marker only, no non-English text.
- **Leakage and policy hallucination:** 0 failures in 40 probes, 0 of 160 runs. Encouraging, but 20 probes per family
  and a fixed output schema limit what that means.

## Evidence quality and the frontier claim (4:45 to 6:15)
- Contamination test: n=8 word overlap flagged 72% of held-out cases, and every match was the intentionally shared
  synthetic policy wording. n=13 and n=20 found nothing. A planted control was detected. **Zero matches does not prove
  the set is clean:** the test misses paraphrase, structure, translation.
- Frontier claim: Wei et al. report abilities that appear abruptly with scale. Schaeffer et al. show much of that
  sharpness comes from the metric chosen (two all-or-nothing metrics account for over 92% of the claimed cases).
  Neither paper tests a fine-tuned small model or any of our risks.
- So what: do not argue from scaling curves. Measure the metric that carries the harm, on the model you deploy, with
  intervals. Our audit has one model size and cannot test emergence.

## Pilot gates and the ask (6:15 to 7:30)
- Blocking: no model-driven routing until escalation is demonstrated on real-format cases (R1). Input handling and
  output validation tested before any third-party text reaches the model (R2).
- Required: a human verifies every cited fact (R3); deterministic decoding and logging (R4); a real-format
  validation sample (R5, R7).
- Condition: disaggregated monitoring plan; **no fairness assurance statement** from this evidence (R6).
- State the status honestly: every control is proposed, none implemented or tested, so no risk reduction is claimed.
- Close on the decision requested of the CRO.

## Show on screen
`probe_failures_by_family.png` (R1/R2/R3), `fairness_slice_differences.png` (hollow markers = unresolved),
`contamination_flagged.png` (brief), and the Register sheet sorted by score.

## Do not say
- "The model is fair" or "no bias" (unresolved is not equal).
- "The test set is clean" (zero n-gram matches is not proof).
- Anything about real patients or clinical validity (all data synthetic).
- That a control reduced a risk (none was tested).
- That injection explains all 17 failed injection probes (7 are clear compliance; 10 are not attributable).
