# Critique: "Emergent Abilities of Large Language Models" (Wei et al.), with Schaeffer et al. as contrasting evidence

**Sources, as read.** Wei, Tay, Bommasani, Raffel, Zoph, Borgeaud, Yogatama, Bosma, Zhou, Metzler, Chi, Hashimoto, Vinyals, Liang, Dean, Fedus. *Emergent Abilities of Large Language Models.* Transactions on Machine Learning Research (08/2022), arXiv:2206.07682v2 (all 30 pages read). Schaeffer, Miranda, Koyejo. *Are Emergent Abilities of Large Language Models a Mirage?* arXiv:2304.15004v2, 22 May 2023, marked "Preprint. Under review." (all 14 pages read). Page numbers below are the printed page numbers of those PDFs.

**Convention.** `[Wei p.X]` and `[Sch p.X]` mark what a paper says or shows. `[Inference]` marks my own reading. Anything I could not find in the text is stated as "not in the text I read", not as "the authors did not do it".

---

## 1. The claim, anatomy

| Element | Wei et al. | Where |
|---|---|---|
| Definition | "An ability is emergent if it is not present in smaller models but is present in larger models." Near-random performance until a critical scale, then well above random: a "phase transition". | Wei p.2 |
| What is unpredictable | Emergent abilities "cannot be predicted simply by extrapolating the performance of smaller models." | Wei abstract, p.1–2 |
| What was measured | Few-shot task scores (accuracy, exact match, BLEU) of separate pretrained models from five families (GPT-3, LaMDA, Gopher, Chinchilla, PaLM), plotted against training FLOPs. "Each point is a separate model." Panels A–D are BIG-Bench, 2-shot. Also augmented prompting and finetuning methods (chain of thought, instruction tuning, scratchpad, calibration). | Wei Fig 2, p.4; Fig 3, p.5; Table 1, p.6 |
| Evidence base | A survey of results from other papers: "we surveyed results in the existing literature, without proposing new methods or models." | Wei p.11 |

## 2. What the evidence supports, what the headline suggests

**Supported (what was actually measured).** For the listed tasks and metrics, the smaller models in a family score near the random baseline and the largest score well above it (Fig 2, Table 1). [Wei p.4, p.6]

**Headline interpretation.** That capabilities appear abruptly and unpredictably with scale: a qualitative change in what the model can do, with the implication that "additional scaling could potentially further expand the range of capabilities" [Wei p.1], and that new abilities, including "emergent risks", may arrive without warning [Wei p.8–9].

**The gap between them, using Wei's own pages.**
- Wei's own cross-entropy analysis shows loss *improving* at small scales while exact match, BLEU and accuracy sit at chance, for all six tasks tested ("Outcome 2"). They conclude small models "do improve in some ways that downstream metrics ... do not capture" [Wei p.19, Fig 5 p.20, Fig 6 p.21] and that this "does not explain why downstream metrics are emergent or enable us to predict the scale at which emergence occurs" [Wei p.7]. So what is demonstrated is a threshold in a *downstream score*, not an abrupt change in the underlying model.
- The scale of emergence is "not an immutable property of the ability" and depends on data quality and other factors [Wei p.2–3]; PaLM 62B is emergent on BIG-Bench tasks where GPT-3 175B and LaMDA 137B are not [Wei p.7, App F p.30]. Scale alone is therefore not the explanatory variable.
- Labeling a task "emergent" was done by hand and is subjective: two co-authors classified 210 BIG-Bench tasks by whether performance was "near-random until a certain scale" [Wei App A.3 p.22].

## 3. Metric discontinuity: Wei's partial answer and Schaeffer's challenge

**Wei's partial answer.** Wei flags that exact match "may disguise compounding incremental improvements as emergence", but argues partial-credit metrics are "at best an incomplete explanation, because emergent abilities are still observed on many classification tasks (Figure 2D–H)" [Wei p.7]. In App A.2 three generative BIG-Bench tasks were rescored with BLEU, ROUGE, BLEURT and others, and "emergent behavior appears to be independent of which evaluation metric is used" [Wei p.22, Fig 7]. Two emergent generative tasks (word unscramble, repeat copy logic) were excluded because exact match is the only sensible metric for them [Wei p.22].

**Schaeffer's mechanism.** If per-token cross-entropy falls smoothly, then a metric that requires all L tokens correct scales like (1 − ε)^L, which looks sharp; token edit distance scales roughly linearly; Multiple Choice Grade is a step function; Brier score is continuous [Sch Sec 2 p.3–4, App A p.12–13]. Two further factors: low resolution in the small-model regime (resolution is limited by the number of test items and target length) and sparse sampling of large models [Sch p.4].

**Schaeffer's tests, and their scope.**
1. InstructGPT/GPT-3 (four available sizes: 350M, 1.3B, 6.7B, 175B) on 2-digit multiplication and 4-digit addition. Holding outputs fixed and switching accuracy to token edit distance removes the sharpness (Fig 3); generating more test items shows above-chance accuracy at all sizes (Fig 4). [Sch p.4–6, footnote 3]
2. BIG-Bench meta-analysis: of 39 preferred metrics at most 5 show any emergent task-model pair, and two metrics, Multiple Choice Grade and Exact String Match, account for more than 92% of hand-annotated emergent abilities (Fig 5). For LaMDA tasks that look emergent under Multiple Choice Grade, Brier score removes the effect (Fig 6). [Sch p.6–7]
3. Induced emergence in autoencoders and transformers on vision data by choosing a sharp metric (Figs 7, 8, 10). [Sch p.7–8]

**Where this leaves the dispute (limits on both sides).**
- Schaeffer's own caveats: the independence assumption in the derivation "is not true" (footnote 1, p.4); the LLM tests cover one model family on two arithmetic tasks plus LaMDA via BIG-Bench, because other families' outputs are not publicly queryable [Sch p.6]; and "nothing in this paper should be interpreted as claiming that large language models cannot display emergent abilities", only that previously claimed ones "might likely be a mirage induced by researcher analyses" [Sch p.9]. The vision experiments show metrics *can* induce emergence, not that they *did* here.
- Tension that neither paper resolves in the pages I read [Inference]: Wei's App A.2 check used BLEU, ROUGE and BLEURT and still saw sharp curves; Schaeffer's App A.4 reports ROUGE-L-Sum behaves as a sharp metric in simulation [Sch p.13, Fig 9], and Schaeffer's linear-metric test (token edit distance) was run on GPT-family arithmetic rather than on Wei's three generative tasks. The partial-credit metrics Wei tried are not the same thing as a metric that is linear or continuous in per-token error.
- Schaeffer's multiple-comparisons point is worth keeping: BIG-Bench has at least 220 tasks, about 40 metrics per task and about 10 model families, roughly 10^6 task-metric-model triplets, so "the probability that *no* ... triplet exhibits an emergent ability by random chance might be small" [Sch p.9]. Wei's App A.3 and App E do classify all 210 BIG-Bench tasks then available as emergent, smoothly increasing or flat [Wei p.22–23, p.29–30], so task-level denominators exist. [Inference] What the text I read does not give is a count at the task-metric-model level, which is the unit Schaeffer's multiple-comparisons argument concerns, and the classification itself rests on a subjective "near-random until a certain scale" judgment (p.22).

## 4. Prompt, few-shot, decoding and seed limitations (only where the texts speak)

- **Prompting / shot count.** Fig 2 A–D are 2-shot BIG-Bench; the other panels cite the source papers for their setups [Wei Fig 2 caption, p.4]. The survey does not vary prompts within a task. The WiC example is a case where results turned on setup: GPT-3 achieved slightly above-random few-shot performance on the dev set (~55%) that "did not appear to be a result of scale and did not hold on the test set server" [Wei footnote 3, p.6].
- **Decoding.** Wei reports both greedy (T=0) and random sampling (T=1) error rates for the three generative tasks in the cross-entropy analysis [Wei Fig 5, p.20]. Schaeffer's Figs 3 and 4 plot temperature 0.0 and 1.0 [Sch p.5]. Neither main-text emergence figure (Wei Fig 2, 3) shows decoding variation.
- **Seeds / uncertainty.** In the text I read, Wei Figs 2–4 show one point per model with no intervals or seed-to-seed variation, and some tasks are tiny: "Logical arguments only has 32 samples, which may contribute to noise" [Wei Fig 6 caption, p.21]. Scale is sampled sparsely: Table 2 lists four sizes each for Gopher and Chinchilla and three for PaLM [Wei p.26], and Schaeffer's Fig 10 uses sparse sampling of model sizes to induce apparent emergence [Sch p.14]. [Inference] A "threshold" located between two or three points cannot be distinguished from a smooth curve with a steep section.
- **Benchmark design.** TruthfulQA "is adversarially curated against GPT-3 models" [Wei p.3], which bears on how to read the flat GPT-3 line in Fig 2E. [Inference]

## 5. Contamination or leakage as a threat to validity

**In the text I read, Wei does not discuss benchmark contamination or train-test leakage as a threat to the emergence curves.** It does acknowledge related confounds: training-data differences across families ("differences in training data composition", p.8; "high-quality ... data", p.7) and that the models differ in tokens seen (Table 2, p.26: e.g., GPT-3 300B, LaMDA 132 to 674B, Chinchilla 1.34T, PaLM 780B).

[Inference, not a finding of either paper] Contamination is a plausible threat because the evaluation sets (MMLU, TruthfulQA, BIG-Bench tasks) are public, the models differ in training corpus and size of corpus, and larger models memorise more (the paper itself cites memorisation rising with scale, p.9). Contamination of a public benchmark would raise scores for larger or longer-trained models and could mimic a step. Whether the cited source papers ran contamination checks is not answerable from these two PDFs. The same caution applies to Schaeffer's tests, which analyse model outputs and do not address contamination.

This project's own contamination test is a small-scale illustration of what can and cannot be shown: an n-gram sweep found zero real matches at n = 13 and n = 20, which does not prove a set is uncontaminated, because it misses paraphrase, structural reuse and translation (see `results/contamination_review.md`).

## 6. What evidence would be needed before transferring the claim to the Meridian appeals pilot

Neither paper tests a fine-tuned small model, a structured-output document task, or any property the pilot cares about (escalation, injection robustness, fairness across member attributes). Wei's own survey base is few-shot and prompted use of very large pretrained models on BIG-Bench, MMLU and similar benchmarks [Wei Sec 3–4]. Before using either "abilities will emerge with scale" or "emergence is a mirage" as a risk argument, the pilot would need:

1. **The operational metric, measured directly on the deployed model.** The harm-relevant outcome for an appeal is conjunctive (every required fact present, every contradiction caught). [Inference] Schaeffer's point cuts both ways here: even if underlying quality improves smoothly, an all-or-nothing operational metric can still look like a step, and a decision-maker is exposed to the step. So the pilot should report the conjunctive metric and a graded companion (per-fact recall, probability or Brier score of the recommended step) side by side, as this audit did for rationale omission and recall.
2. **A dense, same-recipe model-size series for the target task** if any scaling argument is to be made: several sizes of one family under one adaptation recipe, with enough test items for the resolution Schaeffer describes, and intervals instead of single points. This audit has one model size and one adapter and therefore **cannot test emergence at all**.
3. **Variation controls:** prompt wording, decoding (greedy and sampled), sampling seeds and training seeds, each reported with intervals. This audit varied sampling seeds only (generation variability) and found 33.2% of prompts received different recommendations across three seeds.
4. **Contamination controls beyond n-grams:** provenance for real evaluation cases, plus paraphrase-level and template-level overlap checks and a record of the base model's training data composition where available.
5. **Direct safety evidence, not capability proxies.** Wei itself notes that risks such as bias "can increase with scaling" in some benchmarks [Wei p.8–9], so a larger model is not safer by default, and the absence of an emergent jump does not clear a risk. In this audit the failures (non-escalation of contradicted or urgent cases, injection compliance, misattributed evidence) appear in a small model and nothing in either paper supports assuming scale would fix them.

## 7. Bottom line for the briefing

- Wei et al. document a real pattern in downstream scores: near-random, then above-random, across model sizes. They also show loss improving smoothly underneath, and they say the threshold depends on more than scale.
- Schaeffer et al. show convincingly that the *choice of metric* can create or remove the apparent step, and that most claimed emergent abilities use two discontinuous metrics. Their scope is limited (one family on arithmetic, LaMDA on BIG-Bench, an approximate model) and they explicitly do not claim emergence is impossible.
- For the Meridian pilot, the practical lesson is not which paper wins. It is that capability curves from other tasks say little about this task; the relevant metric is the operational one; and the pilot has to be justified by measurements on the actual model, with intervals, not by an argument about scaling.
