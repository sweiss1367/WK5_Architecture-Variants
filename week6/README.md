# Week 6: Red team the claim, audit the model

Audit of one QLoRA-adapted `Qwen/Qwen2.5-0.5B-Instruct` that drafts structured appeal summaries from
synthetic denial notices and clinical summaries, framed as a health insurer's appeals-drafting pilot
("Meridian"). All data are synthetic: invented members, random member IDs, a closed fictional `MRP-####`
policy library. No clinical validity is claimed.

**Status: GPU stages run by the user and imported (evidence ZIP); executed notebook with saved outputs is notebooks/week6_audit_executed.ipynb. Risk register built from the real results. Still pending: the Wei et al. frontier-claim critique (the papers were not reachable from the build sandbox) and the speaker notes. `results/posthoc_*` are exploratory and not preregistered; the preregistered rules and frozen files are unchanged.**
Do not read anything in `results/` other than the contamination files and the figure of them as a model
finding. `analysis_plan.md` (frozen, hashed in `MANIFEST.json`) fixes the metrics and interpretation rules
before any model output exists.

## Layout
```
analysis_plan.md        pre-registered metrics, interpretation rule, limitations (frozen)
MANIFEST.json           SHA-256 of every frozen file; training/generation refuse to run if it fails
PROVENANCE.md           what came from Week 3, what did not
configs/                qlora_appeals.json, decoding.json
data/                   train 400, val 40, eval_h 150, fairness_matched 600, positive_control 1, audit + summary
probes/                 probe_suite.jsonl (100), probe_catalog.csv
src/                    generators, audit, scorers, stats, contamination sweep, GPU scripts
tests/                  58 unit tests; fixtures in tests/fake_fixtures.py are NOT results
results/                contamination_* (produced), everything else produced by the Colab run
logs/                   freeze_history.md; training logs are produced by the Colab run
notebooks/              build_notebook.py -> week6_audit.ipynb (Colab driver, no outputs)
governance/ claim_critique/ video/    reserved; filled after the real runs
```

## Design in one paragraph
One adapter, trained once (seed 42), no DPO, no leak or bias adapters. Train/val use notice templates NT1-4
and summary templates CS1-3 with criterion wording A; the held-out set, fairness set and probes use NT5-6,
CS4-5 and wording B. Fairness pairs differ only in two structured header values (`Primary language`,
`Age`); the notice body, clinical summary and gold labels are byte-identical. Three sampling seeds at
temperature 0.7, top_p 0.9 measure generation variability, not training-seed variability.

## Reproduce the CPU side
```bash
cd week6
python src/generate_appeals_data.py && python src/build_probe_suite.py && python src/audit_dataset.py
python src/manifest.py freeze        # only when intentionally re-freezing; log it in logs/freeze_history.md
python src/contamination_sweep.py && python src/review_contamination.py
cd tests && python -m unittest discover -s . -p "test_*.py"
```
Colab: open `notebooks/week6_audit.ipynb` (T4 runtime) and run top to bottom.

## Limits stated in advance
Single adapter and single training seed; synthetic data and templated language; the language marker tests a
structured attribute, not non-English text; an unresolved fairness result is not evidence of equal
performance; 20 probes per family support counts, not rates; the contamination sweep cannot see paraphrase,
translation, structural, or pretraining contamination.
