# Freeze history

Every re-freeze of `MANIFEST.json` is listed here with the reason. A re-freeze requires re-running
the contamination sweep (`python src/contamination_sweep.py`) and re-checking the tests.

| # | UTC | What changed | Why |
|---|---|---|---|
| 1 | see MANIFEST.json `created_utc` | Initial freeze of datasets, probe suite, analysis plan, configs, data/scoring code | Before any model output exists |

## Decisions that did NOT change the freeze
* After the first contamination sweep, the owner decided to keep the datasets frozen: policy titles stay in the
  held-out templates and the held-out evaluation set is not modified. The n = 8 overlap (618 matches, 35 unique
  spans, all explained by the intentionally shared synthetic policy vocabulary) is reported as audit evidence and
  classified as expected shared-domain vocabulary, not confirmed leakage. Only the review labels and the report
  wording (not frozen files) were updated. `python src/manifest.py verify` still passes.
* Change rule going forward: a frozen file or preregistered rule changes only if execution reveals a genuine bug.
  Then: stop, document the problem here, state which prior results are affected, re-freeze, and rerun only the
  affected experiment.
