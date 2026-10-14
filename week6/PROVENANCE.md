# Provenance

## Starting point
Week 3 repository `sweiss1367/Week-3-Adapter-Sprint-LoRA-QLoRA-`, read-only clone at commit
`158bc9aba3ff121d2b84fdb340f14844049d322d`. **No Week 3 file was modified.** Nothing is imported across
repositories; the pieces below were copied or re-implemented here.

## What was reused
| Week 6 file | Source | How |
|---|---|---|
| `src/resource_monitor.py` | Week 3 `src/resource_monitor.py` | Copied byte-for-byte (SHA-256 `d6e64e2e...cc248a`, identical to the source). Hardware check, peak-memory and step-time instrumentation. |
| `src/train_qlora.py` | Week 3 `src/train_qlora.py` | Same QLoRA recipe (4-bit NF4, FP16 compute, r=16, alpha=32, dropout 0.05, all attention+MLP projections, prompt-masked labels, label validation, target-module verification). Re-written for the appeals data. OOM demo removed; trainer log history now saved; automatic `max_seq_len` with a no-truncation rule; frozen-manifest check; version-compat shims. |
| `src/generate_batched.py` | Week 3 `src/evaluate.py` (model load/unload pattern) | Re-written: batched, resumable, explicit decoding parameters. |
| `configs/qlora_appeals.json` | Week 3 `configs/qlora_config.json` | LoRA/quantization settings carried over; micro-batch 2 x accumulation 8 (effective 16, same as Week 3). |

## What was not reused
DPO code and data, the 311 generator, system prompt, datasets and logs, the Week 3 evaluation metrics, and
the Week 3 notebook. Week 3 adapter weights were never in the repository (`outputs/` is gitignored) and are
not used; the one adapter audited here is trained in Week 6.

## Week 3 findings are prior work, not Week 6 evidence
Nothing recorded in Week 3 (for example the DPO v1 over-generalisation, or any Week 3 metric) is used as
evidence in any Week 6 table, figure, or risk-register row. If cited at all, it appears only in a clearly
labelled "prior work" note. Every Week 6 number comes from a run recorded in `results/` or `logs/`, bound to
`MANIFEST.json`.

## Known differences from the Week 3 environment
Week 3 recorded `transformers 5.16.1`, `trl 1.14.0`, `peft 0.20.0`, `torch 2.11.0+cu128`,
`bitsandbytes 0.50.2` but never saved a `pip freeze`. Week 6 does not pin versions in advance (the Colab image
changes); `src/preflight.py` records the exact versions used into `results/env_manifest.json` and
`logs/pip_freeze.txt`, which is the reproducibility record. TRL is not used.
