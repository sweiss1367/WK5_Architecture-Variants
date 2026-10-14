"""
Freeze and verify the Week 6 evidence base.

    python src/manifest.py freeze     # write MANIFEST.json (SHA-256 of every frozen file)
    python src/manifest.py verify     # recompute and compare; exit 1 on any mismatch

Frozen = the datasets, the probe suite, the pre-registered analysis plan, the configs, and the
code that defines the data and the scoring rules. Training and generation refuse to start if
`verify` fails, so a silent edit after the freeze cannot reach the evidence.

If a frozen file must change, change it, run `freeze` again, re-run the contamination sweep, and
add a line to logs/freeze_history.md saying what changed and why. Never edit MANIFEST.json by hand.
"""
import datetime
import hashlib
import json
import os
import platform
import sys

FROZEN = [
    "data/train.jsonl", "data/val.jsonl", "data/eval_h.jsonl", "data/fairness_matched.jsonl",
    "data/positive_control.jsonl", "data/dataset_summary.json", "data/dataset_audit.json",
    "probes/probe_suite.jsonl", "probes/probe_catalog.csv",
    "analysis_plan.md", "configs/qlora_appeals.json", "configs/decoding.json",
    "src/appeals_common.py", "src/templates.py", "src/case_builder.py",
    "src/generate_appeals_data.py", "src/build_probe_suite.py", "src/audit_dataset.py",
    "src/score_task.py", "src/score_probes.py", "src/stats.py", "src/contamination_sweep.py",
    "src/analyze_fairness.py", "src/run_task_eval.py", "src/sft_data.py",
]
MANIFEST = "MANIFEST.json"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def freeze():
    base = root()
    files = {}
    for rel in FROZEN:
        p = os.path.join(base, rel)
        if not os.path.exists(p):
            raise SystemExit(f"cannot freeze: {rel} does not exist")
        files[rel] = dict(sha256=sha256_file(p), bytes=os.path.getsize(p))
    man = dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        python=sys.version.split()[0], platform=platform.platform(),
        generator_seed=627,
        note=("positive_control.jsonl is a SYNTHETIC positive control for the contamination detector; "
              "it is hashed here but is not part of any evaluation set."),
        files=files,
    )
    with open(os.path.join(base, MANIFEST), "w") as f:
        json.dump(man, f, indent=2)
    print(f"froze {len(files)} files -> {MANIFEST}  (manifest sha256 {sha256_file(os.path.join(base, MANIFEST))[:16]}...)")
    return man


def verify(quiet=False):
    base = root()
    mp = os.path.join(base, MANIFEST)
    if not os.path.exists(mp):
        raise SystemExit("MANIFEST.json missing: run `python src/manifest.py freeze` first")
    with open(mp) as f:
        man = json.load(f)
    bad = []
    for rel, meta in man["files"].items():
        p = os.path.join(base, rel)
        if not os.path.exists(p) or sha256_file(p) != meta["sha256"]:
            bad.append(rel)
    if bad:
        raise SystemExit(f"MANIFEST MISMATCH, frozen files changed or missing: {bad}")
    if not quiet:
        print(f"manifest verified: {len(man['files'])} files match (manifest sha256 {sha256_file(mp)[:16]}...)")
    return man


def manifest_sha256():
    return sha256_file(os.path.join(root(), MANIFEST))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    {"freeze": freeze, "verify": verify}[cmd]()
