"""Run the nine main runs (3 arms x seeds 0, 1, 2) in one session, then print the frozen-rule analysis.

    python src/run_main.py

Order: seed blocks, with the arm order rotated by seed (seed 0: A,B,C; seed 1: B,C,A; seed 2: C,A,B), so all three arms of a
matched seed run back to back and no arm is always first or last. One fresh process per run. Finished runs are skipped.
A failed run stops the driver (the failure is kept as its own record; re-running starts a new attempt). Interrupted runs leave
a marker and are never deleted.

Single session: if main runs already exist from a different session, the driver refuses to continue unless --allow-resume is
passed, because a resumed experiment must be disclosed as such (PREREGISTRATION.md Section 10, EXPERIMENT_PLAN.md 6.6).
Only seeds 0, 1 and 2 exist here. The optional extension (seeds 3 and 4) is a separate stage that needs the user's approval
file and cannot be started from this script. Unknown arguments go to run_ablation.py (tests and debugging only).
"""
import argparse
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ARMS = ("A", "B", "C")
SEEDS = (0, 1, 2)


def arm_order(seed_index):
    k = seed_index % len(ARMS)
    return list(ARMS[k:] + ARMS[:k])


def plan(seeds=SEEDS):
    return [(s, a) for i, s in enumerate(seeds) for a in arm_order(i)]


def existing_sessions(run_dir):
    out = set()
    for f in glob.glob(os.path.join(run_dir, "*.json")):
        r = json.load(open(f))
        if r.get("stage") == "main":
            out.add(r.get("session_id"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--manifest", default="results/session_manifest.json")
    ap.add_argument("--freeze-record", default="results/freeze_record.json")
    ap.add_argument("--allow-resume", action="store_true", help="continue a main experiment started in another session (must be disclosed)")
    ap.add_argument("--allow-cpu", action="store_true", help="debug only")
    ap.add_argument("--no-analysis", action="store_true", help="debug only")
    args, extra = ap.parse_known_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    if not set(args.seeds) <= set(SEEDS):
        sys.exit(f"only the main seeds {SEEDS} can be run here, got {args.seeds}. The optional extension is a separate, approval-gated stage.")

    from run_ablation import default_session_id
    run_dir = os.path.join(args.out, "main")
    prior = existing_sessions(run_dir) - {None}
    here = os.environ.get("ABLATION_SESSION_ID") or default_session_id()
    if prior and prior != {here} and not args.allow_resume:
        sys.exit(f"main runs already exist from another session {sorted(prior)} (this session is {here}).\n"
                 "Continuing would make this a RESUMED experiment, which must be disclosed. Re-run with --allow-resume only if that is intended.")
    if prior and prior != {here}:
        print(f"NOTICE: resuming the main experiment from session(s) {sorted(prior)} in session {here}. This will be reported as a resumed experiment.", flush=True)

    ch = subprocess.run([sys.executable, "src/freeze.py", "--verify", "--record", args.freeze_record], capture_output=True, text=True)
    print(ch.stdout.strip() or ch.stderr.strip(), flush=True)
    if ch.returncode != 0:
        sys.exit("the freeze check failed, so no main run is started. Nothing was run.")

    steps = plan(args.seeds)
    print(f"plan ({len(steps)} runs, seed blocks, rotated arm order): {steps}", flush=True)
    for seed, arm in steps:
        cmd = [sys.executable, "src/run_ablation.py", "--config", f"configs/arm_{arm}.yaml", "--base", args.base, "--seed", str(seed),
               "--stage", "main", "--out", args.out, "--manifest", args.manifest, "--freeze-record", args.freeze_record, *(["--allow-cpu"] if args.allow_cpu else []), *extra]
        p = subprocess.run(cmd, capture_output=True, text=True)
        print(p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr.strip()[-600:], flush=True)
        if p.returncode not in (0, 3):
            sys.exit(f"run failed (arm {arm}, seed {seed}); it is recorded as a failed attempt.\n{p.stderr[-2000:]}")
    if not args.no_analysis:
        print(flush=True)
        subprocess.run([sys.executable, "src/analyze_main.py", "--runs", run_dir, "--config", args.base], check=True)


if __name__ == "__main__":
    main()
