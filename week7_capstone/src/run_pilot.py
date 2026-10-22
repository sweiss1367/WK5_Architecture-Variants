"""Run the noise pilot: arm A, seeds 901 and 902, at the configured step count, then print the seed-level summary.

    python src/run_pilot.py

Only arm A is run, only the pilot seeds below, only in the pilot stage. Finished runs are skipped, so re-running resumes.
Seed 900 at the same step count is reused from calibration (identical merged config, checked by analyze_pilot.py).
It does not touch the main or extension stages. Unknown arguments are passed to run_ablation.py (tests and debugging).
"""
import argparse
import os
import subprocess
import sys

PILOT_SEEDS = (901, 902)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(PILOT_SEEDS))
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--manifest", default="results/session_manifest.json")
    ap.add_argument("--summary", default="results/pilot_summary.json")
    ap.add_argument("--steps", type=int, default=8000, help="step count to summarise (must equal train.steps in the base config)")
    ap.add_argument("--allow-cpu", action="store_true", help="debug only")
    args, extra = ap.parse_known_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    if set(args.seeds) - set(PILOT_SEEDS):
        sys.exit(f"only the pilot seeds {PILOT_SEEDS} are allowed here, got {args.seeds}")

    for seed in args.seeds:
        cmd = [sys.executable, "src/run_ablation.py", "--config", "configs/arm_A.yaml", "--base", args.base, "--seed", str(seed),
               "--stage", "pilot", "--out", args.out, "--manifest", args.manifest, *(["--allow-cpu"] if args.allow_cpu else []), *extra]
        p = subprocess.run(cmd, capture_output=True, text=True)
        print(p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr.strip()[-500:], flush=True)
        if p.returncode not in (0, 3):
            sys.exit(f"run failed (arm A, seed {seed}):\n{p.stderr[-2000:]}")
    print(flush=True)
    subprocess.run([sys.executable, "src/analyze_pilot.py", "--pilot-runs", os.path.join(args.out, "pilot"),
                    "--calibration-runs", os.path.join(args.out, "calibration"), "--steps", str(args.steps), "--out", args.summary],
                   check=True)


if __name__ == "__main__":
    main()
