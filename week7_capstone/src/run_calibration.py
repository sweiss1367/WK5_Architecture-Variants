"""Run the calibration runs that the training-loss rule still needs, then print the decision.

    python src/run_calibration.py

Runs only what calibrate.next_runs asks for (arm A at the grid first, then B and C at S* and the next value, moving S*
up if an arm has not plateaued), one fresh process per run, stage 'calibration', seed 900. It stops when no run is
needed, and also when a human decision is required (no grid value qualifies). It never runs more than --max-runs. Unknown arguments (for example --lock) are passed through to run_ablation.py.
It does not run the noise pilot or any main run. Validation of arms B and C is never computed in this stage.
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrate as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs/calibration")
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--calibration-decision", default="results/calibration_decision.json")
    ap.add_argument("--max-runs", type=int, default=12)
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--manifest", default="results/session_manifest.json")
    ap.add_argument("--allow-cpu", action="store_true", help="debug only")
    ap.add_argument("--grid", nargs="+", type=int, default=C.GRID)
    ap.add_argument("--threshold", type=float, default=C.THRESHOLD)
    args, extra = ap.parse_known_args()          # unknown arguments are passed to run_ablation.py (debug and tests only)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    done = 0
    while True:
        loss = C.load_training_losses(args.runs)
        plan = C.next_runs(loss, args.grid, args.threshold)
        if not plan:
            break
        for arm, steps in plan:
            if done >= args.max_runs:
                sys.exit(f"stopped: reached --max-runs {args.max_runs} with runs still needed")
            cmd = [sys.executable, "src/run_ablation.py", "--config", f"configs/arm_{arm}.yaml", "--base", args.base,
                   "--seed", str(C.CAL_SEED), "--stage", "calibration", "--steps", str(steps), "--out", args.out,
                   "--manifest", args.manifest, *(["--allow-cpu"] if args.allow_cpu else []), *extra]
            t0 = time.time()
            p = subprocess.run(cmd, capture_output=True, text=True)
            print(p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr.strip()[-500:], flush=True)
            if p.returncode not in (0, 3):
                sys.exit(f"run failed ({arm}, S={steps}):\n{p.stderr[-2000:]}")
            done += 1
        print(f"-- {done} runs so far; checking whether the rule needs more --", flush=True)
    print(f"-- no further runs needed ({done} run this invocation) --\n", flush=True)
    sys.stdout.flush()
    subprocess.run([sys.executable, "src/calibrate.py", "--runs", args.runs, "--out", args.calibration_decision,
                    "--grid", *map(str, args.grid), "--threshold", str(args.threshold)], check=True)


if __name__ == "__main__":
    main()
