"""Time determinism policies P1 and P2 on identical workloads, then select one by feasibility only.

    python src/choose_policy.py

Each timing run is the real run_ablation.py script in a fresh process (as in the study), in the smoke stage with
--timing-only, so it never computes validation or test accuracy. P1 and P2 get exactly the same command line apart
from --policy: same arms, seed, data, step counts and batch size. The order of the two policies alternates between
repetitions so drift in GPU clocks or temperature is not tied to one policy.

Sequence (matches PREREGISTRATION.md Section 8):
  1. measure and write results/smoke/policy_timing/policy_timing.json
  2. print the measured timing difference
  3. apply the pre-declared rule and write results/determinism_decision.json

Rule: project the total GPU time of the remaining study (RUN_PLAN, each run at the largest candidate step count) from
the measured P1 timings. If it fits within the planning ceiling, select P1, otherwise P2. Accuracy plays no part.
The decision file does not change any config. determinism.policy in configs/base.yaml is set by hand to match it,
and run_ablation.py refuses to run if the two disagree.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

import yaml

RUN_PLAN = {"calibration_stage1": 3, "calibration_stage2": 4, "noise_pilot": 2, "main": 9, "extension": 6,
            "retry_allowance": 3}
S_MAX = 4000        # largest candidate step count in the calibration grid
POLICIES = ("P1", "P2")


def now():
    return datetime.now(timezone.utc).isoformat()


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout.strip() or None
    except Exception:
        return None


def gpu_state():
    q = "name,uuid,driver_version,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu"
    return sh(["nvidia-smi", f"--query-gpu={q}", "--format=csv"])


def run_one(root, arm, steps, policy, out, extra, session_id, manifest):
    cmd = [sys.executable, os.path.join(root, "src", "run_ablation.py"), "--config", f"configs/arm_{arm}.yaml",
           "--seed", "0", "--stage", "smoke", "--steps", str(steps), "--policy", policy, "--timing-only",
           "--out", out, "--manifest", manifest, "--session-id", session_id, *extra]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(f"timing run failed ({arm}, S={steps}, {policy}):\n{p.stderr[-2000:]}")
    stem = [l for l in p.stdout.splitlines() if "->" in l][-1].split("-> ")[-1]
    r = json.load(open(stem))
    assert r["timing_only"] and r["val_mape"] is None, "policy timing must not compute accuracy"
    return {"arm": arm, "steps": steps, "policy": policy, "wall_s": wall, "train_seconds": r["train_seconds"],
            "train_peak_alloc_mb": r["train_peak_alloc_mb"], "infer_p50_ms": r["inference_latency_ms"]["p50_ms"],
            "infer_p95_ms": r["inference_latency_ms"]["p95_ms"], "infer_peak_alloc_mb": r["inference_peak_alloc_mb"],
            "file": stem}


def summarize(rows, arms, steps_list):
    """Per policy and arm: mean over repetitions, then a two-point linear fit over step count."""
    s1, s2 = steps_list[0], steps_list[-1]
    out = {}
    for pol in POLICIES:
        out[pol] = {}
        for arm in arms:
            mean = lambda key, s: statistics.mean(r[key] for r in rows if (r["policy"], r["arm"], r["steps"]) == (pol, arm, s))
            step_time = (mean("train_seconds", s2) - mean("train_seconds", s1)) / (s2 - s1)
            wall_slope = (mean("wall_s", s2) - mean("wall_s", s1)) / (s2 - s1)
            out[pol][arm] = {
                "train_step_ms": step_time * 1000, "wall_per_step_ms": wall_slope * 1000,
                "wall_fixed_s": mean("wall_s", s1) - wall_slope * s1,
                "projected_run_s_at_Smax": mean("wall_s", s2) + (S_MAX - s2) * wall_slope,
                "infer_p50_ms": statistics.mean(mean("infer_p50_ms", s) for s in steps_list),
                "train_peak_alloc_mb": statistics.mean(mean("train_peak_alloc_mb", s) for s in steps_list)
                if rows[0]["train_peak_alloc_mb"] is not None else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--steps", nargs=2, type=int, default=[200, 600], help="two step counts for the linear fit")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--ceiling-hours", type=float, default=3.0)
    ap.add_argument("--out", default="results/smoke/policy_timing")
    ap.add_argument("--decision", default="results/determinism_decision.json")
    ap.add_argument("--manifest", default="results/session_manifest.json")
    ap.add_argument("--n-series", type=int, help="debug only: smaller dataset (changes the workload)")
    ap.add_argument("--allow-cpu", action="store_true", help="debug only: CPU timings are not meaningful for the budget")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    import torch
    if not torch.cuda.is_available() and not args.allow_cpu:
        sys.exit("No GPU found. Policy timing must run on the Colab GPU (use --allow-cpu only to debug the script).")
    os.makedirs(args.out, exist_ok=True)
    extra = ["--n-series", str(args.n_series)] if args.n_series else []
    session_id = f"policy-timing-{int(time.time())}"
    s1, s2 = sorted(args.steps)

    state_before = gpu_state()
    print("warm-up run (discarded) ...", flush=True)
    run_one(root, args.arms[0], 100, "P1", args.out, extra, session_id, args.manifest)

    rows = []
    for rep in range(args.reps):
        order = POLICIES if rep % 2 == 0 else POLICIES[::-1]
        for arm in args.arms:
            for steps in (s1, s2):
                for pol in order:
                    rows.append(run_one(root, arm, steps, pol, args.out, extra, session_id, args.manifest))
                    print(f"  rep {rep} {arm} S={steps} {pol}: train {rows[-1]['train_seconds']:.1f}s wall {rows[-1]['wall_s']:.1f}s", flush=True)
    state_after = gpu_state()
    summ = summarize(rows, args.arms, [s1, s2])

    env = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda, "python": sys.version.split()[0],
           "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "gpu_state_before": state_before, "gpu_state_after": state_after,
           "git_commit": sh(["git", "rev-parse", "HEAD"])}
    cfg = yaml.safe_load(open("configs/base.yaml"))
    conditions = {"arms": args.arms, "seed": 0, "step_counts": [s1, s2], "repetitions": args.reps,
                  "warmup_run_discarded": True, "order": "policy order alternates each repetition",
                  "one_fresh_process_per_run": True, "stage": "smoke", "timing_only": True, "accuracy_computed": False,
                  "batch_size": cfg["train"]["batch_size"], "n_series": args.n_series or cfg["data"]["n_series"],
                  "model": cfg["model"], "latency_protocol": cfg["measure"]["inference_latency"],
                  "identical_workload_for_both_policies": True,
                  "cpu_debug": not torch.cuda.is_available()}
    timing = {"created_utc": now(), "env": env, "conditions": conditions, "runs": rows, "summary": summ}
    json.dump(timing, open(os.path.join(args.out, "policy_timing.json"), "w"), indent=1)

    # 2. report the measured difference BEFORE selecting
    print("\nMEASURED TIMING DIFFERENCE (P1 deterministic vs P2 seeded only)")
    print(f"{'arm':<4}{'P1 step ms':>12}{'P2 step ms':>12}{'penalty':>10}{'P1 infer p50':>14}{'P2 infer p50':>14}"
          f"{'P1 run s@4000':>15}{'P2 run s@4000':>15}")
    for arm in args.arms:
        a, b = summ["P1"][arm], summ["P2"][arm]
        print(f"{arm:<4}{a['train_step_ms']:>12.2f}{b['train_step_ms']:>12.2f}{(a['train_step_ms'] / b['train_step_ms'] - 1) * 100:>9.1f}%"
              f"{a['infer_p50_ms']:>14.3f}{b['infer_p50_ms']:>14.3f}{a['projected_run_s_at_Smax']:>15.1f}{b['projected_run_s_at_Smax']:>15.1f}")

    # 3. pre-declared rule, feasibility only
    n_runs = sum(RUN_PLAN.values())
    worst = {pol: max(summ[pol][a]["projected_run_s_at_Smax"] for a in args.arms) for pol in POLICIES}
    total = {pol: n_runs * worst[pol] for pol in POLICIES}
    ceiling = args.ceiling_hours * 3600
    selected = "P1" if total["P1"] <= ceiling else "P2"
    decision = {
        "created_utc": now(), "selected_policy": selected, "accuracy_used": False,
        "rule": f"Select P1 if the projected total GPU time of the remaining study ({n_runs} runs, each at {S_MAX} steps, "
                f"slowest arm) under P1 is at most the planning ceiling; otherwise select P2. Accuracy is not used.",
        "run_plan": RUN_PLAN, "runs_projected": n_runs, "s_max": S_MAX, "ceiling_hours": args.ceiling_hours,
        "projected_total_hours": {p: total[p] / 3600 for p in POLICIES},
        "slowest_arm_run_seconds_at_Smax": worst,
        "measured_penalty_percent_train_step": {a: (summ["P1"][a]["train_step_ms"] / summ["P2"][a]["train_step_ms"] - 1) * 100
                                                for a in args.arms},
        "feasible_under_selected": total[selected] <= ceiling,
        "env": env, "conditions": conditions, "timing_file": os.path.join(args.out, "policy_timing.json"),
        "note": "Set determinism.policy in configs/base.yaml to selected_policy by hand and commit it. run_ablation.py "
                "refuses to run if the config and this file disagree. cpu_debug=true means these timings are not usable."}
    os.makedirs(os.path.dirname(args.decision), exist_ok=True)
    json.dump(decision, open(args.decision, "w"), indent=1)
    print(f"\nprojected total: P1 {total['P1'] / 3600:.2f} h, P2 {total['P2'] / 3600:.2f} h, ceiling {args.ceiling_hours} h "
          f"({n_runs} runs at {S_MAX} steps)")
    print(f"SELECTED POLICY (feasibility only): {selected}  -> {args.decision}")
    if not decision["feasible_under_selected"]:
        print("WARNING: even the selected policy exceeds the ceiling. Reduce scope before continuing.")
    if conditions["cpu_debug"]:
        print("WARNING: CPU debug run. These numbers say nothing about the Colab budget.")


if __name__ == "__main__":
    main()
