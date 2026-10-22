"""Noise-pilot summary: seed-to-seed variation of the REFERENCE ARM (A) only. No arms are compared.

    python src/analyze_pilot.py     # writes results/pilot_summary.json and prints the table

Seeds 901 and 902 come from the pilot stage. Seed 900 at the same step count is reused from calibration, which is valid
only because its merged-config hash equals the pilot runs' (checked here). Run files are read through a whitelist, only
arm A is accepted, and test-set metrics are never read, so nothing here can compare positional encodings or touch the
test set. delta and k are NOT set here (PREREGISTRATION.md Section 6): this script only reports scale information and
the noise estimate.

With three seeds the standard deviation is a rough estimate. Its 95% interval is wide, and it is reported next to the
estimate. It informs interpretation. It is not a guarantee of statistical significance.
"""
import argparse
import glob
import json
import math
import os
import statistics as st
from datetime import datetime, timezone

LOADED = ("arm", "seed", "steps", "stage", "status", "attempt", "val_mape", "val_mae_kw", "baselines", "final_train_loss_100",
          "train_seconds", "train_steps_per_second", "train_peak_alloc_mb", "inference_latency_ms", "config_sha256",
          "final_weights_sha256", "segment", "policy", "env", "eval_blinded", "data_sha256")
BASELINE_KEYS = ("seasonal_naive_day_val_mape", "same_hour_last_week_val_mape")


def sd_interval(s, n, level=0.95):
    """Interval for a standard deviation from n observations. For n=3 (2 degrees of freedom) the chi-square quantile has
    a closed form: x = -2 ln(1-p), so no statistics library is needed."""
    if n != 3:
        raise ValueError("closed form implemented for n=3 only")
    df, a = 2, (1 - level) / 2
    lo_q, hi_q = -2 * math.log(1 - a), -2 * math.log(a)          # chi-square quantiles at a and 1-a
    return s * math.sqrt(df / hi_q), s * math.sqrt(df / lo_q)


def num(x, spec):
    """Format a number, or n/a for a field a CPU debug run does not have (peak memory, GPU id)."""
    return "n/a" if x is None else format(x, spec)


def tail(x, n=8):
    return "n/a" if x is None else x[-n:]


def load(path):
    raw = json.load(open(path))
    r = {k: raw.get(k) for k in LOADED}
    del raw
    if r["arm"] != "A":
        raise ValueError(f"{path}: the noise pilot analysis accepts arm A only, got {r['arm']}")
    return r


def collect(pilot_dir, cal_dir, steps, reuse_seed=900):
    runs = []
    for f in sorted(glob.glob(os.path.join(pilot_dir, "A_seed*_attempt*.json"))):
        r = load(f)
        if r["status"] == "completed" and r["stage"] == "pilot" and r["steps"] == steps:
            r["source"] = "pilot"
            runs.append(r)
    f = os.path.join(cal_dir, f"A_seed{reuse_seed}_S{steps}_attempt1.json")
    if os.path.exists(f):
        r = load(f)
        if r["status"] == "completed" and r["steps"] == steps:
            r["source"] = "calibration (reused)"
            runs.append(r)
    return sorted(runs, key=lambda r: r["seed"])


def summarize(runs):
    if len({r["config_sha256"] for r in runs}) != 1:
        raise ValueError("the runs do not share one merged-config hash, so seed 900 cannot be reused as a pilot seed")
    v = [r["val_mape"] for r in runs]
    n = len(v)
    out = {"n_seeds": n, "seeds": [r["seed"] for r in runs], "val_mape": v, "mean": st.mean(v),
           "sd": st.stdev(v) if n > 1 else None, "min": min(v), "max": max(v), "range": max(v) - min(v)}
    if n == 3:
        out["sd_95_interval"] = sd_interval(out["sd"], 3)
    base = runs[0]["baselines"]
    out["baselines"] = {k: base[k] for k in BASELINE_KEYS}
    out["gain_vs_best_baseline_points"] = min(base[k] for k in BASELINE_KEYS) - out["mean"]
    out["gain_vs_best_baseline_percent"] = out["gain_vs_best_baseline_points"] / min(base[k] for k in BASELINE_KEYS) * 100
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-runs", default="results/runs/pilot")
    ap.add_argument("--calibration-runs", default="results/runs/calibration")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--out", default="results/pilot_summary.json")
    ap.add_argument("--k", type=float, default=2.0, help="provisional k from the pre-registration, used only to show scale")
    args = ap.parse_args()
    runs = collect(args.pilot_runs, args.calibration_runs, args.steps)
    if len(runs) < 2:
        raise SystemExit(f"need at least 2 completed arm A runs at {args.steps} steps, found {len(runs)}")
    s = summarize(runs)

    print(f"{'seed':>5} {'source':<22}{'val MAPE %':>11}{'val MAE kW':>12}{'train loss':>11}{'train s':>9}{'peak MiB':>10}{'infer p50':>10}  gpu / segment / commit")
    for r in runs:
        print(f"{r['seed']:>5} {r['source']:<22}{r['val_mape']:>11.4f}{r['val_mae_kw']:>12.3f}{r['final_train_loss_100']:>11.5f}"
              f"{r['train_seconds']:>9.1f}{num(r['train_peak_alloc_mb'], '>10.1f'):>10}{r['inference_latency_ms']['p50_ms']:>10.2f}"
              f"  ...{tail(r['env'].get('gpu_uuid'))} / {r['segment']} / {(r['env'].get('git_commit') or 'n/a')[:8]}")
    print(f"\narm A validation MAPE over {s['n_seeds']} seeds: mean {s['mean']:.4f}, SD (n-1) {s['sd']:.4f}, min {s['min']:.4f}, "
          f"max {s['max']:.4f}, range {s['range']:.4f}")
    if "sd_95_interval" in s:
        lo, hi = s["sd_95_interval"]
        print(f"95% interval for the true SD from 3 seeds: {lo:.4f} to {hi:.4f}  (the estimate is rough: about x0.52 to x6.3)")
    print(f"seasonal baselines (validation MAPE): previous day {s['baselines']['seasonal_naive_day_val_mape']:.3f}, "
          f"same hour last week {s['baselines']['same_hour_last_week_val_mape']:.3f}; arm A beats the better one by "
          f"{s['gain_vs_best_baseline_points']:.3f} points ({s['gain_vs_best_baseline_percent']:.1f}%)")
    if s["sd"] is not None:
        s["scale_for_k_times_sqrt2_sd"] = args.k * math.sqrt(2) * s["sd"]
        print(f"for scale only (delta is NOT set here): k x sqrt(2) x SD_A with provisional k={args.k:g} = {s['scale_for_k_times_sqrt2_sd']:.4f} points")
    sess = sorted({(r["segment"], tail(r["env"].get("gpu_uuid"))) for r in runs})
    print(f"sessions (segment, gpu): {sess}  -> seed 900 is from a different session and GPU than the pilot seeds" if len(sess) > 1 else f"sessions: {sess}")
    s["runs"] = [{k: r[k] for k in ("seed", "source", "segment", "steps", "policy", "val_mape", "final_train_loss_100", "train_seconds",
                                    "train_peak_alloc_mb", "config_sha256", "final_weights_sha256", "data_sha256")}
                 | {"gpu_uuid": r["env"].get("gpu_uuid"), "git_commit": r["env"].get("git_commit"), "infer_p50_ms": r["inference_latency_ms"]["p50_ms"]}
                 for r in runs]
    s.update(created_utc=datetime.now(timezone.utc).isoformat(), arms_compared=False, test_metrics_read=False, delta_set=False)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(s, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
