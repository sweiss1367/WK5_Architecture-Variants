"""Main-experiment analysis under the FROZEN decision rule (PREREGISTRATION.md Sections 4, 5 and 6).

    python src/analyze_main.py      # reads results/runs/main, writes results/main_summary.json and two CSV files

For comparison arm X (B or C) and seed s:  d_s = val_MAPE(X, s) - val_MAPE(A, s)   (arm A from THIS experiment only; the
pilot's arm A runs are never used here). Lower MAPE is better, so a NEGATIVE d_s means X has the lower error. The
magnitude conditions use the absolute mean difference |d_bar|, and the sign of d_bar says which arm is better.

    Supported, tentative : (a) all three d_s share a sign (zero counts as no sign)
                           (b) |d_bar| >= delta
                           (c) |d_bar| >= k * SD_d      (SD_d uses n-1)
                           (d) every leave-one-out mean has the same sign as d_bar and magnitude >= delta
    Direction only       : (a) and (b) hold, but (c) or (d) does not
    No difference detectable at this budget : anything else

These are exploratory screening criteria designed for a three-seed study. They are NOT significance tests. Three
same-sign differences occur by chance 25% of the time (2 x 0.5^3), and SD_d is itself estimated from three numbers.
delta and k come from configs/base.yaml (frozen, and checked by the freeze record). The verdict is computed only when all
three seeds are complete for every arm of the comparison. Every attempt (completed, failed, interrupted) is listed.
"""
import argparse
import csv
import glob
import json
import os
import statistics as st
import sys
from datetime import datetime, timezone

import yaml

CAVEAT = ("Exploratory screening criteria for a three-seed study, not significance tests. Three same-sign differences occur by "
          "chance 25% of the time, and the standard deviation of three differences is itself a rough estimate. 'Supported, "
          "tentative' is not 'confirmed'.")
KEYS = ("arm", "seed", "attempt", "status", "stage", "segment", "session_id", "policy", "steps", "val_mape", "val_mae_kw",
        "test_mape", "final_train_loss_100", "train_seconds", "train_peak_alloc_mb", "inference_peak_alloc_mb",
        "resident_alloc_before_inference_mb", "inference_latency_ms", "baselines", "params", "config_sha256",
        "final_weights_sha256", "data_sha256", "env", "started_utc", "finished_utc", "error", "eval_blinded")


def sign(x):
    return (x > 0) - (x < 0)


def classify(d, delta, k):
    """d: {seed: matched difference X - A}. Returns the statistics, the four conditions and the verdict label."""
    seeds = sorted(d)
    vals = [d[s] for s in seeds]
    n = len(vals)
    mean = st.mean(vals)
    sd = st.stdev(vals) if n > 1 else None
    loo = {s: st.mean([d[t] for t in seeds if t != s]) for s in seeds} if n > 1 else {}
    ms = sign(mean)
    a = n >= 2 and len({sign(v) for v in vals}) == 1 and 0 not in {sign(v) for v in vals}
    b = abs(mean) >= delta
    c = sd is not None and abs(mean) >= k * sd
    dd = ms != 0 and bool(loo) and all(sign(v) == ms and abs(v) >= delta for v in loo.values())
    if a and b and c and dd:
        label = "supported, tentative"
    elif a and b:
        label = "direction only"
    else:
        label = "no difference detectable at this budget"
    direction = ("the arm has LOWER error than A (an improvement)" if mean < 0
                 else "the arm has HIGHER error than A" if mean > 0 else "no difference")
    return {"d_by_seed": {str(s): d[s] for s in seeds}, "mean": mean, "abs_mean": abs(mean), "sd": sd,
            "leave_one_out_means": {str(s): v for s, v in loo.items()}, "k_times_sd": None if sd is None else k * sd,
            "conditions": {"a_same_sign": a, "b_abs_mean_at_least_delta": b, "c_abs_mean_at_least_k_sd": c,
                           "d_leave_one_out_consistent": dd},
            "verdict": label, "direction": direction}


def load_main(run_dir):
    """Every attempt in the main stage (completed, failed, interrupted markers). Pilot, calibration and extension are never read."""
    rows, markers = [], []
    for f in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        raw = json.load(open(f))
        r = {k: raw.get(k) for k in KEYS}
        if r["stage"] != "main":
            continue
        r["file"] = os.path.basename(f)
        rows.append(r)
    for f in sorted(glob.glob(os.path.join(run_dir, "*.running"))):
        markers.append(os.path.basename(f))
    return rows, markers


def completed_by_cell(rows):
    cell = {}
    for r in rows:
        if r["status"] == "completed":
            key = (r["arm"], r["seed"])
            if key in cell:
                raise ValueError(f"more than one completed attempt for arm {key[0]} seed {key[1]}; resolve before analysing")
            cell[key] = r
    return cell


def spread(vals):
    return {"values": vals, "mean": st.mean(vals), "sd": st.stdev(vals) if len(vals) > 1 else None, "min": min(vals), "max": max(vals)}


def analyse(rows, markers, cfg):
    an, seeds = cfg["analysis"], cfg["study"]["main_seeds"]
    delta, k, ref = an["delta_mape_points"], an["k"], an["reference_arm"]
    cell = completed_by_cell(rows)
    arms = sorted({r["arm"] for r in rows} | {ref})
    out = {"delta": delta, "k": k, "reference_arm": ref, "seeds_required": seeds, "caveat": CAVEAT}

    out["ledger"] = {"completed": sum(r["status"] == "completed" for r in rows), "failed": sum(r["status"] == "failed" for r in rows),
                     "interrupted_markers": markers, "attempts": [{k2: r[k2] for k2 in ("file", "arm", "seed", "attempt", "status", "segment")}
                                                                  | {"error": (r["error"] or "").strip().splitlines()[-1] if r["error"] else None}
                                                                  for r in rows]}
    out["complete"] = all((a, s) in cell for a in ("A", "B", "C") for s in seeds)

    out["per_arm"] = {}
    for a in ("A", "B", "C"):
        got = [cell[(a, s)] for s in seeds if (a, s) in cell]
        if got:
            out["per_arm"][a] = {"seeds": [r["seed"] for r in got], "val_mape": spread([r["val_mape"] for r in got]),
                                 "val_mae_kw": spread([r["val_mae_kw"] for r in got]), "test_mape": spread([r["test_mape"] for r in got]),
                                 "final_train_loss": spread([r["final_train_loss_100"] for r in got]),
                                 "train_seconds": spread([r["train_seconds"] for r in got]),
                                 "train_peak_alloc_mb": spread([r["train_peak_alloc_mb"] for r in got]),
                                 "inference_p50_ms": spread([r["inference_latency_ms"]["p50_ms"] for r in got]),
                                 "inference_p95_ms": spread([r["inference_latency_ms"]["p95_ms"] for r in got]),
                                 "inference_peak_alloc_mb": spread([r["inference_peak_alloc_mb"] for r in got]),
                                 "parameters": got[0]["params"]}

    def comparison(x, y, planned):
        have = [s for s in seeds if (x, s) in cell and (y, s) in cell]
        if len(have) < len(seeds):
            return {"arm": x, "reference": y, "planned": planned, "status": "incomplete", "seeds_available": have}
        d = {s: cell[(x, s)]["val_mape"] - cell[(y, s)]["val_mape"] for s in seeds}
        r = classify(d, delta, k)
        # cost differences, matched by seed (descriptive)
        r["matched_cost_differences"] = {
            "train_seconds": {str(s): cell[(x, s)]["train_seconds"] - cell[(y, s)]["train_seconds"] for s in seeds},
            "inference_p50_ms": {str(s): cell[(x, s)]["inference_latency_ms"]["p50_ms"] - cell[(y, s)]["inference_latency_ms"]["p50_ms"] for s in seeds},
            "train_peak_alloc_mb": {str(s): cell[(x, s)]["train_peak_alloc_mb"] - cell[(y, s)]["train_peak_alloc_mb"] for s in seeds}}
        r.update(arm=x, reference=y, planned=planned, status="computed")
        if not planned:
            r["verdict_note"] = "descriptive only; no verdict is claimed for this comparison"
        return r

    out["comparisons"] = {f"{x}_vs_{y}": comparison(x, y, True) for x, y in an["planned_comparisons"]}
    out["comparisons"].update({f"{x}_vs_{y}": comparison(x, y, False) for x, y in an["descriptive_comparisons"]})

    c = out["comparisons"].get("C_vs_A", {})
    if c.get("status") == "computed":
        conj = c["mean"] > 0
        out["secondary_hypothesis_RoPE_higher_error"] = {
            "d_C_mean": c["mean"], "verdict": c["verdict"],
            "reading": ("consistent in direction with the conjecture" if conj and c["verdict"] != "no difference detectable at this budget"
                        else "opposite to the conjecture" if (not conj) and c["verdict"] != "no difference detectable at this budget"
                        else "not supported at this budget")}
    primary = [v["verdict"] for key, v in out["comparisons"].items() if v.get("planned") and v.get("status") == "computed"]
    out["primary_hypothesis_neutral"] = {"statement": "positional-encoding choice affects validation MAPE beyond seed noise",
                                         "planned_verdicts": primary,
                                         "reading": "no planned comparison separated from seed noise at this budget" if primary and all(v == "no difference detectable at this budget" for v in primary)
                                         else "at least one planned comparison passed screening (exploratory, either direction)" if primary else "incomplete"}

    done = list(cell.values())
    segs, gpus, sess, commits = ({r["segment"] for r in done}, {(r["env"] or {}).get("gpu_uuid") for r in done},
                                 {r["session_id"] for r in done}, {(r["env"] or {}).get("git_commit") for r in done})
    out["session"] = {"segments": sorted(segs), "gpus": sorted(g for g in gpus if g), "session_ids": sorted(sess), "commits": sorted(c2 for c2 in commits if c2),
                      "classification": "uninterrupted (one segment, one GPU)" if len(segs) == 1 and len(gpus) == 1 and not markers and out["ledger"]["failed"] == 0
                      else "resumed or with failed/interrupted attempts: disclose which runs ran where"}
    out["training_config_hashes"] = sorted({r["config_sha256"] for r in done})
    out["data_hashes"] = sorted({r["data_sha256"] for r in done})
    return out


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs/main")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default="results/main_summary.json")
    ap.add_argument("--csv-dir", default="results")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    rows, markers = load_main(args.runs)
    out = analyse(rows, markers, cfg)
    out["created_utc"] = datetime.now(timezone.utc).isoformat()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)

    flat = []
    for r in rows:
        flat.append({"arm": r["arm"], "seed": r["seed"], "attempt": r["attempt"], "status": r["status"], "segment": r["segment"],
                     "val_mape": r["val_mape"], "val_mae_kw": r["val_mae_kw"], "test_mape": r["test_mape"],
                     "final_train_loss": r["final_train_loss_100"], "train_seconds": r["train_seconds"],
                     "train_peak_alloc_mb": r["train_peak_alloc_mb"], "inference_p50_ms": (r["inference_latency_ms"] or {}).get("p50_ms"),
                     "inference_p95_ms": (r["inference_latency_ms"] or {}).get("p95_ms"), "inference_peak_alloc_mb": r["inference_peak_alloc_mb"],
                     "gpu_uuid": (r["env"] or {}).get("gpu_uuid"), "git_commit": (r["env"] or {}).get("git_commit"),
                     "final_weights_sha256": r["final_weights_sha256"], "error": (r["error"] or "").strip().splitlines()[-1] if r["error"] else None})
    write_csv(os.path.join(args.csv_dir, "main_runs.csv"), flat, list(flat[0]) if flat else ["arm"])
    pairs = [{"comparison": name, "seed": s, "difference_val_mape": v} for name, c in out["comparisons"].items()
             if c.get("status") == "computed" for s, v in c["d_by_seed"].items()]
    write_csv(os.path.join(args.csv_dir, "paired_differences.csv"), pairs, ["comparison", "seed", "difference_val_mape"])

    L = out["ledger"]
    print(f"attempts: {L['completed']} completed, {L['failed']} failed, {len(L['interrupted_markers'])} interrupted | all nine complete: {out['complete']}")
    for a, v in out["per_arm"].items():
        m = v["val_mape"]
        print(f"arm {a}: seeds {v['seeds']} val MAPE {[round(x, 5) for x in m['values']]} mean {m['mean']:.5f} SD {m['sd'] if m['sd'] is None else round(m['sd'], 5)}")
    for name, c in out["comparisons"].items():
        if c["status"] != "computed":
            print(f"{name}: INCOMPLETE (seeds available: {c['seeds_available']})")
            continue
        print(f"\n{name}  ({'planned' if c['planned'] else 'descriptive only'})")
        print(f"  matched differences d_s (X - reference; negative = X has lower error): { {s: round(v, 5) for s, v in c['d_by_seed'].items()} }")
        print(f"  mean {c['mean']:+.5f}  |mean| {c['abs_mean']:.5f}  SD_d {c['sd']:.5f}  k*SD_d {c['k_times_sd']:.5f}  delta {delta_text(out)}")
        print(f"  leave-one-out means: { {s: round(v, 5) for s, v in c['leave_one_out_means'].items()} }")
        print(f"  conditions: {c['conditions']}")
        print(f"  VERDICT: {c['verdict']}  | {c['direction']}" + ("" if c["planned"] else "  (descriptive: no verdict is claimed)"))
    print("\n" + out["caveat"])
    print("session:", out["session"]["classification"], "| segments", out["session"]["segments"], "| GPUs", [g[-8:] for g in out["session"]["gpus"]])


def delta_text(out):
    return f"{out['delta']:.2f}, k {out['k']:g}"


if __name__ == "__main__":
    main()
