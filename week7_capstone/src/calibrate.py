"""Step-budget calibration from TRAINING LOSS ONLY (PREREGISTRATION.md Section 7).

    python src/calibrate.py            # reads results/runs/calibration/*.json, writes results/calibration_decision.json

Run files are loaded through a whitelist (LOADED_KEYS). Validation and test metrics never enter this module, so they
cannot influence the step budget. The rule uses the mean training loss over the last 100 steps (final_train_loss_100).

Stage 1: arm A, seed 900, grid S. S* is the smallest S whose next-larger grid value improves final training loss by
         less than `threshold` (relative). If none qualifies, the expansion trigger applies.
Stage 2: arms B and C at S* and the next grid value, same rule per arm. If either fails, S* moves up one grid value
         for all arms and the check is repeated once the new runs exist.
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

LOADED_KEYS = ("arm", "seed", "steps", "stage", "status", "attempt", "final_train_loss_100")
GRID = [1000, 2000, 4000]
THRESHOLD = 0.02
CAL_SEED = 900


def load_training_losses(run_dir, seed=CAL_SEED):
    """{(arm, steps): final training loss} for completed calibration runs. Only whitelisted keys are ever read."""
    out = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        raw = json.load(open(f))
        r = {k: raw.get(k) for k in LOADED_KEYS}
        del raw
        if r["status"] == "completed" and r["stage"] == "calibration" and r["seed"] == seed:
            out[(r["arm"], r["steps"])] = r["final_train_loss_100"]
    return out


def improvement(loss, arm, s, s_next):
    a, b = loss.get((arm, s)), loss.get((arm, s_next))
    return None if a is None or b is None else (a - b) / a


def stage1(loss, grid=GRID, thr=THRESHOLD):
    """Return (S*, details). S* is None if no grid value qualifies or the runs are missing."""
    details = []
    for s, nxt in zip(grid[:-1], grid[1:]):
        imp = improvement(loss, "A", s, nxt)
        details.append({"S": s, "next_S": nxt, "relative_improvement": imp, "qualifies": None if imp is None else imp < thr})
        if imp is None:
            return None, details, "missing_runs"
        if imp < thr:
            return s, details, "ok"
    return None, details, "no_grid_value_qualifies"


def stage2(loss, s_star, grid=GRID, thr=THRESHOLD):
    nxt = grid[grid.index(s_star) + 1]
    per_arm = {}
    for arm in ("A", "B", "C"):
        imp = improvement(loss, arm, s_star, nxt)
        per_arm[arm] = {"S": s_star, "next_S": nxt, "relative_improvement": imp, "passes": None if imp is None else imp < thr}
    if any(v["passes"] is None for v in per_arm.values()):
        return per_arm, "missing_runs"
    return per_arm, "ok" if all(v["passes"] for v in per_arm.values()) else "fails_on_some_arm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs/calibration")
    ap.add_argument("--out", default="results/calibration_decision.json")
    ap.add_argument("--grid", nargs="+", type=int, default=GRID)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    args = ap.parse_args()
    loss = load_training_losses(args.runs)
    s_star, d1, st1 = stage1(loss, args.grid, args.threshold)
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "validation_used": False, "grid": args.grid,
              "threshold": args.threshold, "final_train_loss_100": {f"{a}@{s}": v for (a, s), v in sorted(loss.items())},
              "stage1": {"status": st1, "S_star": s_star, "details": d1}}
    if s_star is not None:
        per_arm, st2 = stage2(loss, s_star, args.grid, args.threshold)
        result["stage2"] = {"status": st2, "per_arm": per_arm}
        if st2 == "fails_on_some_arm":
            i = args.grid.index(s_star)
            result["next_action"] = (f"raise S* to {args.grid[i + 1]} for all arms and repeat the check" if i + 2 < len(args.grid)
                                     else "grid exhausted: use the largest value and record non-convergence as a limitation")
        elif st2 == "ok":
            result["recommended_S"] = s_star
    elif st1 == "no_grid_value_qualifies":
        result["next_action"] = "expansion trigger: add S=8000 for arm A only if the timing projection allows, else use the largest grid value and record non-convergence"
    json.dump(result, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "stage1"}, indent=1))
    print("stage1:", st1, "S* =", s_star)


if __name__ == "__main__":
    main()
