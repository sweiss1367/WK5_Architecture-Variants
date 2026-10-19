"""Step-budget calibration from TRAINING LOSS ONLY (PREREGISTRATION.md Section 7).

    python src/calibrate.py            # reads results/runs/calibration/*.json, writes results/calibration_decision.json

Run files are loaded through a whitelist (LOADED_KEYS). Validation and test metrics never enter this module, so they
cannot influence the step budget. The rule uses the mean training loss over the last 100 steps (final_train_loss_100).

Stage 1: arm A, seed 900, every grid value S. S* is the smallest S whose next-larger grid value improves final training
         loss by less than `threshold` (relative). If none qualifies, the expansion trigger applies (a human decision).
Stage 2: arms B and C at S* and the next grid value, same rule per arm (arm A already has its values). If any arm fails,
         S* moves up one grid value for all arms and the check is repeated, until a value passes or the grid is exhausted.
`next_runs` lists the runs still needed, so the driver only runs what the rule requires.
"""
import argparse
import glob
import json
import os
from datetime import datetime, timezone

LOADED_KEYS = ("arm", "seed", "steps", "stage", "status", "attempt", "final_train_loss_100", "train_seconds")
GRID = [1000, 2000, 4000]
THRESHOLD = 0.02
CAL_SEED = 900


def load_runs(run_dir, seed=CAL_SEED):
    """{(arm, steps): {"loss", "train_seconds"}} for completed calibration runs. Only whitelisted keys are ever read."""
    out = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        raw = json.load(open(f))
        r = {k: raw.get(k) for k in LOADED_KEYS}
        del raw
        if r["status"] == "completed" and r["stage"] == "calibration" and r["seed"] == seed:
            out[(r["arm"], r["steps"])] = {"loss": r["final_train_loss_100"], "train_seconds": r["train_seconds"]}
    return out


def load_training_losses(run_dir, seed=CAL_SEED):
    return {k: v["loss"] for k, v in load_runs(run_dir, seed).items()}


def improvement(loss, arm, s, s_next):
    a, b = loss.get((arm, s)), loss.get((arm, s_next))
    return None if a is None or b is None else (a - b) / a


def stage1(loss, grid=GRID, thr=THRESHOLD):
    """Return (S*, details, status). S* is None if no grid value qualifies or runs are missing."""
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


def next_runs(loss, grid=GRID, thr=THRESHOLD):
    """Runs still needed, as [(arm, S)]. Empty when the decision can be made, or when a human decision is required."""
    miss = [("A", s) for s in grid if ("A", s) not in loss]
    if miss:
        return miss
    s1, _, _ = stage1(loss, grid, thr)
    if s1 is None:
        return []
    for s in grid[grid.index(s1):-1]:
        nxt = grid[grid.index(s) + 1]
        miss = [(a, x) for a in ("B", "C") for x in (s, nxt) if (a, x) not in loss]
        if miss:
            return miss
        if stage2(loss, s, grid, thr)[1] == "ok":
            return []
    return []


def decide(loss, grid=GRID, thr=THRESHOLD):
    s1, d1, st1 = stage1(loss, grid, thr)
    result = {"validation_used": False, "grid": grid, "threshold": thr,
              "stage1": {"status": st1, "S_star": s1, "details": d1}}
    if s1 is None:
        result["status"] = st1
        result["next_action"] = ("expansion trigger: add S=8000 for arm A only if the timing projection allows, else use the "
                                 "largest grid value and record non-convergence" if st1 == "no_grid_value_qualifies"
                                 else "runs are missing")
        return result
    trail = []
    for s in grid[grid.index(s1):-1]:
        per, st2 = stage2(loss, s, grid, thr)
        trail.append({"S": s, "status": st2, "per_arm": per})
        if st2 == "ok":
            result.update(status="ok", recommended_S=s)
            break
        if st2 == "missing_runs":
            result["status"] = "missing_runs"
            break
    else:
        result["status"] = "grid_exhausted"
        result["next_action"] = "no grid value passes on all three arms: use the largest grid value and record non-convergence as a limitation"
    result["stage2"] = trail
    if result.get("status") == "ok" and result["recommended_S"] != s1:
        result["note"] = f"stage 1 gave S*={s1} but the other arms had not plateaued there, so S moved up to {result['recommended_S']} for all arms"
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs/calibration")
    ap.add_argument("--out", default="results/calibration_decision.json")
    ap.add_argument("--grid", nargs="+", type=int, default=GRID)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    args = ap.parse_args()
    runs = load_runs(args.runs)
    loss = {k: v["loss"] for k, v in runs.items()}
    result = decide(loss, args.grid, args.threshold)
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    result["final_train_loss_100"] = {f"{a}@{s}": v["loss"] for (a, s), v in sorted(runs.items())}
    result["train_seconds"] = {f"{a}@{s}": v["train_seconds"] for (a, s), v in sorted(runs.items())}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)

    print(f"{'arm':<4}{'steps':>7}{'final train loss':>18}{'train s':>10}{'steps/s':>10}")
    for (a, s), v in sorted(runs.items()):
        print(f"{a:<4}{s:>7}{v['loss']:>18.5f}{v['train_seconds']:>10.1f}{s / v['train_seconds']:>10.1f}")
    print("\nstage 1 (arm A):")
    for d in result["stage1"]["details"]:
        imp = d["relative_improvement"]
        print(f"  S={d['S']} -> {d['next_S']}: improvement {'n/a' if imp is None else f'{imp * 100:.2f}%'}  qualifies={d['qualifies']}")
    for blk in result.get("stage2", []):
        print(f"stage 2 at S={blk['S']}: {blk['status']}")
        for a, v in blk["per_arm"].items():
            imp = v["relative_improvement"]
            print(f"  arm {a}: {v['S']} -> {v['next_S']}: improvement {'n/a' if imp is None else f'{imp * 100:.2f}%'}  passes={v['passes']}")
    print("\nSTATUS:", result["status"], "| recommended S:", result.get("recommended_S"), "| validation used: False")
    for k in ("note", "next_action"):
        if k in result:
            print(k.upper() + ":", result[k])


if __name__ == "__main__":
    main()
