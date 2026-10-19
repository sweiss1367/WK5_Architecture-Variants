"""Smoke-stage checks that need a real run: V4 determinism, V5 shared initialization, V6 embedding scale,
V7 demand floor, V16 inference-memory cleanliness, plus the unit tests.

    python src/smoke_checks.py

Run this AFTER choose_policy.py. The determinism check compares validation MAPE between repeated runs only to see
whether they are bitwise identical. No accuracy value is used for any decision, and the policy was already selected
from timing alone.
Writes results/smoke/checks/smoke_checks.json. Exit code 1 if a required check fails.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_data import generate_or_load  # noqa: E402


def run_one(root, arm, policy, steps, out, manifest, extra):
    cmd = [sys.executable, os.path.join(root, "src", "run_ablation.py"), "--config", f"configs/arm_{arm}.yaml",
           "--seed", "0", "--stage", "smoke", "--steps", str(steps), "--policy", policy, "--out", out,
           "--manifest", manifest, "--session-id", "smoke-checks", *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
    if p.returncode != 0:
        raise RuntimeError(f"run failed ({arm}, {policy}):\n{p.stderr[-2000:]}")
    return json.load(open([l for l in p.stdout.splitlines() if "->" in l][-1].split("-> ")[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/smoke/checks")
    ap.add_argument("--manifest", default="results/session_manifest.json")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--n-series", type=int, help="debug only")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true", help="debug only")
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    import torch
    on_gpu = torch.cuda.is_available()
    if not on_gpu and not args.allow_cpu:
        sys.exit("No GPU found. Run the smoke checks on the Colab GPU (--allow-cpu only to debug the script).")
    os.makedirs(args.out, exist_ok=True)
    extra = ["--n-series", str(args.n_series)] if args.n_series else []

    runs = {}
    for pol in ("P1", "P2"):
        runs[f"B_{pol}_1"] = run_one(root, "B", pol, args.steps, args.out, args.manifest, extra)
        runs[f"B_{pol}_2"] = run_one(root, "B", pol, args.steps, args.out, args.manifest, extra)
    for arm in ("A", "C"):
        runs[f"{arm}_P1_1"] = run_one(root, arm, "P1", args.steps, args.out, args.manifest, extra)

    checks = []
    add = lambda cid, status, detail, required=True: checks.append(
        {"id": cid, "status": status, "required": required, "detail": detail})

    for pol, required in (("P1", False), ("P2", False)):
        a, b = runs[f"B_{pol}_1"], runs[f"B_{pol}_2"]
        same_w, same_m = a["final_weights_sha256"] == b["final_weights_sha256"], a["val_mape"] == b["val_mape"]
        status = "PASS" if (same_w and same_m) else ("WARN" if pol == "P1" else "INFO")
        add(f"V4_{pol}_same_seed_rerun", status,
            {"bitwise_identical_weights": same_w, "identical_val_mape": same_m,
             "abs_val_mape_difference": abs(a["val_mape"] - b["val_mape"]),
             "meaning": "P1 is expected to be bitwise identical. A WARN means the determinism limit must be documented as "
                        "part of the noise floor. P2 is informational: it shows how much residual nondeterminism P2 carries."},
            required=False)

    shared = {k: r["shared_init_weights_sha256"] for k, r in runs.items()}
    add("V5_shared_init_identical_across_arms_and_processes", "PASS" if len(set(shared.values())) == 1 else "FAIL",
        {"distinct_hashes": len(set(shared.values())), "runs": list(shared)})

    ratios = {}
    for arm in ("A", "B"):
        r = runs[f"{arm}_P1_1"]
        ratios[arm] = r["positional_term_norm"] / r["value_embedding_norm"]
    add("V6_positional_vs_value_embedding_scale", "PASS" if all(0.1 <= v <= 10 for v in ratios.values()) else "FAIL",
        {"ratio_positional_to_value_norm": ratios, "criterion": "within [0.1, 10], i.e. the same order of magnitude"})

    d = yaml.safe_load(open("configs/base.yaml"))["data"]
    if args.n_series:
        d["n_series"] = args.n_series
    demand, meta, digest = generate_or_load(d, "data")
    floor_ratio = float((demand / np.array(meta["base_kw"])[:, None]).min())
    add("V7_demand_floor_keeps_mape_stable", "PASS" if floor_ratio >= 0.1 else "FAIL",
        {"min_demand_over_base": floor_ratio, "criterion": ">= 0.1", "data_sha256": digest,
         "outlier_hours": meta["outlier_hours"], "gap_hours": meta["gap_hours"]})

    if on_gpu:
        res = {}
        for k, r in runs.items():
            param_mb = r["params"]["total"] * 4 / 2**20
            res[k] = {"resident_mb": r["resident_alloc_before_inference_mb"], "param_mb": param_mb,
                      "infer_peak_mb": r["inference_peak_alloc_mb"]}
        ok = all(v["resident_mb"] <= 1.5 * v["param_mb"] + 8 for v in res.values())
        add("V16_inference_memory_excludes_training_state", "PASS" if ok else "FAIL",
            {"per_run": res, "criterion": "resident allocated memory <= 1.5 x parameter bytes + 8 MB"})
    else:
        add("V16_inference_memory_excludes_training_state", "SKIP", "no GPU", required=False)

    add("leakage_checks_inside_every_run", "PASS", "check_no_leakage runs inside every run_ablation.py run; all runs completed")

    if args.skip_tests:
        add("unit_tests", "SKIP", "--skip-tests")
    else:
        p = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-x"], capture_output=True, text=True, cwd=root)
        add("unit_tests", "PASS" if p.returncode == 0 else "FAIL", p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-500:])

    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "steps": args.steps, "gpu": on_gpu,
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "gpu_name": torch.cuda.get_device_name(0) if on_gpu else None, "checks": checks,
              "runs": {k: {"policy": r["policy"], "arm": r["arm"], "file_attempt": r["attempt"], "val_mape_not_used_for_decisions": r["val_mape"],
                           "train_seconds": r["train_seconds"], "final_weights_sha256": r["final_weights_sha256"]} for k, r in runs.items()}}
    json.dump(report, open(os.path.join(args.out, "smoke_checks.json"), "w"), indent=1)
    print(f"{'check':<58}{'status':<7}")
    for c in checks:
        print(f"{c['id']:<58}{c['status']:<7}{'' if c['required'] else '(informational)'}")
    if not on_gpu:
        print("WARNING: CPU debug run; GPU-dependent checks are not meaningful.")
    if any(c["required"] and c["status"] == "FAIL" for c in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
