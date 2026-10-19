"""Single entry point: one (arm, seed) run. Every reported number comes from this script.

    python src/run_ablation.py --config configs/arm_B.yaml --seed 1 --stage main

The arm config is merged over configs/base.yaml. Stages keep results apart:
    smoke        pipeline checks, any seed, --steps / --n-series overrides allowed
    calibration  step-budget calibration, seed 900 only, --steps allowed
    pilot        noise pilot, seeds 900-902
    main         seeds 0-2
    extension    seeds 3-4 (only if declared in PREREGISTRATION.md)
Output: results/runs/<stage>/<arm>_seed<s>_attempt<n>.json (written atomically).
An .running marker next to it is removed on completion. A marker with no result means the attempt was interrupted.

Gates (non-smoke stages): requirements.lock.txt must exist and match the installed packages; tracked files must match
the committed code; results/determinism_decision.json must exist and agree with determinism.policy; main and extension
also require PREREGISTRATION.md to be frozen (no `______` placeholders left).

Exit codes: 0 completed, 1 run failed, 2 invalid arguments, config or gate, 3 already completed (skipped).
"""
import argparse
import contextlib
import copy
import gc
import glob
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

# Must be set before the first CUDA call for deterministic cuBLAS (policy P1).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_env import verify as verify_env  # noqa: E402
from make_data import check_no_leakage, generate_or_load, make_windows, normalize  # noqa: E402
from model import POS_TYPES, TRAIN_SEED_OFFSET, build_model, param_counts, weights_sha256  # noqa: E402

STAGES = ("smoke", "calibration", "pilot", "main", "extension")
STAGE_SEEDS = {"calibration": "calibration_seed", "pilot": "pilot_seeds", "main": "main_seeds", "extension": "extension_seeds"}


def now():
    return datetime.now(timezone.utc).isoformat()


def deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_cfg(base_path, arm_path):
    cfg = deep_merge(yaml.safe_load(open(base_path)), yaml.safe_load(open(arm_path)))
    return cfg


def fail(msg, code=2):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def seed_all(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_determinism(policy):
    """P1: deterministic algorithms + math attention backend. P2: seeded only, default kernels."""
    if policy == "P1":
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)


def attention_context(policy):
    if policy == "P1":
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel(SDPBackend.MATH)
    return contextlib.nullcontext()


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def env_info(device):
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    info = {"host": socket.gethostname(), "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "device_type": device.type,
            "packages_sha256": hashlib.sha256("\n".join(packages).encode()).hexdigest(),
            "git_commit": sh(["git", "rev-parse", "HEAD"]), "git_dirty": bool(sh(["git", "status", "--porcelain", "--untracked-files=no"]))}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        info.update(gpu_name=p.name, gpu_memory_gb=round(p.total_memory / 1e9, 2),
                    gpu_uuid=sh(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"]),
                    driver=sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]))
    return info


def enforce_gates(args, cfg, stage, info):
    """Refuse to start a non-smoke run unless the environment, code, policy and (for main) pre-registration are in order."""
    if stage == "smoke":
        return {}
    env = verify_env(args.lock)
    if not env["lock_present"]:
        fail(f"{args.lock} is missing: run `python src/check_env.py --auto` in the smoke session and commit the lock")
    if not env["matches"] and not args.allow_env_mismatch:
        fail(f"installed packages differ from {args.lock}: {env['mismatches']}")
    if info["git_dirty"] and not args.allow_dirty:
        fail("tracked files differ from the committed code; commit before running a non-smoke stage")
    if not os.path.exists(args.decision):
        fail(f"{args.decision} is missing: run `python src/choose_policy.py` in the smoke session first")
    selected = json.load(open(args.decision)).get("selected_policy")
    if selected != cfg["determinism"]["policy"]:
        fail(f"determinism.policy ({cfg['determinism']['policy']}) does not match the recorded decision ({selected})")
    if stage in ("main", "extension") and "______" in open(args.prereg).read():
        fail("PREREGISTRATION.md is not frozen (placeholders remain); main and extension runs are blocked")
    return {"env_matches_lock": env["matches"], "allow_dirty": args.allow_dirty,
            "allow_env_mismatch": args.allow_env_mismatch, "decision_policy": selected}


def default_session_id():
    try:
        return "boot-" + open("/proc/sys/kernel/random/boot_id").read().strip()
    except OSError:
        return f"host-{socket.gethostname()}"


def ensure_segment(manifest_path, session_id, info, policy):
    """Append a segment record the first time a session id is seen. Returns its index."""
    segs = json.load(open(manifest_path)) if os.path.exists(manifest_path) else []
    for s in segs:
        if s["session_id"] == session_id:
            return s["segment"]
    segs.append({"segment": len(segs), "session_id": session_id, "first_seen_utc": now(), "policy": policy,
                 **{k: info.get(k) for k in ("host", "gpu_name", "gpu_uuid", "driver", "cuda_runtime", "torch",
                                             "packages_sha256", "git_commit", "device_type")}})
    atomic_write(manifest_path, segs)
    return len(segs) - 1


def atomic_write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def mape(pred, y):
    return float((np.abs(pred - y) / y).mean() * 100.0)


def mae(pred, y):
    return float(np.abs(pred - y).mean())


@torch.inference_mode()
def predict(model, xn, scale, bs=1024):
    model.eval()
    outs = [(model(xn[i:i + bs]) * scale[i:i + bs]).double().cpu().numpy() for i in range(0, len(xn), bs)]
    return np.concatenate(outs)


def next_attempt(run_dir, prefix):
    nums = [int(m.group(1)) for f in glob.glob(os.path.join(run_dir, prefix + "_attempt*"))
            if (m := re.search(r"_attempt(\d+)\.", f))]
    return max(nums, default=0) + 1


def has_completed(run_dir, prefix):
    for f in glob.glob(os.path.join(run_dir, prefix + "_attempt*.json")):
        try:
            if json.load(open(f)).get("status") == "completed":
                return True
        except Exception:
            pass
    return False


def latency_benchmark(model, xn_val, lcfg, device, sync):
    rng = np.random.default_rng(lcfg["index_seed"])
    idx = rng.choice(len(xn_val), size=min(lcfg["n_windows"], len(xn_val)), replace=False)
    x = xn_val[torch.as_tensor(idx)].to(device)
    model.eval()
    times = []
    with torch.inference_mode():
        for i in range(lcfg["warmup_calls"]):
            model(x[i % len(x)][None])
        sync()
        for i in range(lcfg["timed_calls"]):
            xi = x[i % len(x)][None]
            sync(); t0 = time.perf_counter()
            model(xi)
            sync(); times.append((time.perf_counter() - t0) * 1000.0)
    t = np.array(times)
    return {"p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)), "mean_ms": float(t.mean()),
            "std_ms": float(t.std(ddof=1)), "n_calls": len(t)}


def run(cfg, args, device):
    d, m, tr = cfg["data"], cfg["model"], cfg["train"]
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    steps, B = tr["steps"], tr["batch_size"]
    if tr["grad_accumulation"] != 1:
        raise ValueError("grad_accumulation must stay 1; a silent fallback would be a confound")
    if steps <= tr["warmup_steps"]:
        raise ValueError(f"steps ({steps}) must exceed warmup_steps ({tr['warmup_steps']})")

    demand, dmeta, data_sha = generate_or_load(d, args.data_dir)
    win = make_windows(demand, d)
    check_no_leakage(win, d)
    norm = {s: normalize(w["x"], w["y"]) for s, w in win.items()}

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t = lambda a: torch.as_tensor(a, dtype=torch.float32)
    xtr, ytr = t(norm["train"][0]).to(device), t(norm["train"][1]).to(device)
    xva_n, sva = t(norm["val"][0]), t(norm["val"][2])
    xte_n, ste = t(norm["test"][0]), t(norm["test"][2])
    xva_d, sva_d, xte_d, ste_d = xva_n.to(device), sva.to(device), xte_n.to(device), ste.to(device)

    model = build_model(m, d["context_hours"], d["horizon_hours"], args.seed).to(device)
    pc = param_counts(model)
    init_sha, shared_init_sha = weights_sha256(model), weights_sha256(model, shared_only=True)
    with torch.no_grad():                       # scale check for V6: value embedding vs positional term
        v_norm = float(model.in_proj(xtr[:256].unsqueeze(-1)).norm(dim=-1).mean())
        p_norm = (float(model.pos_emb.weight.norm(dim=-1).mean()) if m["pos_encoding"] == "learned"
                  else float(model.pe.norm(dim=-1).mean()) if m["pos_encoding"] == "sinusoidal" else None)

    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    wu = tr["warmup_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / wu if s < wu else max(0.0, (steps - s) / (steps - wu)))
    gen = torch.Generator().manual_seed(args.seed)
    n = len(xtr)
    per_epoch = n // B                          # drop_last: every batch is exactly B
    losses = torch.empty(steps, device=device)

    seed_all(args.seed + TRAIN_SEED_OFFSET)     # identical dropout stream across arms for a given seed
    model.train()
    perm, pos = torch.randperm(n, generator=gen), 0
    sync(); t0 = time.perf_counter()
    for step in range(steps):
        if pos == per_epoch:
            perm, pos = torch.randperm(n, generator=gen), 0
        idx = perm[pos * B:(pos + 1) * B].to(device); pos += 1
        loss = F.l1_loss(model(xtr[idx]), ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        losses[step] = loss.detach()
    sync(); train_seconds = time.perf_counter() - t0
    train_peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None

    lc = losses.cpu().numpy()
    curve = [{"step": int(i + len(c)), "mean_train_loss": float(c.mean())}
             for i in range(0, steps, 100) for c in [lc[i:i + 100]]]
    final_loss = float(lc[-min(100, steps):].mean())

    # evaluation (final-step weights, no checkpoint selection). Skipped in timing-only mode, which must not touch accuracy.
    if args.timing_only:
        metrics = {"val_mape": None, "val_mae_kw": None, "test_mape": None, "baselines": None, "timing_only": True}
    else:
        yva, yte = win["val"]["y"].astype(np.float64), win["test"]["y"].astype(np.float64)
        pva, pte = predict(model, xva_d, sva_d), predict(model, xte_d, ste_d)
        metrics = {"val_mape": mape(pva, yva), "val_mae_kw": mae(pva, yva), "test_mape": mape(pte, yte),
                   "baselines": {"seasonal_naive_day_val_mape": mape(win["val"]["x"][:, -24:], yva),
                                 "same_hour_last_week_val_mape": mape(win["val"]["x"][:, :24], yva),
                                 "seasonal_naive_day_test_mape": mape(win["test"]["x"][:, -24:], yte),
                                 "same_hour_last_week_test_mape": mape(win["test"]["x"][:, :24], yte)},
                   "timing_only": False}
    final_sha = weights_sha256(model)

    # inference benchmark: drop training state and the on-GPU datasets first, so they do not inflate inference memory
    del opt, sched, loss, losses, xtr, ytr, xva_d, sva_d, xte_d, ste_d
    model.zero_grad(set_to_none=True)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        resident = torch.cuda.memory_allocated() / 2**20
        torch.cuda.reset_peak_memory_stats()
    lat = latency_benchmark(model, xva_n, cfg["measure"]["inference_latency"], device, sync)
    infer_peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None

    return {
        **metrics, "train_seconds": train_seconds, "train_peak_alloc_mb": train_peak,
        "train_steps_per_second": steps / train_seconds,
        "inference_latency_ms": lat, "inference_peak_alloc_mb": infer_peak,
        "resident_alloc_before_inference_mb": resident if device.type == "cuda" else None,
        "final_train_loss_100": final_loss, "train_loss_curve": curve,
        "params": pc, "init_weights_sha256": init_sha, "shared_init_weights_sha256": shared_init_sha, "final_weights_sha256": final_sha,
        "value_embedding_norm": v_norm, "positional_term_norm": p_norm,
        "n_windows": {k: int(len(v["ts"])) for k, v in win.items()}, "data_sha256": data_sha,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="arm config (merged over --base)")
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stage", choices=STAGES, required=True)
    ap.add_argument("--steps", type=int, help="override train.steps (smoke and calibration only)")
    ap.add_argument("--n-series", type=int, help="override data.n_series (smoke only)")
    ap.add_argument("--policy", choices=("P1", "P2"), help="determinism policy (smoke only; otherwise from config)")
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--manifest", default="results/session_manifest.json")
    ap.add_argument("--session-id", default=os.environ.get("ABLATION_SESSION_ID"))
    ap.add_argument("--allow-cpu", action="store_true", help="allow CPU for non-smoke stages (debug only)")
    ap.add_argument("--timing-only", action="store_true", help="skip all accuracy evaluation (smoke only; used for policy timing)")
    ap.add_argument("--lock", default="requirements.lock.txt")
    ap.add_argument("--decision", default="results/determinism_decision.json")
    ap.add_argument("--prereg", default="PREREGISTRATION.md")
    ap.add_argument("--allow-dirty", action="store_true", help="debug/test only; recorded in the run file")
    ap.add_argument("--allow-env-mismatch", action="store_true", help="debug/test only; recorded in the run file")
    args = ap.parse_args()

    cfg = load_cfg(args.base, args.config)
    arm = os.path.splitext(os.path.basename(args.config))[0].replace("arm_", "")
    if cfg["model"]["pos_encoding"] not in POS_TYPES:
        fail(f"model.pos_encoding must be one of {POS_TYPES}")
    st = args.stage
    if st in ("calibration", "pilot", "main", "extension"):
        allowed = cfg["study"][STAGE_SEEDS[st]]
        allowed = [allowed] if isinstance(allowed, int) else allowed
        if args.seed not in allowed:
            fail(f"seed {args.seed} is not allowed in stage '{st}' (allowed: {allowed})")
    if args.steps is not None:
        if st not in ("smoke", "calibration"):
            fail("--steps is only allowed in smoke and calibration stages")
        cfg["train"]["steps"] = args.steps
        if st == "smoke":
            cfg["train"]["warmup_steps"] = min(cfg["train"]["warmup_steps"], max(1, args.steps // 10))
    if args.n_series is not None:
        if st != "smoke":
            fail("--n-series is only allowed in the smoke stage")
        cfg["data"]["n_series"] = args.n_series
    if args.timing_only and st != "smoke":
        fail("--timing-only is only allowed in the smoke stage")
    if args.policy:
        if st != "smoke":
            fail("--policy is only allowed in the smoke stage")
        cfg["determinism"]["policy"] = args.policy
    if cfg["train"]["steps"] is None:
        fail("train.steps is null: set it from the calibration and freeze it in PREREGISTRATION.md first")
    if cfg["determinism"]["policy"] is None:
        fail("determinism.policy is null: select it in the smoke stage and freeze it first")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and st != "smoke" and not args.allow_cpu:
        fail("no GPU found; CPU runs are only allowed in the smoke stage (or with --allow-cpu for debugging)")

    policy = cfg["determinism"]["policy"]
    info = env_info(device)
    gates = enforce_gates(args, cfg, st, info)
    apply_determinism(policy)
    run_dir = os.path.join(args.out, st)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
    prefix = f"{arm}_seed{args.seed}"
    if st != "smoke" and has_completed(run_dir, prefix):
        print(f"{prefix}: already completed in stage '{st}', skipping")
        sys.exit(3)
    attempt = next_attempt(run_dir, prefix)
    stem = os.path.join(run_dir, f"{prefix}_attempt{attempt}")
    marker = stem + ".running"
    atomic_write(marker, {"started_utc": now(), "pid": os.getpid()})

    session_id = args.session_id or default_session_id()
    segment = ensure_segment(args.manifest, session_id, info, policy)
    record = {"status": "failed", "stage": st, "arm": arm, "seed": args.seed, "attempt": attempt,
              "session_id": session_id, "segment": segment, "policy": policy, "started_utc": now(),
              "pos_encoding": cfg["model"]["pos_encoding"], "steps": cfg["train"]["steps"], "env": info,
              "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(), "gates": gates,
              "cpu_debug": device.type == "cpu"}
    try:
        with attention_context(policy):
            record.update(run(cfg, args, device))
        record["status"] = "completed"
    except Exception:
        record["error"] = traceback.format_exc()
        print(record["error"], file=sys.stderr)
    record["finished_utc"] = now()
    atomic_write(stem + ".json", record)
    os.remove(marker)
    if record["status"] != "completed":
        sys.exit(1)
    val = "n/a (timing only)" if record["val_mape"] is None else f"{record['val_mape']:.3f}"
    print(f"{arm} seed {args.seed} [{st}] val_mape={val} train_s={record['train_seconds']:.1f} "
          f"final_train_loss={record['final_train_loss_100']:.4f} -> {stem}.json")


if __name__ == "__main__":
    main()
