"""CLI behaviour: stage guards, skip-if-completed, attempt counters and interruption markers.
Runs the real script on CPU with a tiny config. These are plumbing tests, not experiment results."""
import copy
import json
import os
import subprocess
import sys

import pytest
import torch
import yaml

import check_env

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "src", "run_ablation.py")


@pytest.fixture()
def env(tmp_path, base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["n_series"] = 2
    cfg["train"]["steps"], cfg["train"]["warmup_steps"] = 12, 2
    cfg["determinism"]["policy"] = "P1"
    cfg["measure"]["inference_latency"].update(warmup_calls=2, timed_calls=5, n_windows=5)
    base = tmp_path / "base.yaml"; base.write_text(yaml.safe_dump(cfg))
    lock = tmp_path / "requirements.lock.txt"; check_env.capture(str(lock))
    decision = tmp_path / "decision.json"; decision.write_text(json.dumps({"selected_policy": "P1"}))
    prereg = tmp_path / "PREREG.md"; prereg.write_text("frozen, no placeholders")
    return {"base": str(base), "out": str(tmp_path / "runs"), "data": str(tmp_path / "data"),
            "manifest": str(tmp_path / "manifest.json"), "tmp": tmp_path, "lock": str(lock),
            "decision": str(decision), "prereg": str(prereg)}


def run(env, arm="A", seed=900, stage="calibration", *extra):
    cmd = [sys.executable, SCRIPT, "--config", os.path.join(ROOT, "configs", f"arm_{arm}.yaml"), "--base", env["base"],
           "--seed", str(seed), "--stage", stage, "--out", env["out"], "--data-dir", env["data"],
           "--manifest", env["manifest"], "--allow-cpu", "--lock", env["lock"], "--decision", env["decision"],
           "--prereg", env["prereg"], "--allow-dirty", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


def test_main_stage_refuses_null_steps(base_cfg, tmp_path):
    p = subprocess.run([sys.executable, SCRIPT, "--config", os.path.join(ROOT, "configs", "arm_A.yaml"),
                        "--seed", "0", "--stage", "main", "--out", str(tmp_path)], capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 2 and "train.steps is null" in p.stderr


def test_seed_must_belong_to_stage(env):
    p = run(env, seed=5)
    assert p.returncode == 2 and "not allowed in stage" in p.stderr


def test_steps_override_only_in_smoke_and_calibration(env):
    p = run(env, "A", 0, "main", "--steps", "10")
    assert p.returncode == 2 and "--steps is only allowed" in p.stderr


def test_completed_run_is_written_then_skipped(env):
    p = run(env)
    assert p.returncode == 0, p.stderr
    f = os.path.join(env["out"], "calibration", "A_seed900_attempt1.json")
    r = json.load(open(f))
    assert r["status"] == "completed" and r["stage"] == "calibration" and r["policy"] == "P1"
    assert r["cpu_debug"] == (not torch.cuda.is_available())
    assert r["gates"]["env_matches_lock"] is True and r["gates"]["decision_policy"] == "P1"
    assert not os.path.exists(f.replace(".json", ".running"))
    for key in ("val_mape", "train_seconds", "inference_latency_ms", "shared_init_weights_sha256", "segment"):
        assert key in r
    assert run(env).returncode == 3                   # already completed: skipped, no second completed attempt


def test_interrupted_attempt_is_kept_and_next_attempt_is_numbered(env):
    d = os.path.join(env["out"], "calibration"); os.makedirs(d)
    marker = os.path.join(d, "A_seed900_attempt1.running"); open(marker, "w").write("{}")
    p = run(env)
    assert p.returncode == 0, p.stderr
    assert os.path.exists(marker)                     # the interrupted attempt is never deleted
    assert os.path.exists(os.path.join(d, "A_seed900_attempt2.json"))


def test_segment_manifest_records_session_once(env):
    run(env, "A", 900); run(env, "B", 900)
    segs = json.load(open(env["manifest"]))
    assert len(segs) == 1 and segs[0]["segment"] == 0
    run(env, "C", 900, "calibration", "--session-id", "other-session")
    assert len(json.load(open(env["manifest"]))) == 2


def test_same_seed_same_result_under_policy_p1(env):
    run(env, "B", 900, "calibration")
    a = json.load(open(os.path.join(env["out"], "calibration", "B_seed900_attempt1.json")))
    env2 = dict(env, out=env["out"] + "2")
    run(env2, "B", 900, "calibration")
    b = json.load(open(os.path.join(env2["out"], "calibration", "B_seed900_attempt1.json")))
    assert a["final_weights_sha256"] == b["final_weights_sha256"] and a["val_mape"] == b["val_mape"]


def test_gate_blocks_when_lock_is_missing(env):
    os.remove(env["lock"])
    p = run(env)
    assert p.returncode == 2 and "is missing" in p.stderr


def test_gate_blocks_when_installed_packages_differ_from_lock(env):
    text = open(env["lock"]).read().replace(f"numpy=={check_env.installed('numpy')}", "numpy==0.0.1")
    open(env["lock"], "w").write(text)
    p = run(env)
    assert p.returncode == 2 and "differ" in p.stderr
    assert run(env, "A", 900, "calibration", "--allow-env-mismatch").returncode == 0   # explicit, recorded override


def test_gate_blocks_when_decision_is_missing_or_disagrees(env):
    open(env["decision"], "w").write(json.dumps({"selected_policy": "P2"}))
    p = run(env)
    assert p.returncode == 2 and "does not match the recorded decision" in p.stderr
    os.remove(env["decision"])
    assert "is missing" in run(env).stderr


def test_main_stage_is_blocked_until_preregistration_is_frozen(env):
    open(env["prereg"], "w").write("Freeze record: commit `______`")
    p = run(env, "A", 0, "main")
    assert p.returncode == 2 and "not frozen" in p.stderr
    assert not os.path.exists(os.path.join(env["out"], "main"))      # nothing was written
    open(env["prereg"], "w").write("frozen")
    assert run(env, "A", 0, "main").returncode == 0


def test_timing_only_records_no_accuracy(env):
    p = run(env, "A", 0, "smoke", "--timing-only")
    assert p.returncode == 0, p.stderr
    r = json.load(open(os.path.join(env["out"], "smoke", "A_seed0_attempt1.json")))
    assert r["timing_only"] is True
    assert r["val_mape"] is None and r["val_mae_kw"] is None and r["test_mape"] is None and r["baselines"] is None
    assert "train_seconds" in r and "inference_latency_ms" in r


def test_timing_only_is_refused_outside_smoke(env):
    assert run(env, "A", 900, "calibration", "--timing-only").returncode == 2
