"""CLI behaviour: stage guards, skip-if-completed, attempt counters and interruption markers.
Runs the real script on CPU with a tiny config. These are plumbing tests, not experiment results."""
import copy
import json
import os
import subprocess
import sys

import pytest
import yaml

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
    return {"base": str(base), "out": str(tmp_path / "runs"), "data": str(tmp_path / "data"),
            "manifest": str(tmp_path / "manifest.json"), "tmp": tmp_path}


def run(env, arm="A", seed=900, stage="calibration", *extra):
    cmd = [sys.executable, SCRIPT, "--config", os.path.join(ROOT, "configs", f"arm_{arm}.yaml"), "--base", env["base"],
           "--seed", str(seed), "--stage", stage, "--out", env["out"], "--data-dir", env["data"],
           "--manifest", env["manifest"], "--allow-cpu", *extra]
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
    assert r["status"] == "completed" and r["stage"] == "calibration" and r["policy"] == "P1" and r["cpu_debug"]
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
