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


def test_calibration_runs_of_one_arm_at_different_step_counts_are_separate_runs(env):
    """Regression: the run key must include the step count, or the second grid value is skipped as already completed."""
    assert run(env, "A", 900, "calibration", "--steps", "10").returncode == 0
    assert run(env, "A", 900, "calibration", "--steps", "20").returncode == 0
    d = os.path.join(env["out"], "calibration")
    assert sorted(os.listdir(d)) == ["A_seed900_S10_attempt1.json", "A_seed900_S20_attempt1.json"]
    assert run(env, "A", 900, "calibration", "--steps", "20").returncode == 3            # same arm, seed and steps: skipped


def test_completed_run_is_written_then_skipped(env):
    p = run(env)
    assert p.returncode == 0, p.stderr
    f = os.path.join(env["out"], "calibration", "A_seed900_S12_attempt1.json")
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
    marker = os.path.join(d, "A_seed900_S12_attempt1.running"); open(marker, "w").write("{}")
    p = run(env)
    assert p.returncode == 0, p.stderr
    assert os.path.exists(marker)                     # the interrupted attempt is never deleted
    assert os.path.exists(os.path.join(d, "A_seed900_S12_attempt2.json"))


def test_segment_manifest_records_session_once(env):
    run(env, "A", 900); run(env, "B", 900)
    segs = json.load(open(env["manifest"]))
    assert len(segs) == 1 and segs[0]["segment"] == 0
    run(env, "C", 900, "calibration", "--session-id", "other-session")
    assert len(json.load(open(env["manifest"]))) == 2


def test_same_seed_same_result_under_policy_p1(env):
    run(env, "A", 900, "calibration")
    a = json.load(open(os.path.join(env["out"], "calibration", "A_seed900_S12_attempt1.json")))
    env2 = dict(env, out=env["out"] + "2")
    run(env2, "A", 900, "calibration")
    b = json.load(open(os.path.join(env2["out"], "calibration", "A_seed900_S12_attempt1.json")))
    assert a["val_mape"] is not None
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


def test_arms_b_and_c_are_blinded_in_calibration_and_pilot_but_not_arm_a_or_main(env):
    def load(stage, arm, seed):
        suffix = "_S12" if stage == "calibration" else ""
        return json.load(open(os.path.join(env["out"], stage, f"{arm}_seed{seed}{suffix}_attempt1.json")))
    for arm in ("B", "C"):
        assert run(env, arm, 900, "calibration").returncode == 0
        r = load("calibration", arm, 900)
        assert r["eval_blinded"] is True
        assert r["val_mape"] is None and r["val_mae_kw"] is None and r["test_mape"] is None and r["baselines"] is None
        assert "train_seconds" in r and "final_train_loss_100" in r
    assert run(env, "B", 901, "pilot").returncode == 0
    assert load("pilot", "B", 901)["val_mape"] is None                                   # blinded in the pilot too
    assert run(env, "A", 900, "calibration").returncode == 0
    ra = load("calibration", "A", 900)
    assert ra["eval_blinded"] is False and ra["val_mape"] is not None                    # the reference arm is evaluated
    rm = run(env, "B", 0, "main")
    assert rm.returncode == 0, rm.stderr
    rb = load("main", "B", 0)
    assert rb["eval_blinded"] is False and rb["val_mape"] is not None                    # never blinded in the main study


def test_blinded_runs_never_print_a_validation_number(env):
    p = run(env, "B", 900, "calibration")
    assert p.returncode == 0 and "val_mape=not computed" in p.stdout
    assert "val_mape=" in p.stdout and not any(ch.isdigit() for ch in p.stdout.split("val_mape=")[1].split()[0])


def driver_cmd(env, cal_decision, *extra):
    return [sys.executable, os.path.join(ROOT, "src", "run_calibration.py"), "--grid", "10", "20", "40", "--base", env["base"],
            "--runs", os.path.join(env["out"], "calibration"), "--out", env["out"], "--manifest", env["manifest"],
            "--calibration-decision", cal_decision, "--allow-cpu", "--data-dir", env["data"], "--lock", env["lock"],
            "--decision", env["decision"], "--prereg", env["prereg"], "--allow-dirty", *extra]


def test_calibration_driver_runs_only_what_the_rule_needs_and_never_touches_pilot_or_main(env):
    cal_decision = str(env["tmp"] / "calibration_decision.json")
    cmd = driver_cmd(env, cal_decision, "--threshold", "5.0")      # a loose threshold, so stage 1 qualifies at the first value
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stdout + p.stderr
    d = json.load(open(cal_decision))
    assert d["validation_used"] is False and d["status"] == "ok" and d["recommended_S"] == 10
    cal_dir = os.path.join(env["out"], "calibration")
    names = sorted(n for n in os.listdir(cal_dir) if n.endswith(".json"))
    assert names == ["A_seed900_S10_attempt1.json", "A_seed900_S20_attempt1.json", "A_seed900_S40_attempt1.json",
                     "B_seed900_S10_attempt1.json", "B_seed900_S20_attempt1.json",
                     "C_seed900_S10_attempt1.json", "C_seed900_S20_attempt1.json"]       # stage 1 on A, stage 2 only at S* and the next
    for n in names:
        r = json.load(open(os.path.join(cal_dir, n)))
        assert (r["val_mape"] is None) == (not n.startswith("A_"))                           # B and C never evaluated
    assert not os.path.exists(os.path.join(env["out"], "pilot")) and not os.path.exists(os.path.join(env["out"], "main"))
    again = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert again.returncode == 0 and "0 run this invocation" in again.stdout               # nothing left to run


def test_calibration_driver_stops_for_a_human_when_no_grid_value_qualifies(env):
    cal_decision = str(env["tmp"] / "calibration_decision.json")
    p = subprocess.run(driver_cmd(env, cal_decision, "--threshold", "-5.0"), capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stdout + p.stderr
    d = json.load(open(cal_decision))
    assert d["status"] == "no_grid_value_qualifies" and "8000" in d["next_action"]
    cal_dir = os.path.join(env["out"], "calibration")
    assert sorted(n[:1] for n in os.listdir(cal_dir) if n.endswith(".json")) == ["A", "A", "A"]    # arm A only; B and C not run


def test_calibration_driver_respects_max_runs(env):
    p = subprocess.run(driver_cmd(env, str(env["tmp"] / "d.json"), "--max-runs", "1"), capture_output=True, text=True, cwd=ROOT)
    assert p.returncode != 0 and "max-runs" in p.stderr + p.stdout


def test_pilot_driver_runs_arm_a_seeds_901_and_902_only_and_summarises_without_comparing_arms(env):
    summary = str(env["tmp"] / "pilot_summary.json")
    cmd = [sys.executable, os.path.join(ROOT, "src", "run_pilot.py"), "--base", env["base"], "--out", env["out"], "--manifest", env["manifest"],
           "--summary", summary, "--steps", "12", "--allow-cpu", "--data-dir", env["data"], "--lock", env["lock"],
           "--decision", env["decision"], "--prereg", env["prereg"], "--allow-dirty"]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stdout + p.stderr
    pilot_dir = os.path.join(env["out"], "pilot")
    assert sorted(os.listdir(pilot_dir)) == ["A_seed901_attempt1.json", "A_seed902_attempt1.json"]
    s = json.load(open(summary))
    assert s["arms_compared"] is False and s["test_metrics_read"] is False and s["delta_set"] is False
    assert s["seeds"] == [901, 902] and s["n_seeds"] == 2
    assert not os.path.exists(os.path.join(env["out"], "main")) and not os.path.exists(os.path.join(env["out"], "extension"))
    assert subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).returncode == 0       # re-running resumes, nothing is duplicated
    assert len(os.listdir(pilot_dir)) == 2


def test_pilot_driver_refuses_other_seeds(env):
    p = subprocess.run([sys.executable, os.path.join(ROOT, "src", "run_pilot.py"), "--seeds", "0", "--base", env["base"]],
                       capture_output=True, text=True, cwd=ROOT)
    assert p.returncode != 0 and "only the pilot seeds" in p.stderr + p.stdout
