import json
import os
import subprocess
import sys

import pytest
import yaml

import analyze_main as AM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELTA, K = 0.10, 2.0


def d3(a, b, c):
    return {0: a, 1: b, 2: c}


# ---- the rule itself ------------------------------------------------------------------------------------------------
def test_supported_tentative_for_a_clear_improvement_negative_differences():
    r = AM.classify(d3(-0.30, -0.32, -0.28), DELTA, K)                       # X has LOWER error than A in every seed
    assert r["verdict"] == "supported, tentative" and r["mean"] < 0 and r["abs_mean"] == pytest.approx(0.30)
    assert "LOWER error" in r["direction"] and all(r["conditions"].values())


def test_supported_tentative_for_a_clear_deterioration_positive_differences():
    r = AM.classify(d3(0.30, 0.32, 0.28), DELTA, K)
    assert r["verdict"] == "supported, tentative" and "HIGHER error" in r["direction"]


def test_magnitude_uses_the_absolute_mean_so_sign_does_not_change_the_verdict():
    for sgn in (1, -1):
        r = AM.classify(d3(sgn * 0.30, sgn * 0.32, sgn * 0.28), DELTA, K)
        assert r["verdict"] == "supported, tentative" and r["abs_mean"] == pytest.approx(0.30)
    assert AM.classify(d3(-0.30, -0.32, -0.28), DELTA, K)["mean"] == -AM.classify(d3(0.30, 0.32, 0.28), DELTA, K)["mean"]


def test_mixed_signs_fail_condition_a_even_with_a_large_mean():
    r = AM.classify(d3(0.50, 0.50, -0.01), DELTA, K)
    assert r["conditions"]["a_same_sign"] is False and r["verdict"] == "no difference detectable at this budget"


def test_a_zero_difference_is_not_a_sign():
    assert AM.classify(d3(0.2, 0.2, 0.0), DELTA, K)["conditions"]["a_same_sign"] is False


def test_same_sign_but_below_delta_is_no_difference():
    r = AM.classify(d3(0.02, 0.03, 0.04), DELTA, K)
    assert r["conditions"]["a_same_sign"] and not r["conditions"]["b_abs_mean_at_least_delta"]
    assert r["verdict"] == "no difference detectable at this budget"


def test_delta_boundary_is_inclusive():
    r = AM.classify(d3(0.10, 0.10, 0.10), DELTA, K)
    assert r["conditions"]["b_abs_mean_at_least_delta"] is True and r["verdict"] == "supported, tentative"   # SD_d = 0, so (c) holds
    assert AM.classify(d3(0.0999, 0.0999, 0.0999), DELTA, K)["conditions"]["b_abs_mean_at_least_delta"] is False


def test_direction_only_when_the_spread_is_too_large_for_condition_c():
    r = AM.classify(d3(0.10, 0.30, 0.20), DELTA, K)                          # mean 0.20, SD 0.10: |mean| >= 2*SD holds exactly
    assert r["conditions"]["c_abs_mean_at_least_k_sd"] is True
    r = AM.classify(d3(0.05, 0.40, 0.15), DELTA, K)                          # mean 0.20, SD ~0.18: (c) fails
    assert r["conditions"]["a_same_sign"] and r["conditions"]["b_abs_mean_at_least_delta"]
    assert r["conditions"]["c_abs_mean_at_least_k_sd"] is False and r["verdict"] == "direction only"


def test_direction_only_when_a_leave_one_out_mean_drops_below_delta():
    r = AM.classify(d3(0.02, 0.50, 0.50), DELTA, K)                          # dropping seed 1 or 2 leaves a mean of 0.26; dropping 0 gives 0.50
    assert r["conditions"]["d_leave_one_out_consistent"] is True
    r = AM.classify(d3(0.30, 0.01, 0.01), DELTA, K)                          # mean 0.107 >= delta, but leaving seed 0 out gives 0.01
    assert r["conditions"]["b_abs_mean_at_least_delta"] and not r["conditions"]["d_leave_one_out_consistent"]
    assert r["verdict"] == "direction only"


def test_leave_one_out_means_are_reported_for_every_seed():
    r = AM.classify(d3(0.1, 0.2, 0.6), DELTA, K)
    assert r["leave_one_out_means"] == {"0": pytest.approx(0.4), "1": pytest.approx(0.35), "2": pytest.approx(0.15)}
    assert r["sd"] == pytest.approx(0.2645751311)                            # n-1 denominator
    assert r["k_times_sd"] == pytest.approx(2 * r["sd"])


# ---- from run files to the summary ----------------------------------------------------------------------------------
def write_run(d, arm, seed, val, stage="main", status="completed", attempt=1, seg=0, gpu="GPU-aaaa1111", commit="c" * 40, **kw):
    os.makedirs(d, exist_ok=True)
    r = {"arm": arm, "seed": seed, "attempt": attempt, "status": status, "stage": stage, "segment": seg, "session_id": "boot-x",
         "policy": "P1", "steps": 8000, "val_mape": val, "val_mae_kw": 50.0, "test_mape": 4.7, "final_train_loss_100": 0.044,
         "train_seconds": 300.0 + seed, "train_peak_alloc_mb": 623.3, "inference_peak_alloc_mb": 66.9,
         "resident_alloc_before_inference_mb": 65.6, "inference_latency_ms": {"p50_ms": 4.4, "p95_ms": 5.0},
         "baselines": {}, "params": {"total": 1}, "config_sha256": "t" * 64, "final_weights_sha256": "w" * 64, "data_sha256": "d" * 64,
         "env": {"gpu_uuid": gpu, "git_commit": commit}, "started_utc": "t", "finished_utc": "t", "error": None, "eval_blinded": False, **kw}
    json.dump(r, open(os.path.join(d, f"{arm}_seed{seed}_attempt{attempt}" + ("" if stage == "main" else f"_{stage}") + ".json"), "w"))


@pytest.fixture()
def cfg():
    return yaml.safe_load(open(os.path.join(ROOT, "configs", "base.yaml")))


def full(d, a=(4.67, 4.68, 4.66), b=(4.40, 4.41, 4.39), c=(4.90, 4.92, 4.91)):
    for arm, vals in (("A", a), ("B", b), ("C", c)):
        for s, v in enumerate(vals):
            write_run(d, arm, s, v)


def test_frozen_parameters_are_read_from_the_config(cfg):
    assert cfg["analysis"]["delta_mape_points"] == 0.10 and cfg["analysis"]["k"] == 2.0 and cfg["analysis"]["reference_arm"] == "A"
    assert cfg["analysis"]["planned_comparisons"] == [["B", "A"], ["C", "A"]] and cfg["analysis"]["descriptive_comparisons"] == [["C", "B"]]


def test_summary_pairs_arms_by_seed_and_uses_main_arm_a_only(tmp_path, cfg):
    d = str(tmp_path)
    full(d)
    write_run(d, "A", 0, 99.0, stage="pilot")                                # a pilot arm A run must be ignored
    write_run(d, "A", 3, 99.0, stage="extension")                            # and so must an extension run
    rows, markers = AM.load_main(d)
    s = AM.analyse(rows, markers, cfg)
    b = s["comparisons"]["B_vs_A"]
    assert s["complete"] and b["status"] == "computed"
    assert b["d_by_seed"]["0"] == pytest.approx(4.40 - 4.67) and b["d_by_seed"]["2"] == pytest.approx(4.39 - 4.66)
    assert b["verdict"] == "supported, tentative" and "LOWER" in b["direction"]
    c = s["comparisons"]["C_vs_A"]
    assert c["verdict"] == "supported, tentative" and "HIGHER" in c["direction"]
    assert s["secondary_hypothesis_RoPE_higher_error"]["reading"] == "consistent in direction with the conjecture"
    assert s["comparisons"]["C_vs_B"]["planned"] is False and "descriptive only" in s["comparisons"]["C_vs_B"]["verdict_note"]
    assert s["per_arm"]["A"]["val_mape"]["values"] == [4.67, 4.68, 4.66]


def test_opposite_direction_for_the_secondary_hypothesis_is_reported_plainly(tmp_path, cfg):
    d = str(tmp_path); full(d, c=(4.40, 4.41, 4.39))
    s = AM.analyse(*AM.load_main(d), cfg)
    assert s["secondary_hypothesis_RoPE_higher_error"]["reading"] == "opposite to the conjecture"


def test_no_verdict_until_all_three_seeds_exist_for_both_arms(tmp_path, cfg):
    d = str(tmp_path); full(d)
    os.remove(os.path.join(d, "C_seed2_attempt1.json"))
    s = AM.analyse(*AM.load_main(d), cfg)
    assert s["complete"] is False and s["comparisons"]["C_vs_A"]["status"] == "incomplete"
    assert s["comparisons"]["C_vs_A"].get("verdict") is None and s["comparisons"]["B_vs_A"]["status"] == "computed"


def test_failed_and_interrupted_attempts_are_listed_and_change_the_session_classification(tmp_path, cfg):
    d = str(tmp_path); full(d)
    write_run(d, "B", 1, None, status="failed", attempt=0, error="Traceback...\nRuntimeError: boom")
    open(os.path.join(d, "C_seed0_attempt2.running"), "w").write("{}")
    s = AM.analyse(*AM.load_main(d), cfg)
    assert s["ledger"]["completed"] == 9 and s["ledger"]["failed"] == 1 and s["ledger"]["interrupted_markers"] == ["C_seed0_attempt2.running"]
    assert any(a["error"] == "RuntimeError: boom" for a in s["ledger"]["attempts"])
    assert s["session"]["classification"].startswith("resumed or with failed")


def test_a_clean_single_session_is_classified_uninterrupted(tmp_path, cfg):
    d = str(tmp_path); full(d)
    s = AM.analyse(*AM.load_main(d), cfg)
    assert s["session"]["classification"].startswith("uninterrupted") and s["session"]["segments"] == [0]


def test_two_gpus_or_segments_are_flagged_as_resumed(tmp_path, cfg):
    d = str(tmp_path); full(d)
    write_run(d, "C", 2, 4.91, attempt=1, seg=1, gpu="GPU-bbbb2222")
    s = AM.analyse(*AM.load_main(d), cfg)
    assert s["session"]["classification"].startswith("resumed")


def test_two_completed_attempts_for_one_cell_are_refused(tmp_path, cfg):
    d = str(tmp_path); full(d)
    write_run(d, "A", 0, 4.67, attempt=2)
    with pytest.raises(ValueError, match="more than one completed attempt"):
        AM.analyse(*AM.load_main(d), cfg)


def test_the_cli_writes_the_summary_and_csv_files_and_prints_the_caveat(tmp_path):
    d = str(tmp_path / "main"); full(d)
    out = str(tmp_path / "summary.json")
    p = subprocess.run([sys.executable, os.path.join(ROOT, "src", "analyze_main.py"), "--runs", d, "--out", out, "--csv-dir", str(tmp_path)],
                       capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr
    assert "not significance tests" in p.stdout and "VERDICT" in p.stdout and "descriptive: no verdict is claimed" in p.stdout
    s = json.load(open(out))
    assert s["delta"] == 0.10 and s["k"] == 2.0
    assert len(open(tmp_path / "main_runs.csv").read().strip().splitlines()) == 10        # header + nine runs
    assert len(open(tmp_path / "paired_differences.csv").read().strip().splitlines()) == 1 + 3 * 3   # three comparisons x three seeds
