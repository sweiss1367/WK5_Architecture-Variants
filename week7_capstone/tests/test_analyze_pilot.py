import json
import math
import os
import statistics as st

import pytest

import analyze_pilot as AP


def run_file(d, name, arm="A", seed=901, stage="pilot", steps=8000, val=4.7, cfg="c" * 64, status="completed", **extra):
    os.makedirs(d, exist_ok=True)
    r = {"arm": arm, "seed": seed, "steps": steps, "stage": stage, "status": status, "attempt": 1, "val_mape": val, "val_mae_kw": 20.0,
         "baselines": {"seasonal_naive_day_val_mape": 7.08, "same_hour_last_week_val_mape": 7.17, "seasonal_naive_day_test_mape": 99.0},
         "final_train_loss_100": 0.043, "train_seconds": 290.0, "train_steps_per_second": 27.5, "train_peak_alloc_mb": 623.3,
         "inference_latency_ms": {"p50_ms": 4.4}, "config_sha256": cfg, "final_weights_sha256": "w" * 64, "segment": 1,
         "policy": "P1", "env": {"gpu_uuid": "GPU-abcdef12", "git_commit": "a" * 40}, "eval_blinded": False, "data_sha256": "d" * 64,
         "test_mape": 1234.5, **extra}
    json.dump(r, open(os.path.join(d, name), "w"))


def three(tmp_path, vals=(4.674, 4.60, 4.75), cfg="c" * 64):
    pd, cd = str(tmp_path / "pilot"), str(tmp_path / "cal")
    run_file(pd, "A_seed901_attempt1.json", seed=901, val=vals[1], cfg=cfg)
    run_file(pd, "A_seed902_attempt1.json", seed=902, val=vals[2], cfg=cfg)
    run_file(cd, "A_seed900_S8000_attempt1.json", seed=900, stage="calibration", val=vals[0], cfg=cfg)
    return pd, cd


def test_sd_interval_matches_known_chi_square_quantiles():
    lo, hi = AP.sd_interval(1.0, 3)
    assert lo == pytest.approx(math.sqrt(2 / 7.3778), rel=1e-3) and hi == pytest.approx(math.sqrt(2 / 0.050636), rel=1e-3)
    assert lo == pytest.approx(0.52, abs=0.01) and hi == pytest.approx(6.29, abs=0.02)      # "about x0.52 to x6.3"
    with pytest.raises(ValueError):
        AP.sd_interval(1.0, 4)


def test_summary_uses_sample_sd_and_reports_the_interval(tmp_path):
    pd, cd = three(tmp_path)
    s = AP.summarize(AP.collect(pd, cd, 8000))
    v = [4.674, 4.60, 4.75]
    assert s["seeds"] == [900, 901, 902] and s["n_seeds"] == 3
    assert s["mean"] == pytest.approx(st.mean(v)) and s["sd"] == pytest.approx(st.stdev(v))        # n-1 denominator
    assert s["range"] == pytest.approx(0.15) and "sd_95_interval" in s
    assert s["gain_vs_best_baseline_points"] == pytest.approx(7.08 - st.mean(v))


def test_only_arm_a_is_accepted(tmp_path):
    pd, cd = three(tmp_path)
    run_file(pd, "A_seed903_attempt1.json", arm="B", seed=903)
    run_file(pd, "A_seed904_attempt1.json", arm="C", seed=904)         # arm field wrong in an arm-A-named file: must be refused
    with pytest.raises(ValueError, match="arm A only"):
        AP.collect(pd, cd, 8000)


def test_arm_b_and_c_named_files_are_never_even_read(tmp_path):
    pd, cd = three(tmp_path)
    run_file(pd, "B_seed901_attempt1.json", arm="B", seed=901, val=None)
    assert [r["seed"] for r in AP.collect(pd, cd, 8000)] == [900, 901, 902]


def test_seed_900_is_reused_only_with_an_identical_config(tmp_path):
    pd, cd = three(tmp_path)
    run_file(cd, "A_seed900_S8000_attempt1.json", seed=900, stage="calibration", cfg="x" * 64)     # a different merged config
    with pytest.raises(ValueError, match="cannot be reused"):
        AP.summarize(AP.collect(pd, cd, 8000))


def test_other_step_counts_stages_and_failed_runs_are_ignored(tmp_path):
    pd, cd = three(tmp_path)
    run_file(pd, "A_seed905_attempt1.json", seed=905, steps=4000)
    run_file(pd, "A_seed906_attempt1.json", seed=906, status="failed")
    run_file(pd, "A_seed907_attempt1.json", seed=907, stage="smoke")
    assert [r["seed"] for r in AP.collect(pd, cd, 8000)] == [900, 901, 902]


def test_test_set_metrics_are_never_loaded_or_reported(tmp_path):
    assert "test_mape" not in AP.LOADED and not any("test" in k for k in AP.LOADED)
    pd, cd = three(tmp_path)
    runs = AP.collect(pd, cd, 8000)
    assert all("test_mape" not in r for r in runs)
    assert all(not k.startswith("seasonal_naive_day_test") for k in runs[0]["baselines"] if k in AP.BASELINE_KEYS)
    assert set(AP.summarize(runs)["baselines"]) == set(AP.BASELINE_KEYS)
