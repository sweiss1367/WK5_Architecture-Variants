import copy

import numpy as np
import pytest

import make_data as md


@pytest.fixture(scope="module")
def data(small_data_cfg):
    demand, meta = md.generate(small_data_cfg)
    return demand, meta, md.make_windows(demand, small_data_cfg)


def test_generator_is_deterministic(small_data_cfg):
    a, _ = md.generate(small_data_cfg)
    b, _ = md.generate(small_data_cfg)
    assert md.array_sha256(a) == md.array_sha256(b)


def test_different_data_seed_changes_data(small_data_cfg):
    d2 = copy.deepcopy(small_data_cfg); d2["data_seed"] += 1
    a, _ = md.generate(small_data_cfg)
    b, _ = md.generate(d2)
    assert md.array_sha256(a) != md.array_sha256(b)


def test_demand_is_positive_finite_and_floored(data, small_data_cfg):
    demand, meta, _ = data
    assert np.isfinite(demand).all()
    floor = small_data_cfg["demand_floor_frac_of_base"] * np.array(meta["base_kw"])[:, None]
    assert (demand >= floor * 0.9999).all()          # float32 rounding tolerance


def test_split_days(small_data_cfg):
    assert md.split_days(small_data_cfg) == (730, 182, 183)
    b = md.split_bounds(small_data_cfg)
    assert b["train"][1] == b["val"][0] and b["val"][1] == b["test"][0] and b["test"][1] == 1095 * 24


def test_windows_start_at_midnight_and_have_right_shape(data, small_data_cfg):
    _, _, w = data
    ctx, hor = small_data_cfg["context_hours"], small_data_cfg["horizon_hours"]
    for split, ww in w.items():
        assert ww["x"].shape[1] == ctx and ww["y"].shape[1] == hor
        assert (ww["ts"] % 24 == 0).all()
        assert ((ww["ts"] - ctx) % 24 == 0).all()    # first context hour is midnight, so position 24k is midnight


def test_window_contents_match_the_series(data, small_data_cfg):
    demand, _, w = data
    ctx = small_data_cfg["context_hours"]
    for split, ww in w.items():
        for i in (0, len(ww["ts"]) // 2, len(ww["ts"]) - 1):
            s, ts = ww["series"][i], ww["ts"][i]
            assert np.array_equal(ww["x"][i], demand[s, ts - ctx:ts])
            assert np.array_equal(ww["y"][i], demand[s, ts:ts + 24])


def test_leakage_checker_passes_on_real_windows(data, small_data_cfg):
    md.check_no_leakage(data[2], small_data_cfg)


def test_leakage_checker_catches_a_leaking_split(data, small_data_cfg):
    """V19: the checker must fail when a training window's target reaches into validation."""
    w = copy.deepcopy(data[2])
    val_start = md.split_bounds(small_data_cfg)["val"][0]
    w["train"]["ts"] = w["train"]["ts"].copy()
    w["train"]["ts"][-1] = val_start                 # target now lies in the validation period
    with pytest.raises(AssertionError):
        md.check_no_leakage(w, small_data_cfg)


def test_leakage_checker_catches_val_target_inside_train_context(data, small_data_cfg):
    w = copy.deepcopy(data[2])
    w["val"]["ts"] = w["val"]["ts"].copy()
    w["val"]["ts"][0] = md.split_bounds(small_data_cfg)["train"][1] - 24   # val window inside the train period
    with pytest.raises(AssertionError):
        md.check_no_leakage(w, small_data_cfg)


def test_training_windows_do_not_depend_on_later_periods(small_data_cfg):
    demand, _ = md.generate(small_data_cfg)
    b = md.split_bounds(small_data_cfg)
    w0 = md.make_windows(demand, small_data_cfg)
    changed = demand.copy(); changed[:, b["val"][0]:] *= 3.0          # perturb validation and test periods
    w1 = md.make_windows(changed, small_data_cfg)
    assert np.array_equal(w0["train"]["x"], w1["train"]["x"]) and np.array_equal(w0["train"]["y"], w1["train"]["y"])


def test_validation_targets_do_not_depend_on_test_period(small_data_cfg):
    demand, _ = md.generate(small_data_cfg)
    b = md.split_bounds(small_data_cfg)
    w0 = md.make_windows(demand, small_data_cfg)
    changed = demand.copy(); changed[:, b["test"][0]:] *= 3.0
    w1 = md.make_windows(changed, small_data_cfg)
    assert np.array_equal(w0["val"]["x"], w1["val"]["x"]) and np.array_equal(w0["val"]["y"], w1["val"]["y"])


def test_normalization_uses_only_the_context(data):
    _, _, w = data
    x, y = w["val"]["x"], w["val"]["y"]
    xn, yn, scale = md.normalize(x, y)
    assert np.allclose(scale[:, 0], x.mean(1))
    xn2, _, _ = md.normalize(x, y * 7.0)                # changing the target must not change the input
    assert np.array_equal(xn, xn2)


def test_causal_fill_uses_only_earlier_values():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 200)); x[:, 5:9] = np.nan; x[:, 100:103] = np.nan
    base = md.causal_fill(x)
    assert not np.isnan(base).any()
    later = x.copy(); later[:, 120:] = rng.normal(size=(4, 80)) * 100   # change only the future
    assert np.array_equal(md.causal_fill(later)[:, :120], base[:, :120])
    assert (base[:, 5:9] == base[:, [4]]).all()


def test_causal_fill_rejects_leading_nan():
    x = np.array([[np.nan, 1.0, 2.0]])
    with pytest.raises(AssertionError):
        md.causal_fill(x)
