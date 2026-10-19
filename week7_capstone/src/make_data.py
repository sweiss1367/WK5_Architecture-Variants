"""Seeded synthetic hourly electricity-demand generator and leakage-safe windowing.

Run `python src/make_data.py --config configs/base.yaml` to generate the dataset, print a
summary and write data/demand_<cfg-hash>.npz plus a .sha256 file. The same config and data
seed always give byte-identical arrays.

Generator in short (all parameters drawn once per series from the data seed):
  level  = base_kw * (1 + daily_profile(hour)) * weekend_factor(day) * (1 + temperature_load)
  demand = level * exp(AR(1) noise), with rare outlier hours and short gaps
The temperature is a latent series (annual + daily cycle + AR(1) daily anomaly). The model
never sees it. Gaps are filled with the last earlier value (causal), and demand has a
positive floor so MAPE is well defined.
"""
import argparse
import hashlib
import json
import os

import numpy as np
import yaml

GENERATOR_VERSION = "gen-v1"
START_DOW = 4              # day 0 is a Friday (2021-01-01); fixed so weekday structure is reproducible
OUTLIER_PROB = 0.0005      # per series-hour
OUTLIER_MULTS = [0.6, 0.7, 1.3, 1.5]
GAP_PROB_PER_DAY = 0.003   # per series-day, gap length 1-6 hours
AR_NOISE_PHI = 0.6


def split_days(d):
    """Days in train / val / test. Months are converted at 365/12 days and floored; test takes the remainder."""
    total = d["years"] * 365
    train = int(d["split_months"]["train"] * 365 / 12)
    val = int(d["split_months"]["val"] * 365 / 12)
    return train, val, total - train - val


def split_bounds(d):
    """Half-open hour ranges [lo, hi) of each split. All boundaries are multiples of 24."""
    tr, va, te = split_days(d)
    return {"train": (0, tr * 24), "val": (tr * 24, (tr + va) * 24), "test": ((tr + va) * 24, (tr + va + te) * 24)}


def causal_fill(x):
    """Forward-fill NaNs along the last axis using only earlier values.
    The first element of every row must be valid, so no value is ever taken from the future."""
    x = np.array(x, copy=True)
    assert not np.isnan(x[..., 0]).any(), "first element must be valid for causal fill"
    valid = ~np.isnan(x)
    idx = np.where(valid, np.arange(x.shape[-1]), 0)
    idx = np.maximum.accumulate(idx, axis=-1)
    return np.take_along_axis(x, idx, axis=-1)


def generate(d):
    rng = np.random.default_rng(d["data_seed"])
    S, D = d["n_series"], d["years"] * 365
    T = D * 24
    hours = np.arange(T)
    hod, day = hours % 24, hours // 24
    dow, doy = (day + START_DOW) % 7, day % 365

    base = np.exp(rng.uniform(np.log(200.0), np.log(2000.0), S))          # kW
    cooling_dom = rng.random(S) < 0.5                                       # summer- vs winter-peaking
    cool = np.where(cooling_dom, rng.uniform(0.35, 0.60, S), rng.uniform(0.03, 0.10, S))
    heat = np.where(cooling_dom, rng.uniform(0.03, 0.10, S), rng.uniform(0.35, 0.60, S))
    m_amp, e_amp = rng.uniform(0.15, 0.30, S), rng.uniform(0.25, 0.45, S)
    m_hr, e_hr = rng.uniform(6.5, 8.5, S), rng.uniform(17.5, 20.0, S)
    width = rng.uniform(1.8, 3.0, S)
    weekend = rng.uniform(0.80, 0.95, S)
    sigma = rng.uniform(0.015, 0.04, S)
    t_mean, t_year, t_day = rng.uniform(8, 18, S), rng.uniform(8, 14, S), rng.uniform(3, 6, S)

    # latent temperature (deg C)
    anom = np.zeros((S, D))
    for k in range(1, D):
        anom[:, k] = 0.8 * anom[:, k - 1] + rng.normal(0.0, 2.0, S)
    temp = (t_mean[:, None]
            + t_year[:, None] * np.sin(2 * np.pi * (doy - 110) / 365)[None, :]
            + t_day[:, None] * np.sin(2 * np.pi * (hod - 9) / 24)[None, :]
            + np.repeat(anom, 24, axis=1) + rng.normal(0.0, 0.3, (S, T)))

    prof = (m_amp[:, None] * np.exp(-0.5 * ((hod[None, :] - m_hr[:, None]) / width[:, None]) ** 2)
            + e_amp[:, None] * np.exp(-0.5 * ((hod[None, :] - e_hr[:, None]) / width[:, None]) ** 2))
    workday = np.where((dow < 5)[None, :], 1.0, weekend[:, None])
    temp_load = (cool[:, None] * np.maximum(0.0, temp - 21.0) / 10.0
                 + heat[:, None] * np.maximum(0.0, 10.0 - temp) / 10.0)
    level = base[:, None] * (1.0 + prof) * workday * (1.0 + temp_load)

    eps = np.zeros((S, T))
    innov = rng.normal(0.0, 1.0, (S, T)) * sigma[:, None]
    for t in range(1, T):
        eps[:, t] = AR_NOISE_PHI * eps[:, t - 1] + innov[:, t]
    demand = level * np.exp(eps)

    out_mask = rng.random((S, T)) < OUTLIER_PROB
    out_mult = rng.choice(OUTLIER_MULTS, size=(S, T))
    demand = np.where(out_mask, demand * out_mult, demand)

    demand = np.maximum(demand, d["demand_floor_frac_of_base"] * base[:, None])

    gap_hours = 0
    for s in range(S):
        for k in np.nonzero(rng.random(D) < GAP_PROB_PER_DAY)[0]:
            start = max(1, int(k) * 24 + int(rng.integers(0, 24)))      # never hour 0
            length = int(rng.integers(1, 7))
            demand[s, start:start + length] = np.nan
            gap_hours += min(length, T - start)
    demand = causal_fill(demand).astype(np.float32)

    meta = {"generator": GENERATOR_VERSION, "data_cfg": d, "base_kw": base.tolist(),
            "cooling_dominated": cooling_dom.tolist(), "outlier_hours": int(out_mask.sum()),
            "gap_hours": gap_hours, "n_hours": T}
    return demand, meta


def make_windows(demand, d):
    """Windows keyed by split. A window belongs to the split containing its target hours.
    context = [ts-ctx, ts), target = [ts, ts+hor). ts is a multiple of the stride (and of 24), so every
    window starts at 00:00 and position 24k inside the context is always midnight."""
    ctx, hor, stride = d["context_hours"], d["horizon_hours"], d["window_stride_hours"]
    assert stride % 24 == 0 and d["window_start_hour"] == 0
    out = {}
    for split, (lo, hi) in split_bounds(d).items():
        first = -(-max(lo, ctx) // stride) * stride
        ts = np.arange(first, hi - hor + 1, stride)
        xs, ys, ser, tss = [], [], [], []
        for s in range(demand.shape[0]):
            xs.append(demand[s][ts[:, None] + np.arange(-ctx, 0)[None, :]])
            ys.append(demand[s][ts[:, None] + np.arange(hor)[None, :]])
            ser.append(np.full(len(ts), s)); tss.append(ts)
        out[split] = {"x": np.concatenate(xs), "y": np.concatenate(ys),
                      "series": np.concatenate(ser), "ts": np.concatenate(tss)}
    return out


def normalize(x, y):
    """Divide a window and its target by the mean of the window's own context only."""
    scale = x.mean(axis=1, keepdims=True)
    return x / scale, y / scale, scale


def check_no_leakage(windows, d):
    """Raise AssertionError if any rule in PREREGISTRATION.md Section 9 is broken."""
    ctx, hor = d["context_hours"], d["horizon_hours"]
    bounds = split_bounds(d)
    for split, w in windows.items():
        lo, hi = bounds[split]
        ts = w["ts"]
        assert (ts >= lo).all() and (ts + hor <= hi).all(), f"{split}: target hours outside the split"
        assert (ts - ctx >= 0).all(), f"{split}: context starts before the series"
        assert (ts % 24 == 0).all(), f"{split}: window not aligned to midnight"
    order = ["train", "val", "test"]
    for i, early in enumerate(order):
        touched_max = (windows[early]["ts"] + hor - 1).max()      # latest hour any context or target reaches
        for late in order[i + 1:]:
            late_target_min = windows[late]["ts"].min()
            assert touched_max < late_target_min, f"{late} target hour appears in a {early} window"


def cfg_hash(d):
    return hashlib.sha256(json.dumps([GENERATOR_VERSION, d], sort_keys=True).encode()).hexdigest()[:10]


def array_sha256(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def generate_or_load(d, data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    stem = os.path.join(data_dir, f"demand_{cfg_hash(d)}")
    if os.path.exists(stem + ".npz"):
        z = np.load(stem + ".npz", allow_pickle=False)
        return z["demand"], json.loads(str(z["meta"])), array_sha256(z["demand"])
    demand, meta = generate(d)
    digest = array_sha256(demand)
    meta["array_sha256"] = digest
    np.savez_compressed(stem + ".npz", demand=demand, meta=json.dumps(meta))
    with open(stem + ".sha256", "w") as f:
        f.write(f"{digest}  float32 {demand.shape}  {GENERATOR_VERSION}\n")
    return demand, meta, digest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--verify", action="store_true", help="regenerate and compare with the stored hash")
    args = ap.parse_args()
    d = yaml.safe_load(open(args.config))["data"]
    demand, meta, digest = generate_or_load(d, args.data_dir)
    print(f"series={demand.shape[0]} hours={demand.shape[1]} sha256={digest}")
    print(f"min={demand.min():.1f} max={demand.max():.1f} outlier_hours={meta['outlier_hours']} gap_hours={meta['gap_hours']}")
    w = make_windows(demand, d)
    check_no_leakage(w, d)
    print({k: len(v["ts"]) for k, v in w.items()}, "windows; leakage check passed")
    if args.verify:
        again, _ = generate(d)
        assert array_sha256(again) == digest, "regenerated data differs from stored hash"
        print("verify: regenerated hash matches")
