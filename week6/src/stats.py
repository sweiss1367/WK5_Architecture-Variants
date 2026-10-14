"""
Statistics for the fairness audit. numpy/pandas only.

Interpretation rule (fixed in analysis_plan.md before any model output exists):
    * 95% CI of the paired slice difference excludes 0  -> "evidence of a difference"
    * 95% CI includes 0                                  -> "unresolved / not statistically established"
      An unresolved result is NEVER reported as evidence that the groups perform equally.
    * Point estimate, CI and sample sizes are always reported together.

Unit of resampling = the matched base case. Every base case contributes all of its variants
and all of its sampling seeds to each resample, so the CI reflects case-to-case variability
and generation (sampling) variability. It does NOT reflect training-seed variability: there is
exactly one trained adapter.
"""
import numpy as np
import pandas as pd

BOOT_B = 10000
BOOT_SEED = 6006

UNRESOLVED = "unresolved: difference not statistically established (95% CI includes 0); this is not evidence that the groups perform equally"
DIFFERENCE = "evidence of a difference (95% CI excludes 0)"


def wilson(k: int, n: int, z: float = 1.959964):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_mean_ci(values, B: int = BOOT_B, seed: int = BOOT_SEED, alpha: float = 0.05):
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], dtype=float)
    n = len(v)
    if n == 0:
        return dict(estimate=float("nan"), ci_low=float("nan"), ci_high=float("nan"), n=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boots = v[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(estimate=float(v.mean()), ci_low=float(lo), ci_high=float(hi), n=n)


def interpret(ci_low: float, ci_high: float) -> str:
    if np.isnan(ci_low) or np.isnan(ci_high):
        return "not computable"
    return DIFFERENCE if (ci_low > 0 or ci_high < 0) else UNRESOLVED


def _case_means(df, metric, slice_col, level, case_col):
    return df[df[slice_col] == level].groupby(case_col)[metric].mean()


def paired_slice_difference(
    df: pd.DataFrame, metric: str, slice_col: str, level_a, level_b,
    case_col: str = "base_id", seed_col: str = "seed",
    B: int = BOOT_B, boot_seed: int = BOOT_SEED,
):
    """mean over cases of [metric(level_a) - metric(level_b)], where each case's value at a
    level is the mean over all of that case's rows at that level (other slice and seeds)."""
    ma = _case_means(df, metric, slice_col, level_a, case_col)
    mb = _case_means(df, metric, slice_col, level_b, case_col)
    d = (ma - mb).dropna()
    n = len(d)
    if n == 0:
        raise ValueError("no matched cases with a defined metric at both levels")
    rng = np.random.default_rng(boot_seed)
    idx = rng.integers(0, n, size=(B, n))
    boots = d.values[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])

    per_seed = {}
    for s in sorted(df[seed_col].unique()):
        ds = df[df[seed_col] == s]
        dd = (_case_means(ds, metric, slice_col, level_a, case_col) - _case_means(ds, metric, slice_col, level_b, case_col)).dropna()
        per_seed[int(s)] = float(dd.mean()) if len(dd) else float("nan")

    sub = df[[slice_col, metric]].dropna()
    return dict(
        metric=metric, slice=slice_col, level_a=level_a, level_b=level_b,
        estimate=float(d.mean()), ci_low=float(lo), ci_high=float(hi),
        n_cases=int(n),
        n_generations_a=int((sub[slice_col] == level_a).sum()),
        n_generations_b=int((sub[slice_col] == level_b).sum()),
        per_seed_estimates=per_seed,
        interpretation=interpret(lo, hi),
    )


def cell_summary(df: pd.DataFrame, metric: str, mask, case_col: str = "base_id"):
    """Mean and case-level bootstrap CI for the rows selected by `mask`."""
    sub = df[mask]
    per_case = sub.groupby(case_col)[metric].mean()
    out = bootstrap_mean_ci(per_case.values)
    out["n_generations"] = int(sub[metric].notna().sum())
    return out
