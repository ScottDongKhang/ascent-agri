"""The N=1 fix: rolling time-series z-score must produce non-zero scores for
a single instrument (the ported cross-sectional z-score silently returned 0
everywhere at N=1), and must be causal."""
import numpy as np
import pandas as pd

from ascentagri.alpha.normalize import ts_normalize


def _trending_series(n=400, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    return pd.Series(np.cumsum(rng.normal(0.5, 1.0, n)) + 100, index=dates)


def test_single_series_scores_are_not_all_zero():
    s = _trending_series()
    z = ts_normalize(s, window=100, min_periods=30)
    # after warmup, scores must carry signal — the N=1 failure mode is all-zero
    assert (z.iloc[50:].abs() > 1e-6).mean() > 0.5


def test_clip_bounds():
    s = _trending_series()
    z = ts_normalize(s, window=100, min_periods=30, clip=3.0)
    assert z.max() <= 3.0 + 1e-12
    assert z.min() >= -3.0 - 1e-12


def test_warmup_is_zero_not_nan():
    s = _trending_series()
    z = ts_normalize(s, window=100, min_periods=60)
    assert not z.isna().any()
    assert (z.iloc[:59] == 0).all()   # before min_periods → no signal


def test_causality_future_data_does_not_change_past_scores():
    s = _trending_series()
    z_full = ts_normalize(s, window=100, min_periods=30)
    # Recompute on a truncated series: overlapping scores must be identical
    cut = len(s) - 50
    z_trunc = ts_normalize(s.iloc[:cut], window=100, min_periods=30)
    pd.testing.assert_series_equal(z_full.iloc[:cut], z_trunc)


def test_constant_series_yields_zero():
    dates = pd.bdate_range("2022-01-03", periods=200)
    s = pd.Series(100.0, index=dates)
    z = ts_normalize(s, window=50, min_periods=20)
    assert (z == 0).all()
