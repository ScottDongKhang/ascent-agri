"""Exposure overlays: MA filter and causal vol targeting."""
import numpy as np
import pandas as pd

from ascentagri.alpha.vol_sizing import ma_filter_scale, vol_target_scale


def test_ma_filter_cuts_below_ma():
    dates = pd.bdate_range("2020-01-01", periods=500)
    # rise for 300 days then crash 40%
    px = np.concatenate([np.linspace(100, 200, 300), np.linspace(200, 120, 200)])
    close = pd.Series(px, index=dates)
    scale = ma_filter_scale(close, dates, ma_window=200, ma_min_periods=150,
                            multiplier=0.7)
    assert set(scale.unique()) <= {0.7, 1.0}
    assert (scale.iloc[:300] == 1.0).all()        # uptrend: above MA
    assert (scale.iloc[-50:] == 0.7).all()        # deep in the crash: below MA


def test_ma_filter_no_cut_before_history():
    dates = pd.bdate_range("2020-01-01", periods=100)
    close = pd.Series(np.linspace(200, 100, 100), index=dates)  # falling
    scale = ma_filter_scale(close, dates, ma_window=200, ma_min_periods=150)
    assert (scale == 1.0).all()   # MA unknown → no cut on unknowable info


def test_vol_target_scale_bounds_and_causality():
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2020-01-01", periods=400)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 400))), index=dates)
    scale = vol_target_scale(close, dates, target_vol=0.20, floor=0.25, cap=1.0)
    assert (scale >= 0.25 - 1e-12).all()
    assert (scale <= 1.0 + 1e-12).all()
    # causality: value at date d must not change when future data is appended
    scale_trunc = vol_target_scale(close.iloc[:300], dates[:300],
                                   target_vol=0.20, floor=0.25, cap=1.0)
    pd.testing.assert_series_equal(scale.iloc[:300], scale_trunc)


def test_vol_target_scales_down_high_vol():
    rng = np.random.default_rng(6)
    dates = pd.bdate_range("2020-01-01", periods=300)
    calm = rng.normal(0, 0.005, 150)     # ~8% ann vol → scale capped at 1.0
    wild = rng.normal(0, 0.04, 150)      # ~63% ann vol → scale ~0.3
    close = pd.Series(100 * np.exp(np.cumsum(np.concatenate([calm, wild]))), index=dates)
    scale = vol_target_scale(close, dates, target_vol=0.20, floor=0.25, cap=1.0)
    assert scale.iloc[140] > scale.iloc[-1]
    assert scale.iloc[-1] < 0.5
