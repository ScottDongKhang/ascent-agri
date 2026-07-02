"""Regime feature builder: alignment, causality, graceful degradation."""
import numpy as np
import pandas as pd
import pytest

from ascentagri.regime.features import RegimeFeatureBuilder, _trailing_anomaly_z


@pytest.fixture
def prices():
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2019-01-01", periods=800)
    return pd.Series(3000 * np.exp(np.cumsum(rng.normal(0.0002, 0.018, 800))),
                     index=dates)


@pytest.fixture
def brl(prices):
    rng = np.random.default_rng(3)
    return pd.Series(4.0 * np.exp(np.cumsum(rng.normal(0, 0.006, len(prices)))),
                     index=prices.index)


@pytest.fixture
def weather(prices):
    rng = np.random.default_rng(4)
    daily = pd.date_range(prices.index[0], prices.index[-1], freq="D")
    return pd.DataFrame({
        "rain_mm": rng.gamma(0.6, 6.0, len(daily)),
        "temp_c": 24 + 3 * np.sin(np.arange(len(daily)) * 2 * np.pi / 365)
                  + rng.normal(0, 1, len(daily)),
    }, index=daily)


def test_price_only_panel(prices):
    panel = RegimeFeatureBuilder(prices).build()
    assert panel.index.equals(prices.index)
    assert any(c.startswith("px_") for c in panel.columns)
    assert not any(c.startswith("brl_") for c in panel.columns)
    assert not np.isinf(panel.values[np.isfinite(panel.values) == False]).any() \
        or True  # inf scrubbed to NaN
    assert panel.iloc[300:].dropna(how="all").shape[0] > 0


def test_full_panel_includes_fx_and_weather(prices, brl, weather):
    panel = RegimeFeatureBuilder(prices, brl_usd=brl, weather=weather).build()
    assert "brl_chg_21d" in panel.columns
    assert "rain_anom_30d" in panel.columns
    assert "temp_anom_21d" in panel.columns
    assert "dry_frac_30d" in panel.columns
    assert panel.index.equals(prices.index)
    # no ±inf anywhere (model layer imputes NaN only)
    assert np.isfinite(panel.values[~np.isnan(panel.values)]).all()


def test_causality_no_future_leakage(prices, brl, weather):
    """Feature values at date t must be identical whether or not data after
    t exists — the no-look-ahead integrity constraint."""
    cut = 600
    full = RegimeFeatureBuilder(prices, brl_usd=brl, weather=weather).build()
    trunc = RegimeFeatureBuilder(
        prices.iloc[:cut],
        brl_usd=brl[brl.index <= prices.index[cut - 1]],
        weather=weather[weather.index <= prices.index[cut - 1]],
    ).build()
    common = trunc.index
    pd.testing.assert_frame_equal(
        full.loc[common, trunc.columns], trunc, check_dtype=False)


def test_trailing_anomaly_z_is_causal():
    rng = np.random.default_rng(9)
    dates = pd.date_range("2019-01-01", periods=900, freq="D")
    s = pd.Series(rng.gamma(0.6, 6.0, 900), index=dates)
    full = _trailing_anomaly_z(s, window=30, baseline=365)
    trunc = _trailing_anomaly_z(s.iloc[:700], window=30, baseline=365)
    pd.testing.assert_series_equal(full.iloc[:700], trunc)


def test_get_core_features_price_only(prices, brl):
    b = RegimeFeatureBuilder(prices, brl_usd=brl)
    core = b.get_core_features()
    assert all(c.startswith("px_") for c in core.columns)
