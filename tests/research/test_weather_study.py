"""Weather study: planted effects must be detected, placebos must not."""
import numpy as np
import pandas as pd
import pytest

from ascentagri.research.weather_study import (
    detect_threshold_events,
    event_study,
    forward_returns,
    lead_lag,
    rain_anomaly,
)


def _mk_weather(n=1500, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    rain = rng.gamma(0.7, 5.0, n)
    return pd.DataFrame({"rain_mm": rain}, index=dates), dates


def test_detect_events_cooldown():
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    s = pd.Series(0.0, index=dates)
    s.iloc[50:60] = -2.0     # one dry episode
    s.iloc[62:70] = -2.0     # brief re-cross inside cooldown → NOT a new event
    s.iloc[150:155] = -2.0   # after cooldown → new event
    events = detect_threshold_events(s, -1.25, below=True, cooldown_days=30)
    assert len(events) == 2
    assert events[0] == dates[50]
    assert events[1] == dates[150]


def test_forward_returns_alignment():
    dates = pd.bdate_range("2024-01-01", periods=10)
    close = pd.Series(np.arange(100.0, 110.0), index=dates)
    fwd = forward_returns(close, 2)
    assert fwd.iloc[0] == pytest.approx(102 / 100 - 1)
    assert np.isnan(fwd.iloc[-1])


def test_event_study_detects_planted_effect():
    """Prices jump +8% in the 5 days after each planted event — the study
    must find a large positive excess with a small p-value."""
    rng = np.random.default_rng(3)
    n = 1200
    dates = pd.bdate_range("2020-01-01", periods=n)
    rets = rng.normal(0.0, 0.01, n)
    event_pos = np.arange(100, n - 80, 90)
    for pos in event_pos:
        rets[pos + 1: pos + 6] += 0.016          # ~+8% over 5 days
    close = pd.Series(100 * np.exp(np.cumsum(rets)), index=dates)
    events = pd.DatetimeIndex(dates[event_pos])
    res = event_study(close, events, horizons=(5,), n_perm=500)
    assert res["5d"]["n"] == len(event_pos)
    assert res["5d"]["excess"] > 0.05
    assert res["5d"]["p_value"] < 0.02


def test_event_study_placebo_is_flat():
    rng = np.random.default_rng(4)
    n = 1200
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=dates)
    events = pd.DatetimeIndex(dates[np.arange(100, n - 80, 90)])
    res = event_study(close, events, horizons=(5,), n_perm=500)
    assert res["5d"]["p_value"] > 0.05           # nothing planted → not significant
    assert abs(res["5d"]["excess"]) < 0.02


def test_lead_lag_sign_on_planted_relationship():
    """Low anomaly → high forward returns (planted) must give a negative
    Spearman with a CI excluding zero."""
    rng = np.random.default_rng(5)
    n = 1000
    dates = pd.bdate_range("2020-01-01", periods=n)
    anom = pd.Series(rng.normal(0, 1, n), index=dates).rolling(10, min_periods=1).mean()
    rets = 0.004 * (-anom.shift(1).fillna(0)) + rng.normal(0, 0.003, n)
    close = pd.Series(100 * np.exp(np.cumsum(rets)), index=dates)
    res = lead_lag(anom, close, horizons=(21,), n_boot=300)
    assert res["21d"]["spearman"] < -0.1
    assert res["21d"]["ci_hi"] < 0


def test_lead_lag_placebo_ci_contains_zero():
    rng = np.random.default_rng(6)
    n = 1000
    dates = pd.bdate_range("2020-01-01", periods=n)
    anom = pd.Series(rng.normal(0, 1, n), index=dates).rolling(10, min_periods=1).mean()
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=dates)
    res = lead_lag(anom, close, horizons=(21,), n_boot=300)
    assert res["21d"]["ci_lo"] <= 0 <= res["21d"]["ci_hi"]


def test_rain_anomaly_is_causal():
    wx, _ = _mk_weather()
    full = rain_anomaly(wx)
    trunc = rain_anomaly(wx.iloc[:1000])
    pd.testing.assert_series_equal(full.iloc[:1000], trunc)
