"""Tests for the 14-day forecast layer: parsing, cache, norm, outlook,
snapshot ledger, and verification. Synthetic data only — no network."""
import json

import numpy as np
import pandas as pd
import pytest

from ascentagri.agronomy import forecast as fc


def test_frame_from_payload_parses_daily_block():
    payload = {"daily": {
        "time": ["2026-07-06", "2026-07-07", "2026-07-08"],
        "precipitation_sum": [1.5, None, 12.0],
    }}
    df = fc._frame_from_payload(payload)
    assert list(df.columns) == ["rain_mm"]
    assert df.index[0] == pd.Timestamp("2026-07-06")
    assert df["rain_mm"].iloc[0] == 1.5
    assert np.isnan(df["rain_mm"].iloc[1])
    assert df["rain_mm"].iloc[2] == 12.0


def test_frame_from_payload_empty_raises():
    with pytest.raises(RuntimeError):
        fc._frame_from_payload({"daily": {}})


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(fc, "FORECAST_CACHE", tmp_path / "forecast.csv")
    df = pd.DataFrame(
        {"rain_mm": [0.0, 3.5]},
        index=pd.to_datetime(["2026-07-06", "2026-07-07"]))
    df.index.name = "date"
    fc.write_cache(df, issued="2026-07-06")
    loaded = fc.load_forecast()
    assert loaded is not None
    got, issued = loaded
    assert issued == pd.Timestamp("2026-07-06")
    assert list(got["rain_mm"]) == [0.0, 3.5]


def test_load_forecast_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(fc, "FORECAST_CACHE", tmp_path / "absent.csv")
    assert fc.load_forecast() is None


def _history(rain_by_month=None, year_drift=0.0):
    """Synthetic daily history: constant mm/day, optionally varying by month.
    year_drift adds a small per-year increment so the norm's std is nonzero."""
    idx = pd.date_range("2019-01-01", "2026-07-05", freq="D")
    vals = np.full(len(idx), 2.0)
    if rain_by_month:
        for m, v in rain_by_month.items():
            vals[idx.month == m] = v
    vals = vals + year_drift * (idx.year.values - 2019)
    df = pd.DataFrame({"rain_mm": vals}, index=idx)
    df.index.name = "date"
    return df


def test_seasonal_norm_constant_history():
    hist = _history()
    norm, std, n = fc.seasonal_norm(hist["rain_mm"], pd.Timestamp("2026-07-06"))
    assert norm == pytest.approx(28.0)      # 14 days × 2.0 mm
    assert std == pytest.approx(0.0)
    assert n >= 3


def test_seasonal_norm_too_few_years_raises():
    idx = pd.date_range("2025-01-01", "2026-07-05", freq="D")
    rain = pd.Series(1.0, index=idx)
    with pytest.raises(ValueError):
        fc.seasonal_norm(rain, pd.Timestamp("2026-07-06"))


def _forecast_frame(start="2026-07-06", days=16, mm_per_day=0.5):
    idx = pd.date_range(start, periods=days, freq="D")
    df = pd.DataFrame({"rain_mm": np.full(days, mm_per_day)}, index=idx)
    df.index.name = "date"
    return df


def test_compute_outlook_dry_july_is_stressed():
    """July (fruit filling, drought_weight 0.5): a big deficit → stress.
    year_drift makes the norm's std nonzero so the z-score is defined."""
    hist = _history(rain_by_month={7: 8.0}, year_drift=0.01)
    out = fc.compute_outlook(_forecast_frame(mm_per_day=0.0), hist,
                             issued="2026-07-06")
    assert out.window_start == "2026-07-06"
    assert out.window_end == "2026-07-19"
    assert out.expected_mm == pytest.approx(0.0)
    assert out.anom_z < 0
    assert out.projected_stress > 0
    assert out.projected_band in {"low", "watch", "elevated", "severe"}
    assert out.stage_label == "fruit filling"


def test_compute_outlook_stage_boundary_mixes_weights():
    """A window spanning Mar 25 → Apr 7 mixes flowering (1.0 drought weight)
    and early fruit (0.8): the mean weight must sit strictly between."""
    hist = _history()
    out = fc.compute_outlook(_forecast_frame(start="2026-03-25"), hist,
                             issued="2026-03-25")
    assert 0.8 < out.drought_w < 1.0
    assert out.wetness_w == pytest.approx(0.0)


def test_compute_outlook_zero_std_means_zero_z():
    hist = _history()                                 # constant → std 0
    out = fc.compute_outlook(_forecast_frame(mm_per_day=9.9), hist,
                             issued="2026-07-06")
    assert out.anom_z == 0.0


def test_compute_outlook_short_forecast_raises():
    hist = _history()
    with pytest.raises(ValueError):
        fc.compute_outlook(_forecast_frame(days=5), hist)
