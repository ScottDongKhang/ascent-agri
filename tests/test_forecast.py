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


def _outlook(issued="2026-07-06", band="low"):
    return fc.ForwardOutlook(
        issued=issued, window_start=issued, window_end="2026-07-19",
        expected_mm=10.0, norm_mm=28.0, std_mm=4.0, anom_z=-4.5,
        drought_w=0.5, wetness_w=0.1, stage_label="fruit filling",
        projected_stress=2.25, projected_band=band)


def test_snapshot_append_and_read(tmp_path):
    path = tmp_path / "weather_forecasts.jsonl"
    assert fc.append_snapshot(_outlook("2026-07-06"), path=path) is True
    assert fc.append_snapshot(_outlook("2026-07-07"), path=path) is True
    entries = fc.read_snapshots(path)
    assert [e["date_issued"] for e in entries] == ["2026-07-06", "2026-07-07"]
    assert entries[0]["schema"] == 1
    assert entries[0]["projected_band"] == "low"
    assert entries[0]["drought_w"] == 0.5      # weights stored for scoring


def test_snapshot_append_is_idempotent_per_day(tmp_path):
    path = tmp_path / "weather_forecasts.jsonl"
    fc.append_snapshot(_outlook("2026-07-06"), path=path)
    before = path.read_text()
    assert fc.append_snapshot(_outlook("2026-07-06", band="severe"),
                              path=path) is False
    assert path.read_text() == before          # never edited, never duplicated


def _snapshot(issued, start, end, expected, norm=28.0, std=4.0, band="low",
              d_w=0.5, w_w=0.1):
    return {"schema": 1, "date_issued": issued, "window_start": start,
            "window_end": end, "expected_mm": expected, "norm_mm": norm,
            "std_mm": std, "anom_z": (expected - norm) / std,
            "drought_w": d_w, "wetness_w": w_w,
            "stage_label": "fruit filling", "projected_stress": 0.0,
            "projected_band": band}


def test_score_snapshots_math():
    """Realized 28mm on both windows. Forecast said 24 and 36 → MAE 6;
    climatology (norm 30) → MAE 2. Skill = 1 − 6/2 = −2, published as-is."""
    rain = pd.Series(2.0, index=pd.date_range("2026-01-01", "2026-03-01"))
    entries = [
        _snapshot("2026-01-05", "2026-01-05", "2026-01-18", 24.0, norm=30.0),
        _snapshot("2026-01-19", "2026-01-19", "2026-02-01", 36.0, norm=30.0),
    ]
    ver = fc.score_snapshots(entries, rain=rain)
    assert ver.n_snapshots == 2 and ver.n_closed == 2
    assert ver.mae_forecast_mm == pytest.approx(6.0)     # |24-28|, |36-28| → 4,8
    assert ver.mae_climatology_mm == pytest.approx(2.0)  # |30-28| both
    assert ver.bias_mm == pytest.approx(2.0)             # (−4 + 8) / 2
    assert ver.skill == pytest.approx(1 - 6.0 / 2.0)     # negative — published
    assert 0.0 <= ver.band_hit_rate <= 1.0


def test_score_snapshots_open_window_not_scored():
    rain = pd.Series(2.0, index=pd.date_range("2026-01-01", "2026-01-10"))
    entries = [_snapshot("2026-01-05", "2026-01-05", "2026-01-18", 24.0)]
    ver = fc.score_snapshots(entries, rain=rain)
    assert ver.n_snapshots == 1 and ver.n_closed == 0
    assert ver.first_scoreable == "2026-01-24"    # window_end + 6-day lag
    assert "not yet scoreable" in ver.summary_line()


def test_score_snapshots_empty():
    ver = fc.score_snapshots([], rain=pd.Series(dtype=float))
    assert ver.n_snapshots == 0 and ver.n_closed == 0
    assert ver.first_scoreable is None
