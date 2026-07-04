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
