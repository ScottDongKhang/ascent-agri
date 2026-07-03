"""Phenology: same statistical anomaly, different biology by crop stage."""
import numpy as np
import pandas as pd
import pytest

from ascentagri.agronomy.economics import (
    futures_usd_tonne_to_vnd_kg,
    transmission_line,
)
from ascentagri.agronomy.phenology import (
    crop_stress_index,
    stage_for,
    stress_label,
)


def test_calendar_covers_all_months():
    for m in range(1, 13):
        s = stage_for(pd.Timestamp(2026, m, 15))
        assert s.name in {"flowering", "early_fruit", "fruit_filling", "harvest"}


def test_stage_boundaries():
    assert stage_for(pd.Timestamp("2026-02-10")).name == "flowering"
    assert stage_for(pd.Timestamp("2026-04-10")).name == "early_fruit"
    assert stage_for(pd.Timestamp("2026-07-10")).name == "fruit_filling"
    assert stage_for(pd.Timestamp("2026-11-10")).name == "harvest"


def test_dry_anomaly_hurts_most_at_flowering():
    """-2σ dry in February (flowering) must score higher than the identical
    anomaly in August (filling) and much higher than in November (harvest)."""
    dates = pd.DatetimeIndex(["2026-02-15", "2026-08-15", "2026-11-15"])
    z = pd.Series([-2.0, -2.0, -2.0], index=dates)
    stress = crop_stress_index(z)
    feb, aug, nov = stress.values
    assert feb == pytest.approx(2.0)
    assert aug == pytest.approx(1.0)
    assert nov == pytest.approx(0.2)
    assert feb > aug > nov


def test_wet_anomaly_hurts_at_harvest_not_flowering():
    dates = pd.DatetimeIndex(["2026-02-15", "2026-11-15"])
    z = pd.Series([+2.0, +2.0], index=dates)
    stress = crop_stress_index(z)
    assert stress.iloc[0] == pytest.approx(0.0)   # wet flowering: no penalty
    assert stress.iloc[1] == pytest.approx(1.6)   # wet harvest: mold/delays


def test_neutral_weather_scores_zero():
    dates = pd.date_range("2026-01-01", periods=365, freq="D")
    z = pd.Series(0.0, index=dates)
    assert (crop_stress_index(z) == 0).all()


def test_nan_anomaly_propagates_not_crashes():
    dates = pd.DatetimeIndex(["2026-02-15"])
    z = pd.Series([np.nan], index=dates)
    out = crop_stress_index(z)
    assert np.isnan(out.iloc[0])
    assert stress_label(float(out.iloc[0])) == "unknown"


def test_stress_labels():
    assert stress_label(0.2) == "low"
    assert stress_label(0.7) == "watch"
    assert stress_label(1.5) == "elevated"
    assert stress_label(2.5) == "severe"


def test_futures_to_vnd_per_kg():
    # 4,000 USD/tonne at 26,000 VND/USD = 104,000 VND/kg
    assert futures_usd_tonne_to_vnd_kg(4000, 26000) == pytest.approx(104_000)


def test_transmission_line_reads_sanely():
    line = transmission_line(4000, 26000, chg_1m=-0.05)
    assert "104,000" in line
    assert "đồng/kg" in line
    assert "less" in line          # price fell → worth less to a grower
    line2 = transmission_line(4000, 26000)
    assert line2.endswith(".")
