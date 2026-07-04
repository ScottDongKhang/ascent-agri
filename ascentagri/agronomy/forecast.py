"""The 14-day forward look: Open-Meteo rainfall forecast for the robusta belt,
compared against a same-source seasonal norm and weighted by crop phenology.

Every issued forecast is snapshotted to an append-only ledger
(data/ledger/weather_forecasts.jsonl) BEFORE the outcome is known, and scored
against realized rainfall once the window closes — skill vs climatology,
published whatever it says. Same discipline as the position ledger.

Usage:
    python -m ascentagri.agronomy.forecast fetch   # cache + snapshot (workflow)
    python -m ascentagri.agronomy.forecast score   # verification from committed files
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ascentagri.macro_fetch import (
    PROCESSED, ROOT, WEATHER_LAT, WEATHER_LON, _http_get, load_weather)

from .phenology import stage_for, stress_label

log = logging.getLogger(__name__)

FORECAST_WINDOW_DAYS = 14      # the scored 14-day outlook
FORECAST_HORIZON_DAYS = 16     # what the API is asked for (free tier max)
MIN_NORM_YEARS = 3             # complete prior-year windows required for a norm
MIN_VERIFIED_WINDOWS = 5       # closed windows before the site shows the table
ARCHIVE_LAG_DAYS = 6           # Open-Meteo archive lags realtime by ~5 days

FORECAST_CACHE = PROCESSED / "forecast_central_highlands.csv"
SNAPSHOT_PATH = ROOT / "data" / "ledger" / "weather_forecasts.jsonl"

FORECAST_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude={lat}&longitude={lon}"
    "&daily=precipitation_sum&forecast_days={days}&timezone=UTC"
)


# ── fetch + cache ───────────────────────────────────────────────────────────

def _frame_from_payload(payload: dict) -> pd.DataFrame:
    """Open-Meteo daily payload → date-indexed frame with one rain_mm column."""
    daily = payload.get("daily", {})
    if not daily.get("time"):
        raise RuntimeError(f"Open-Meteo forecast returned no daily data: {payload}")
    out = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "rain_mm": pd.to_numeric(daily["precipitation_sum"], errors="coerce"),
    })
    return out.set_index("date").sort_index()


def fetch_forecast(lat: float = WEATHER_LAT, lon: float = WEATHER_LON,
                   days: int = FORECAST_HORIZON_DAYS) -> pd.DataFrame:
    """Fetch the daily rainfall forecast at the Buon Ma Thuot grid point."""
    url = FORECAST_URL.format(lat=lat, lon=lon, days=days)
    return _frame_from_payload(json.loads(_http_get(url)))


def write_cache(fc: pd.DataFrame, issued: "str | None" = None) -> None:
    out = fc.reset_index()
    out["issued"] = issued or str(dt.date.today())
    FORECAST_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(FORECAST_CACHE, index=False)


def load_forecast() -> "tuple[pd.DataFrame, pd.Timestamp] | None":
    """(forecast frame, issue date) from the cache; None if never fetched."""
    if not FORECAST_CACHE.exists():
        return None
    df = pd.read_csv(FORECAST_CACHE, parse_dates=["date", "issued"])
    issued = df["issued"].iloc[0]
    return df.set_index("date")[["rain_mm"]].sort_index(), issued
