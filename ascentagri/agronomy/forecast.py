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


# ── seasonal norm + forward outlook ─────────────────────────────────────────

def seasonal_norm(rain: pd.Series, window_start: pd.Timestamp,
                  days: int = FORECAST_WINDOW_DAYS) -> "tuple[float, float, int]":
    """Mean/std of total rainfall over the same calendar window in every
    complete prior year of the history. Same-source norm (Open-Meteo history)
    so forecast and norm are comparable."""
    rain = rain.dropna().sort_index()
    window_start = pd.Timestamp(window_start)
    totals = []
    for year in range(int(rain.index[0].year), int(window_start.year)):
        try:
            start_y = window_start.replace(year=year)
        except ValueError:                    # Feb 29 in a non-leap year
            start_y = window_start.replace(year=year, day=28)
        window = rain.loc[start_y:start_y + pd.Timedelta(days=days - 1)]
        if len(window) < days:                # incomplete coverage — skip year
            continue
        totals.append(float(window.sum()))
    if len(totals) < MIN_NORM_YEARS:
        raise ValueError(
            f"seasonal norm needs >= {MIN_NORM_YEARS} complete prior-year "
            f"windows at {window_start.date()}, got {len(totals)}")
    arr = np.asarray(totals)
    return float(arr.mean()), float(arr.std(ddof=1)), len(totals)


@dataclass(frozen=True)
class ForwardOutlook:
    issued: str            # YYYY-MM-DD the forecast was issued
    window_start: str
    window_end: str
    expected_mm: float
    norm_mm: float
    std_mm: float
    anom_z: float
    drought_w: float       # day-weighted phenology weights over the window —
    wetness_w: float       #   stored so verification is reproducible later
    stage_label: str       # the stage covering most of the window
    projected_stress: float
    projected_band: str


def _stress_from(z: float, drought_w: float, wetness_w: float) -> float:
    return drought_w * max(0.0, -z) + wetness_w * max(0.0, z)


def compute_outlook(forecast: pd.DataFrame, history: pd.DataFrame,
                    days: int = FORECAST_WINDOW_DAYS,
                    issued: "str | None" = None) -> ForwardOutlook:
    """The 14-day forward read: expected rainfall vs seasonal norm, weighted
    by which crop stage the window lands on (day-weighted across boundaries)."""
    window = forecast["rain_mm"].sort_index().iloc[:days]
    if len(window) < days:
        raise ValueError(f"forecast has {len(window)} days; need {days}")
    expected = float(window.fillna(0.0).sum())
    norm, std, _ = seasonal_norm(history["rain_mm"], window.index[0], days)
    z = 0.0 if std < 1e-9 else (expected - norm) / std
    day_stages = [stage_for(d) for d in window.index]
    d_w = float(np.mean([s.drought_weight for s in day_stages]))
    w_w = float(np.mean([s.wetness_weight for s in day_stages]))
    labels = [s.label for s in day_stages]
    stress = _stress_from(z, d_w, w_w)
    return ForwardOutlook(
        issued=issued or str(dt.date.today()),
        window_start=str(window.index[0].date()),
        window_end=str(window.index[-1].date()),
        expected_mm=expected, norm_mm=norm, std_mm=std, anom_z=float(z),
        drought_w=d_w, wetness_w=w_w,
        stage_label=max(set(labels), key=labels.count),
        projected_stress=float(stress), projected_band=stress_label(stress))


# ── the snapshot ledger (append-only, committed) ────────────────────────────

def read_snapshots(path: Path = SNAPSHOT_PATH) -> List[Dict]:
    if not path.exists():
        return []
    entries = [json.loads(line) for line in path.read_text().splitlines()
               if line.strip()]
    return sorted(entries, key=lambda e: e["date_issued"])


def append_snapshot(outlook: ForwardOutlook, path: Path = SNAPSHOT_PATH) -> bool:
    """Write today's issued forecast down before the outcome is known.
    One entry per issue date; existing lines are never touched."""
    entry = {"schema": 1, "date_issued": outlook.issued}
    entry.update({k: v for k, v in asdict(outlook).items() if k != "issued"})
    if any(e["date_issued"] == outlook.issued for e in read_snapshots(path)):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    return True


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["fetch", "score"])
    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        fcst = fetch_forecast()
        write_cache(fcst)
        outlook = compute_outlook(fcst, load_weather())
        added = append_snapshot(outlook)
        print(f"[forecast] cached {len(fcst)} days; snapshot "
              f"{'appended' if added else 'already present'} for "
              f"{outlook.issued}: {outlook.expected_mm:.0f}mm vs norm "
              f"{outlook.norm_mm:.0f}mm ({outlook.anom_z:+.1f} sigma) -> "
              f"{outlook.projected_band}")
    else:
        print(score_snapshots().summary_line())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
