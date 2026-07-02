"""Fetch the two external regime inputs: BRL/USD and a Vietnam growing-region
weather proxy. Both are free, keyless, documented public APIs.

1. BRL/USD — FRED series DEXBZUS ("Brazilian Reals to One U.S. Dollar"),
   via the keyless fredgraph.csv endpoint. Brazil is the dominant coffee
   producer; a weaker BRL (DEXBZUS up) raises Brazilian producers' local-currency
   revenue per exported bag and is a well-documented bearish driver of world
   coffee prices.

2. Weather — Open-Meteo Historical Weather API (archive-api.open-meteo.com),
   daily precipitation and mean temperature at Buon Ma Thuot (12.68N, 108.04E),
   Dak Lak province — the center of Vietnam's Central Highlands robusta belt.
   Free, no API key, daily history from 1940.

Usage:
    python -m ascentagri.macro_fetch            # writes both caches
    python -m ascentagri.macro_fetch --force    # re-fetch even if cached

Caches (git-ignored, reproducible):
    data/processed/brlusd.csv                    [date, brl_per_usd]
    data/processed/weather_central_highlands.csv [date, rain_mm, temp_c]
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import logging
import urllib.request
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

BRLUSD_PATH = PROCESSED / "brlusd.csv"
WEATHER_PATH = PROCESSED / "weather_central_highlands.csv"

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXBZUS"

# Buon Ma Thuot, Dak Lak — heart of the Central Highlands robusta belt
WEATHER_LAT = 12.68
WEATHER_LON = 108.04
OPEN_METEO_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude={lat}&longitude={lon}"
    "&start_date={start}&end_date={end}"
    "&daily=precipitation_sum,temperature_2m_mean&timezone=UTC"
)

DEFAULT_START = "2017-01-01"   # 1y of runway before the earliest price data


def _http_get(url: str, timeout: int = 60, retries: int = 3) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    )
    last_exc: Exception = RuntimeError("unreachable")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:   # timeouts, transient 5xx
            last_exc = exc
            log.warning("fetch attempt %d/%d failed for %s: %s",
                        attempt + 1, retries, url.split("?")[0], exc)
    raise last_exc


def _fetch_brlusd_fred(start: str) -> pd.DataFrame:
    raw = _http_get(FRED_CSV_URL, timeout=30, retries=1)
    df = pd.read_csv(io.BytesIO(raw))
    # fredgraph.csv columns: observation_date, DEXBZUS
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    value_col = [c for c in df.columns if c != date_col][0]
    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col]),
        "brl_per_usd": pd.to_numeric(df[value_col], errors="coerce"),
    }).dropna()
    out = out[out["date"] >= pd.Timestamp(start)].reset_index(drop=True)
    if out.empty:
        raise RuntimeError("FRED DEXBZUS returned no rows after start filter")
    out.attrs["source"] = "FRED DEXBZUS"
    return out


def _fetch_brlusd_yahoo(start: str) -> pd.DataFrame:
    """Fallback: Yahoo Finance BRL=X (USD/BRL — same orientation as DEXBZUS)."""
    import yfinance as yf
    df = yf.download("BRL=X", start=start, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError("yfinance BRL=X returned no rows")
    close = df["Close"]
    if hasattr(close, "columns"):          # MultiIndex ticker column
        close = close.iloc[:, 0]
    out = pd.DataFrame({
        "date": pd.to_datetime(close.index).tz_localize(None),
        "brl_per_usd": pd.to_numeric(close.values, errors="coerce"),
    }).dropna().reset_index(drop=True)
    out.attrs["source"] = "Yahoo Finance BRL=X (yfinance)"
    return out


def fetch_brlusd(start: str = DEFAULT_START) -> pd.DataFrame:
    """Fetch BRL per USD. FRED DEXBZUS primary, Yahoo BRL=X fallback.
    Returns tidy [date, brl_per_usd] with .attrs['source'] set."""
    try:
        return _fetch_brlusd_fred(start)
    except Exception as exc:
        log.warning("FRED DEXBZUS unavailable (%s) — falling back to Yahoo BRL=X", exc)
        return _fetch_brlusd_yahoo(start)


def fetch_weather(start: str = DEFAULT_START, end: str | None = None) -> pd.DataFrame:
    """Fetch daily precipitation + mean temperature for Buon Ma Thuot.
    Returns tidy [date, rain_mm, temp_c]."""
    # archive API lags realtime by ~5 days
    end = end or (dt.date.today() - dt.timedelta(days=6)).isoformat()
    url = OPEN_METEO_URL.format(lat=WEATHER_LAT, lon=WEATHER_LON, start=start, end=end)
    payload = json.loads(_http_get(url))
    daily = payload.get("daily", {})
    if not daily.get("time"):
        raise RuntimeError(f"Open-Meteo returned no daily data: {payload}")
    out = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "rain_mm": daily["precipitation_sum"],
        "temp_c": daily["temperature_2m_mean"],
    })
    out["rain_mm"] = pd.to_numeric(out["rain_mm"], errors="coerce")
    out["temp_c"] = pd.to_numeric(out["temp_c"], errors="coerce")
    return out


def load_brlusd() -> pd.Series:
    """Load cached BRL/USD as a date-indexed Series (fetches if missing)."""
    if not BRLUSD_PATH.exists():
        ensure_caches()
    df = pd.read_csv(BRLUSD_PATH, parse_dates=["date"])
    return df.set_index("date")["brl_per_usd"].sort_index()


def load_weather() -> pd.DataFrame:
    """Load cached weather as a date-indexed DataFrame (fetches if missing)."""
    if not WEATHER_PATH.exists():
        ensure_caches()
    df = pd.read_csv(WEATHER_PATH, parse_dates=["date"])
    return df.set_index("date").sort_index()


def ensure_caches(force: bool = False) -> None:
    """Fetch and cache both series if missing (or force=True)."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    brl_source = "FRED DEXBZUS (or Yahoo BRL=X fallback)"
    if force or not BRLUSD_PATH.exists():
        brl = fetch_brlusd()
        brl_source = brl.attrs.get("source", brl_source)
        brl.to_csv(BRLUSD_PATH, index=False)
        print(f"[macro_fetch] wrote {BRLUSD_PATH.name}: {len(brl)} rows "
              f"({brl['date'].min().date()} -> {brl['date'].max().date()}) "
              f"source={brl_source}")
    if force or not WEATHER_PATH.exists():
        wx = fetch_weather()
        wx.to_csv(WEATHER_PATH, index=False)
        print(f"[macro_fetch] wrote {WEATHER_PATH.name}: {len(wx)} rows "
              f"({wx['date'].min().date()} -> {wx['date'].max().date()})")
    _append_provenance(brl_source)


def _append_provenance(brl_source: str) -> None:
    prov = PROCESSED / "PROVENANCE.md"
    marker = "brlusd.csv"
    existing = prov.read_text() if prov.exists() else "# data/processed provenance\n"
    if marker in existing:
        return
    today = dt.date.today().isoformat()
    existing += (
        f"\n- `brlusd.csv` — BRL per USD. Source: {brl_source}. Fetched {today}.\n"
        f"- `weather_central_highlands.csv` — daily precipitation (mm) and mean "
        f"temperature (C) at Buon Ma Thuot, Dak Lak, Vietnam ({WEATHER_LAT}N, "
        f"{WEATHER_LON}E) from the Open-Meteo Historical Weather API (free, "
        f"keyless). Fetched {today}.\n"
    )
    prov.write_text(existing)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-fetch even if cached")
    args = ap.parse_args()
    ensure_caches(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
