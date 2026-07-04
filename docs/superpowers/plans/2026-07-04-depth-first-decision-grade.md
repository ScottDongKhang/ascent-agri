# Depth-First Decision-Grade Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 14-day rainfall forecast layer (with an append-only snapshot ledger), public forecast verification vs climatology, a printable weekly one-pager, and machine-readable change alerts to the Robusta Coffee Monitor.

**Architecture:** One new module `ascentagri/agronomy/forecast.py` owns everything forecast-shaped (fetch, cache, seasonal norm, projected stress, snapshot ledger, verification scoring). `site/build_site.py` consumes it read-only at build time (never fetches — existing discipline), gains a forecast section, brief sentences (EN/VI), an API block, `api/changes.json`, and `alerts.xml`. A new `site/onepager.py` renders the print-styled weekly brief. The daily GitHub workflow gains one non-blocking fetch step and commits the snapshot ledger alongside the position ledger.

**Tech Stack:** Python (3.9-compatible syntax — the dev machine is 3.9, CI is 3.11), pandas/numpy, matplotlib (Agg), pytest, Open-Meteo forecast API (free, keyless), GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-07-04-depth-first-decision-grade-design.md`

## Global Constraints

- **3.9-compatible typing:** `from __future__ import annotations` at top of every new module; string-quoted unions in signatures (`"str | None"`) exactly as `macro_fetch.py` does.
- **build_site.py NEVER fetches.** All network I/O lives in `ascentagri/` fetchers invoked by the workflow.
- **Fail-safe publishing:** the forecast layer is optional everywhere — any failure logs a warning and the site builds without it. The core fetches keep their all-or-nothing gate.
- **Append-only ledgers:** `data/ledger/weather_forecasts.jsonl` entries are never edited or deleted; one entry per `date_issued`; scoring reads only committed files + the weather cache.
- **No network in tests.** Synthetic fixtures only. Full suite must stay green: `pytest -q` (currently 133 tests).
- **Honest labeling:** every forecast surface names the source ("Open-Meteo forecast model"), carries its issue date, and says forecast skill is unverified until scored. Verification numbers are published whatever they say, including negative skill.
- **No strategy/backtest changes.** Nothing in `ascentagri/alpha`, `ascentagri/backtest`, `ascentagri/regime` is touched.
- **Constants:** forecast window = 14 days, fetch horizon = 16 days, norm requires ≥ 3 complete prior-year windows, forecast cache stale after 3 days, verification panel needs ≥ 5 closed windows, `changes.json` carries the 20 most recent changes (newest first).
- Commit after every task with the message given in the task's final step.

---

### Task 1: Forecast fetch, payload parsing, and cache

**Files:**
- Create: `ascentagri/agronomy/forecast.py`
- Test: `tests/test_forecast.py`

**Interfaces:**
- Consumes: `ascentagri.macro_fetch._http_get`, `WEATHER_LAT`, `WEATHER_LON`, `PROCESSED`
- Produces (used by Tasks 2–5):
  - `FORECAST_WINDOW_DAYS: int = 14`, `FORECAST_HORIZON_DAYS: int = 16`
  - `_frame_from_payload(payload: dict) -> pd.DataFrame` (date-indexed, one `rain_mm` column)
  - `fetch_forecast(lat: float = WEATHER_LAT, lon: float = WEATHER_LON, days: int = FORECAST_HORIZON_DAYS) -> pd.DataFrame`
  - `write_cache(fc: pd.DataFrame, issued: "str | None" = None) -> None`
  - `load_forecast() -> "tuple[pd.DataFrame, pd.Timestamp] | None"` — `(date-indexed df with rain_mm, issued timestamp)`, `None` if no cache
  - `FORECAST_CACHE: Path` = `data/processed/forecast_central_highlands.csv`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_forecast.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_forecast.py -q`
Expected: FAIL / collection error — `cannot import name 'forecast'`.

- [ ] **Step 3: Write the module**

Create `ascentagri/agronomy/forecast.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_forecast.py -q`
Expected: 4 passed.

- [ ] **Step 5: Run the whole suite, then commit**

Run: `pytest -q` — expected: all pass.

```bash
git add ascentagri/agronomy/forecast.py tests/test_forecast.py
git commit -m "feat: forecast fetch + cache for the 14-day robusta-belt outlook"
```

---

### Task 2: Seasonal norm and the ForwardOutlook computation

**Files:**
- Modify: `ascentagri/agronomy/forecast.py` (append after the cache block)
- Test: `tests/test_forecast.py` (append)

**Interfaces:**
- Consumes: `stage_for`, `stress_label` (already imported), Task 1's frames
- Produces (used by Tasks 3–5):
  - `seasonal_norm(rain: pd.Series, window_start: pd.Timestamp, days: int = FORECAST_WINDOW_DAYS) -> "tuple[float, float, int]"` — `(norm_mm, std_mm, n_years)`; raises `ValueError` below `MIN_NORM_YEARS`
  - `@dataclass(frozen=True) ForwardOutlook` with fields: `issued: str`, `window_start: str`, `window_end: str`, `expected_mm: float`, `norm_mm: float`, `std_mm: float`, `anom_z: float`, `drought_w: float`, `wetness_w: float`, `stage_label: str`, `projected_stress: float`, `projected_band: str`
  - `compute_outlook(forecast: pd.DataFrame, history: pd.DataFrame, days: int = FORECAST_WINDOW_DAYS, issued: "str | None" = None) -> ForwardOutlook`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_forecast.py`)

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_forecast.py -q`
Expected: new tests FAIL with `AttributeError: ... has no attribute 'seasonal_norm'`.

- [ ] **Step 3: Implement** (append to `ascentagri/agronomy/forecast.py`)

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_forecast.py -q` — expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add ascentagri/agronomy/forecast.py tests/test_forecast.py
git commit -m "feat: seasonal norm + phenology-weighted ForwardOutlook"
```

---

### Task 3: Snapshot ledger + `fetch` CLI

**Files:**
- Modify: `ascentagri/agronomy/forecast.py` (append)
- Test: `tests/test_forecast.py` (append)

**Interfaces:**
- Consumes: `ForwardOutlook`, `fetch_forecast`, `write_cache`, `load_weather`
- Produces (used by Tasks 4, 5, 7 and the workflow):
  - `read_snapshots(path: Path = SNAPSHOT_PATH) -> List[Dict]` (sorted by `date_issued`)
  - `append_snapshot(outlook: ForwardOutlook, path: Path = SNAPSHOT_PATH) -> bool` (False if that issue date already exists)
  - CLI: `python -m ascentagri.agronomy.forecast fetch`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_forecast.py`)

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_forecast.py -q`
Expected: FAIL with `AttributeError: ... 'append_snapshot'`.

- [ ] **Step 3: Implement** (append to `ascentagri/agronomy/forecast.py`)

```python
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
```

And the CLI at the very end of the file (`score` arrives in Task 4 — wire the
choice now, with a stub that explains itself):

```python
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
```

(`score_snapshots` does not exist yet — Task 4 adds it. The `fetch` path must
not reference it, so this file is importable and `fetch` runnable now; running
`score` before Task 4 raises `NameError`, which Task 4 resolves.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_forecast.py -q` — expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add ascentagri/agronomy/forecast.py tests/test_forecast.py
git commit -m "feat: append-only weather-forecast snapshot ledger + fetch CLI"
```

---

### Task 4: Verification scoring — skill vs climatology

**Files:**
- Modify: `ascentagri/agronomy/forecast.py` (insert before the CLI block)
- Test: `tests/test_forecast.py` (append)

**Interfaces:**
- Consumes: `read_snapshots`, `load_weather`, `_stress_from`, `stress_label`
- Produces (used by Tasks 5–6):
  - `@dataclass ForecastVerification` with fields: `n_snapshots: int`, `n_closed: int`, `mae_forecast_mm: float`, `mae_climatology_mm: float`, `bias_mm: float`, `skill: float`, `band_hit_rate: float`, `first_scoreable: Optional[str]`; method `summary_line() -> str`
  - `score_snapshots(entries: "List[Dict] | None" = None, rain: "pd.Series | None" = None) -> ForecastVerification`
  - `MIN_VERIFIED_WINDOWS = 5` (already defined in Task 1)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_forecast.py`)

```python
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
    climatology said 28 → MAE 0 is degenerate, so use norm 30 → MAE 2."""
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_forecast.py -q`
Expected: FAIL with `AttributeError: ... 'score_snapshots'`.

- [ ] **Step 3: Implement** (insert into `ascentagri/agronomy/forecast.py` before the CLI block)

```python
# ── verification: skill vs climatology ──────────────────────────────────────

@dataclass
class ForecastVerification:
    n_snapshots: int
    n_closed: int
    mae_forecast_mm: float
    mae_climatology_mm: float
    bias_mm: float                 # mean(expected − realized): + = too wet
    skill: float                   # 1 − MAE_forecast / MAE_climatology
    band_hit_rate: float           # projected band == realized band
    first_scoreable: Optional[str]

    def summary_line(self) -> str:
        if self.n_closed == 0:
            when = (f" — first window scoreable ~{self.first_scoreable}"
                    if self.first_scoreable else "")
            return (f"{self.n_snapshots} forecasts issued, none with a closed "
                    f"window yet: not yet scoreable{when}.")
        return (f"{self.n_closed}/{self.n_snapshots} windows closed · "
                f"forecast MAE {self.mae_forecast_mm:.1f}mm vs climatology "
                f"{self.mae_climatology_mm:.1f}mm · skill {self.skill:+.2f} · "
                f"bias {self.bias_mm:+.1f}mm · band hit rate "
                f"{self.band_hit_rate:.0%}")


def score_snapshots(entries: "List[Dict] | None" = None,
                    rain: "pd.Series | None" = None) -> ForecastVerification:
    """Score every issued forecast whose window has fully closed, against
    realized rainfall — reproducible from the committed snapshot ledger and
    the weather cache alone."""
    entries = read_snapshots() if entries is None else sorted(
        entries, key=lambda e: e["date_issued"])
    if rain is None:
        rain = load_weather()["rain_mm"]
    rain = rain.dropna().sort_index()

    err_f, err_c, bias, hits = [], [], [], []
    for e in entries:
        start, end = pd.Timestamp(e["window_start"]), pd.Timestamp(e["window_end"])
        window = rain.loc[start:end]
        if len(window) < (end - start).days + 1:      # window not closed yet
            continue
        realized = float(window.sum())
        err_f.append(abs(e["expected_mm"] - realized))
        err_c.append(abs(e["norm_mm"] - realized))
        bias.append(e["expected_mm"] - realized)
        std = float(e.get("std_mm", 0.0))
        rz = 0.0 if std < 1e-9 else (realized - e["norm_mm"]) / std
        realized_band = stress_label(
            _stress_from(rz, e["drought_w"], e["wetness_w"]))
        hits.append(1.0 if realized_band == e["projected_band"] else 0.0)

    n_closed = len(err_f)
    first = None
    if entries and n_closed == 0:
        first = str((pd.Timestamp(entries[0]["window_end"])
                     + pd.Timedelta(days=ARCHIVE_LAG_DAYS)).date())
    if n_closed == 0:
        return ForecastVerification(len(entries), 0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                    first)
    mae_f, mae_c = float(np.mean(err_f)), float(np.mean(err_c))
    return ForecastVerification(
        n_snapshots=len(entries), n_closed=n_closed,
        mae_forecast_mm=mae_f, mae_climatology_mm=mae_c,
        bias_mm=float(np.mean(bias)),
        skill=0.0 if mae_c < 1e-9 else 1.0 - mae_f / mae_c,
        band_hit_rate=float(np.mean(hits)),
        first_scoreable=None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_forecast.py -q` — expected: all pass.
Also run: `python -m ascentagri.agronomy.forecast score`
Expected: `0 forecasts issued, none with a closed window yet: not yet scoreable.`

- [ ] **Step 5: Commit**

```bash
git add ascentagri/agronomy/forecast.py tests/test_forecast.py
git commit -m "feat: forecast verification — skill vs climatology from committed files"
```

---

### Task 5: The forecast on the site — state, chart, section, briefs, API

**Files:**
- Modify: `site/build_site.py`
- Test: `tests/test_build_site.py` (append)

**Interfaces:**
- Consumes: `load_forecast`, `compute_outlook`, `ForwardOutlook` from Task 2
- Produces (used by Tasks 6–8):
  - `MonitorState.outlook: Optional[object] = None` (a `ForwardOutlook` when live)
  - `chart_forecast(outlook, fc_frame, out: Path) -> None`
  - `render_forecast_section(outlook, has_chart: bool) -> str`
  - `daily_brief` / `daily_brief_vi` gain one trailing sentence when `s.outlook` is set
  - `render_api_latest` gains a `"forecast"` key (object or `None`)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_build_site.py`)

```python
def _stub_outlook(band="elevated", z=-2.1):
    class O:
        issued = "2026-07-06"
        window_start = "2026-07-06"
        window_end = "2026-07-19"
        expected_mm = 38.0
        norm_mm = 92.0
        std_mm = 25.0
        anom_z = z
        drought_w = 0.5
        wetness_w = 0.1
        stage_label = "fruit filling"
        projected_stress = 1.05
        projected_band = band
    return O()


def test_forecast_section_renders_honestly():
    html = build_site.render_forecast_section(_stub_outlook(), has_chart=False)
    for required in ["The next two weeks", "38", "92", "Open-Meteo",
                     "2026-07-06", "fruit filling", "elevated"]:
        assert required in html, f"missing: {required!r}"


def test_brief_includes_forward_look():
    class P:
        posture = "defensive"
        risk_multiplier = 0.65
    base = dict(close=None, signals=None, feature_panel=None, brl=None,
                weather=None, posture=P(), label="stressed", dwell=7,
                price=250.0, chg_1w=-0.02, chg_1m=0.05,
                price_asof="", weather_asof="", brl_asof="",
                rain_z=0.1, dry_frac=0.5, brl_chg_21d=0.001)
    with_fc = build_site.MonitorState(**base, outlook=_stub_outlook())
    without = build_site.MonitorState(**base)
    b_with, b_without = build_site.daily_brief(with_fc), build_site.daily_brief(without)
    assert "Looking ahead" in b_with and "38" in b_with
    assert "Looking ahead" not in b_without
    vi = build_site.daily_brief_vi(with_fc)
    assert "14 ngày" in vi


def test_api_latest_has_forecast_key(built):
    import json
    latest = json.loads((built / "api" / "latest.json").read_text())
    assert "forecast" in latest            # object when cache present, else null
    if latest["forecast"] is not None:
        f = latest["forecast"]
        assert f["source"] == "Open-Meteo forecast model"
        assert f["projected_band"] in {"low", "watch", "elevated", "severe"}
        assert "issued" in f and "expected_rain_mm" in f
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_build_site.py -q`
Expected: the three new tests FAIL (`render_forecast_section` missing /
unexpected keyword `outlook` / `forecast` key absent).

- [ ] **Step 3: Implement in `site/build_site.py`**

3a. Add the field at the end of `MonitorState` (after `farm_gate_asof`):

```python
    outlook: Optional[object] = None  # ForwardOutlook when the cache is fresh
```

3b. In `compute_state(...)`, right before the `return MonitorState(` line,
add the optional forecast layer (same pattern as farm-gate):

```python
    # forward look (optional layer: forecast cache must exist and be fresh;
    # any failure → outlook silently absent, never a blocker)
    outlook = None
    try:
        from ascentagri.agronomy.forecast import compute_outlook, load_forecast
        loaded = load_forecast()
        if loaded is not None:
            fc_frame, issued = loaded
            age = (pd.Timestamp.today().normalize()
                   - issued.normalize()).days
            if age <= 3:
                outlook = compute_outlook(fc_frame, weather,
                                          issued=str(issued.date()))
            else:
                log.warning("forecast cache stale (%dd) — skipped", age)
    except Exception as exc:
        log.warning("forecast layer skipped: %s", exc)
```

and pass `outlook=outlook,` in the `MonitorState(` constructor call.

3c. Append the forward-look sentence at the end of `daily_brief()` (after the
BRL block, before `return`):

```python
    if s.outlook is not None:
        o = s.outlook
        parts.append(
            f"Looking ahead: the 14-day forecast calls for "
            f"{o.expected_mm:.0f} mm of rain in the robusta belt against a "
            f"{o.norm_mm:.0f} mm seasonal norm ({o.anom_z:+.1f}σ) — projected "
            f"crop stress: {o.projected_band} (Open-Meteo forecast model, "
            f"issued {o.issued}; skill scored publicly as windows close).")
```

3d. Same for `daily_brief_vi()` before its `return` (find the function's final
`return " ".join(parts)` and insert above it):

```python
    if s.outlook is not None:
        o = s.outlook
        parts.append(
            f"Dự báo 14 ngày tới: khoảng {o.expected_mm:.0f} mm mưa so với "
            f"mức trung bình mùa vụ {o.norm_mm:.0f} mm ({o.anom_z:+.1f}σ) — "
            f"mức căng thẳng dự kiến cho cây: "
            f"{STRESS_VI.get(o.projected_band, o.projected_band)}.")
```

3e. Add the chart + section (place after `render_brazil_section`):

```python
def chart_forecast(outlook, fc_frame: pd.DataFrame, out: Path):
    """Daily forecast rainfall bars with the seasonal-norm daily average."""
    days = fc_frame["rain_mm"].iloc[:14]
    fig, ax = plt.subplots(figsize=(9.6, 3.2))
    ax.bar(days.index, days.values, width=0.7, color=BLUE,
           label="forecast rain (mm/day)")
    ax.axhline(outlook.norm_mm / 14.0, color=YELLOW, lw=1.4, ls="--",
               label="seasonal norm (daily avg)")
    ax.set_ylabel("mm/day")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    _save(fig, out)


def render_forecast_section(outlook, has_chart: bool) -> str:
    o = outlook
    chart_html = ('<figure><img src="assets/forecast.png" '
                  'alt="14-day rainfall forecast vs seasonal norm"></figure>'
                  if has_chart else "")
    return f"""
<section>
  <h2>The next two weeks</h2>
  <p class="asof">Open-Meteo forecast model · issued {o.issued} ·
  window {o.window_start} → {o.window_end} · crop stage: {o.stage_label}</p>
  <p class="asof"><strong style="color:var(--ink)">{o.expected_mm:.0f} mm
  expected vs {o.norm_mm:.0f} mm seasonal norm ({o.anom_z:+.1f}σ) —
  projected stress: {o.projected_band}</strong></p>
  {chart_html}
  <figcaption style="color:var(--muted);font-size:13.5px;max-width:720px">
  The only forward-looking panel on this page — and therefore the one that
  gets scored. Every issued forecast is committed to an append-only ledger
  before the outcome is known and verified against realized rainfall once the
  window closes (see the track record below). Until enough windows close,
  treat this as an unverified model output, not a fact.</figcaption>
</section>
"""
```

3f. In `render_api_latest`, add after the `"farm_gate"` entry:

```python
        "forecast": ({
            "source": "Open-Meteo forecast model",
            "issued": s.outlook.issued,
            "window": [s.outlook.window_start, s.outlook.window_end],
            "expected_rain_mm": round(s.outlook.expected_mm, 1),
            "seasonal_norm_mm": round(s.outlook.norm_mm, 1),
            "anom_z": round(s.outlook.anom_z, 2),
            "projected_stress": round(s.outlook.projected_stress, 3),
            "projected_band": s.outlook.projected_band,
            "note": ("forward-looking model output; verified against realized "
                     "rainfall in the public track record as windows close"),
        } if s.outlook is not None else None),
```

3g. In `render_html`, add a `forecast_html: str = ""` keyword parameter and
place `{forecast_html}` between the Vietnam growing-conditions section and
`{brazil_html}`.

3h. In `build()`, after the Brazil panel block:

```python
    # forward look — optional layer, never publish-blocking
    forecast_html = ""
    if state.outlook is not None:
        try:
            from ascentagri.agronomy.forecast import load_forecast
            fc_frame, _ = load_forecast()
            chart_forecast(state.outlook, fc_frame, assets / "forecast.png")
            forecast_html = render_forecast_section(state.outlook, True)
        except Exception as exc:
            log.warning("forecast chart skipped: %s", exc)
            forecast_html = render_forecast_section(state.outlook, False)
```

and pass `forecast_html=forecast_html` to `render_html(...)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_build_site.py tests/test_forecast.py -q`
Expected: all pass (API test tolerates absent cache via the `None` branch).

- [ ] **Step 5: Build the site locally and eyeball the output**

Run: `python -m ascentagri.agronomy.forecast fetch && python site/build_site.py`
Expected: `[forecast] cached 16 days; snapshot appended ...`, then the build
prints a brief containing "Looking ahead". Open `site/_build/index.html` and
confirm "The next two weeks" renders. (If offline, skip the fetch — the build
must still succeed without the section.)

- [ ] **Step 6: Commit**

```bash
git add site/build_site.py tests/test_build_site.py
git commit -m "feat: 14-day outlook on the site — section, chart, briefs EN/VI, API block"
```

---

### Task 6: The track-record panel — verification joins the ledger section

**Files:**
- Modify: `site/build_site.py` (`render_ledger_section`, `build()`, ledger imports)
- Test: `tests/test_build_site.py` (append)

**Interfaces:**
- Consumes: `score_snapshots`, `ForecastVerification`, `MIN_VERIFIED_WINDOWS` from Task 4
- Produces: `render_ledger_section(score, has_chart, verification=None)` — existing callers with two args keep working

- [ ] **Step 1: Write the failing tests** (append to `tests/test_build_site.py`)

```python
def test_ledger_section_shows_forecast_verification_young():
    from ascentagri.agronomy.forecast import ForecastVerification
    from ascentagri.ledger import score_ledger
    score = score_ledger([])
    ver = ForecastVerification(
        n_snapshots=3, n_closed=0, mae_forecast_mm=0.0,
        mae_climatology_mm=0.0, bias_mm=0.0, skill=0.0, band_hit_rate=0.0,
        first_scoreable="2026-07-25")
    html = build_site.render_ledger_section(score, False, verification=ver)
    assert "Forecast verification" in html
    assert "2026-07-25" in html


def test_ledger_section_shows_forecast_verification_scored():
    from ascentagri.agronomy.forecast import ForecastVerification
    from ascentagri.ledger import score_ledger
    ver = ForecastVerification(
        n_snapshots=9, n_closed=6, mae_forecast_mm=12.0,
        mae_climatology_mm=10.0, bias_mm=3.0, skill=-0.2, band_hit_rate=0.5,
        first_scoreable=None)
    html = build_site.render_ledger_section(score_ledger([]), False,
                                            verification=ver)
    assert "skill" in html and "-0.20" in html
    assert "climatology" in html and "does not beat" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_build_site.py -q`
Expected: FAIL — `render_ledger_section() got an unexpected keyword argument 'verification'`.

- [ ] **Step 3: Implement**

3a. Change the signature and body of `render_ledger_section`:

```python
def render_ledger_section(score: LedgerScore, has_chart: bool,
                          verification=None) -> str:
```

Before the final `return`, build the verification block:

```python
    from ascentagri.agronomy.forecast import MIN_VERIFIED_WINDOWS
    snap_url = ("https://github.com/ScottDongKhang/ascent-agri/blob/main/"
                "data/ledger/weather_forecasts.jsonl")
    if verification is None or verification.n_snapshots == 0:
        ver_html = ""
    elif verification.n_closed < MIN_VERIFIED_WINDOWS:
        when = (f" First scored table appears after {MIN_VERIFIED_WINDOWS} "
                f"closed windows"
                + (f" (~{verification.first_scoreable})."
                   if verification.first_scoreable else "."))
        ver_html = (
            f'<h3 style="margin-top:24px;font-size:16px">Forecast '
            f'verification</h3>'
            f'<p class="asof">{verification.n_snapshots} rainfall forecasts '
            f'issued and committed; {verification.n_closed} windows closed.'
            f'{when} <a href="{snap_url}">Inspect the raw snapshots</a>.</p>')
    else:
        v = verification
        verdict = ("beats" if v.skill > 0 else "does not beat")
        ver_html = (
            f'<h3 style="margin-top:24px;font-size:16px">Forecast '
            f'verification</h3>'
            f'<p class="asof">{v.n_closed} closed 14-day windows · forecast '
            f'MAE {v.mae_forecast_mm:.1f} mm vs climatology '
            f'{v.mae_climatology_mm:.1f} mm · skill {v.skill:+.2f} '
            f'({verdict} the climatology baseline) · bias {v.bias_mm:+.1f} mm '
            f'· stress-band hit rate {v.band_hit_rate:.0%} · '
            f'<a href="{snap_url}">raw snapshots</a>.</p>')
```

and change the returned f-string to include `{ver_html}` right after `{body}`.

3b. In `build()`, replace the ledger lines with:

```python
    ledger_score = score_ledger(read_ledger())
    has_ledger_chart = chart_ledger(ledger_score, assets / "ledger.png")
    forecast_ver = None
    try:
        from ascentagri.agronomy.forecast import score_snapshots
        forecast_ver = score_snapshots()
    except Exception as exc:
        log.warning("forecast verification skipped: %s", exc)
    ledger_html = render_ledger_section(ledger_score, has_ledger_chart,
                                        verification=forecast_ver)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_build_site.py -q` — expected: all pass (including the
pre-existing `test_ledger_section_present`).

- [ ] **Step 5: Commit**

```bash
git add site/build_site.py tests/test_build_site.py
git commit -m "feat: forecast verification joins the public track-record panel"
```

---

### Task 7: `api/changes.json` + `alerts.xml` — the machine alert surface

**Files:**
- Modify: `site/build_site.py`
- Test: `tests/test_build_site.py` (append)

**Interfaces:**
- Consumes: `MonitorState.signals` (`label` column), `MonitorState.feature_panel` (`rain_anom_30d`), `crop_stress_index`, `stress_label`, `read_snapshots` from Task 3
- Produces:
  - `compute_changes(s: MonitorState, snapshots: "list[dict] | None" = None, n: int = 20) -> "list[dict]"` — newest first, each `{date, type, from, to}` with `type` ∈ {`regime_change`, `stress_band_change`, `projected_stress_band_change`}
  - `render_changes_json(changes) -> str`
  - `render_alerts_feed(changes, site_url: str) -> str`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_build_site.py`)

```python
def _mini_state(labels, rain_z):
    idx = pd.bdate_range("2026-06-01", periods=len(labels))
    signals = pd.DataFrame({"label": labels}, index=idx)
    panel = pd.DataFrame({"rain_anom_30d": rain_z}, index=idx)

    class P:
        posture = "neutral"
        risk_multiplier = 1.0
    return build_site.MonitorState(
        close=None, signals=signals, feature_panel=panel, brl=None,
        weather=None, posture=P(), label=labels[-1], dwell=1, price=100.0,
        chg_1w=0.0, chg_1m=0.0, rain_z=rain_z[-1], dry_frac=None,
        brl_chg_21d=None, price_asof="", weather_asof="", brl_asof="")


def test_compute_changes_finds_transitions():
    s = _mini_state(["calm_bull", "calm_bull", "stressed", "stressed"],
                    [0.0, 0.0, -3.0, -3.0])   # June: filling, d_w 0.5 → band flip
    changes = build_site.compute_changes(s)
    types = {c["type"] for c in changes}
    assert "regime_change" in types and "stress_band_change" in types
    reg = [c for c in changes if c["type"] == "regime_change"][0]
    assert reg["from"] == "calm_bull" and reg["to"] == "stressed"
    dates = [c["date"] for c in changes]
    assert dates == sorted(dates, reverse=True)     # newest first


def test_compute_changes_projected_band_from_snapshots():
    s = _mini_state(["calm_bull", "calm_bull"], [0.0, 0.0])
    snaps = [{"date_issued": "2026-06-01", "projected_band": "low"},
             {"date_issued": "2026-06-02", "projected_band": "elevated"}]
    changes = build_site.compute_changes(s, snapshots=snaps)
    proj = [c for c in changes if c["type"] == "projected_stress_band_change"]
    assert len(proj) == 1
    assert proj[0]["from"] == "low" and proj[0]["to"] == "elevated"


def test_compute_changes_no_changes_is_empty():
    s = _mini_state(["calm_bull", "calm_bull"], [0.0, 0.0])
    assert build_site.compute_changes(s, snapshots=[]) == []


def test_changes_json_and_alerts_feed_in_build(built):
    import json
    import xml.etree.ElementTree as ET
    payload = json.loads((built / "api" / "changes.json").read_text())
    assert payload["schema_version"] == 1
    assert isinstance(payload["changes"], list)
    root = ET.fromstring((built / "alerts.xml").read_text())
    assert root.tag == "rss"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_build_site.py -q`
Expected: FAIL — `compute_changes` missing.

- [ ] **Step 3: Implement** (place after `render_api_history` in `site/build_site.py`)

```python
def compute_changes(s: MonitorState, snapshots: "list[dict] | None" = None,
                    n: int = 20) -> "list[dict]":
    """State CHANGES only (not daily state), for pollers: regime label flips,
    stress-band crossings, and projected-band flips from the snapshot ledger.
    Derived deterministically from already-committed history — no state file."""
    events = []

    lab = s.signals["label"].astype(str)
    flips = lab[lab != lab.shift(1)].iloc[1:]         # first row has no prior
    for date in flips.index:
        pos = lab.index.get_loc(date)
        events.append({"date": str(date.date()), "type": "regime_change",
                       "from": str(lab.iloc[pos - 1]), "to": str(lab.loc[date])})

    if s.feature_panel is not None and "rain_anom_30d" in s.feature_panel:
        z = s.feature_panel["rain_anom_30d"].dropna()
        if len(z) > 1:
            bands = crop_stress_index(z).map(stress_label)
            bflips = bands[bands != bands.shift(1)].iloc[1:]
            for date in bflips.index:
                pos = bands.index.get_loc(date)
                events.append({"date": str(date.date()),
                               "type": "stress_band_change",
                               "from": str(bands.iloc[pos - 1]),
                               "to": str(bands.loc[date])})

    prev = None
    for e in (snapshots or []):
        band = e.get("projected_band")
        if prev is not None and band != prev:
            events.append({"date": e["date_issued"],
                           "type": "projected_stress_band_change",
                           "from": prev, "to": band})
        prev = band

    events.sort(key=lambda d: (d["date"], d["type"]))
    return list(reversed(events[-n:]))                 # newest first


def render_changes_json(changes: "list[dict]") -> str:
    import json
    return json.dumps({
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "doc": ("State changes only — poll this instead of scraping the page. "
                "Types: regime_change, stress_band_change, "
                "projected_stress_band_change. Newest first."),
        "changes": changes,
        "attribution": API_ATTRIBUTION,
        "license": API_LICENSE,
    }, indent=2)


def render_alerts_feed(changes: "list[dict]", site_url: str) -> str:
    """RSS that fires ONLY on state changes — the subscription for people who
    want to hear from the monitor only when something moved."""
    from email.utils import format_datetime
    from xml.sax.saxutils import escape
    items = []
    for c in changes:
        d = (pd.Timestamp(c["date"]).to_pydatetime()
             .replace(hour=21, minute=30, tzinfo=timezone.utc))
        title = escape(f"{c['type'].replace('_', ' ')}: "
                       f"{c['from']} → {c['to']} ({c['date']})")
        items.append(
            f"    <item>\n"
            f"      <title>{title}</title>\n"
            f"      <link>{site_url}</link>\n"
            f"      <guid isPermaLink=\"false\">ascent-agri-alert-"
            f"{c['type']}-{c['date']}</guid>\n"
            f"      <pubDate>{format_datetime(d)}</pubDate>\n"
            f"      <description>{title}</description>\n"
            f"    </item>")
    now = format_datetime(datetime.now(timezone.utc))
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0">\n'
        "  <channel>\n"
        "    <title>Robusta Coffee Monitor — alerts</title>\n"
        f"    <link>{site_url}</link>\n"
        "    <description>Fires only when the regime or crop-stress state "
        "changes.</description>\n"
        f"    <lastBuildDate>{now}</lastBuildDate>\n"
        + "\n".join(items) + "\n"
        "  </channel>\n"
        "</rss>\n"
    )
```

In `build()`, after the `api_dir` block:

```python
    snapshots = []
    try:
        from ascentagri.agronomy.forecast import read_snapshots
        snapshots = read_snapshots()
    except Exception as exc:
        log.warning("snapshot read skipped: %s", exc)
    changes = compute_changes(state, snapshots=snapshots)
    (api_dir / "changes.json").write_text(render_changes_json(changes))
    (out_dir / "alerts.xml").write_text(render_alerts_feed(changes, site_url))
```

In `render_html`'s Data & API `<ul>`, add two entries after the `feed.xml` line:

```html
    <li><a href="api/changes.json"><code>api/changes.json</code></a> — state
        <em>changes</em> only (regime flips, stress-band crossings) — poll
        this on your schedule instead of scraping the page.</li>
    <li><a href="alerts.xml"><code>alerts.xml</code></a> — the same changes as
        an RSS feed: fires only when something moved (pairs well with any
        RSS-to-email bridge).</li>
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_build_site.py -q` — expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add site/build_site.py tests/test_build_site.py
git commit -m "feat: changes.json + alerts.xml — fire-on-change machine surfaces"
```

---

### Task 8: The weekly one-pager

**Files:**
- Create: `site/onepager.py`
- Modify: `site/build_site.py` (import + `build()` wiring + page links)
- Test: `tests/test_build_site.py` (append)

**Interfaces:**
- Consumes: `MonitorState` (duck-typed), the daily brief string, `LedgerScore.summary_line()`, `ForecastVerification.summary_line()`
- Produces:
  - `render_onepager(s, brief: str, ledger_line: str, verification_line: str, updated: str) -> str` (full HTML document, print-first styling, no external assets)
  - `archive_name(now: "datetime | None" = None) -> "str | None"` — `"YYYY-MM-DD.html"` on Fridays (UTC), else `None`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_build_site.py`)

```python
def _load_onepager():
    spec2 = importlib.util.spec_from_file_location(
        "onepager", ROOT / "site" / "onepager.py")
    onepager = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(onepager)
    return onepager


def test_onepager_renders_complete_page():
    onepager = _load_onepager()

    class P:
        posture = "defensive"
        risk_multiplier = 0.65
    s = build_site.MonitorState(
        close=None, signals=None, feature_panel=None, brl=None, weather=None,
        posture=P(), label="stressed", dwell=7, price=250.0, chg_1w=-0.02,
        chg_1m=0.05, rain_z=-1.2, dry_frac=0.6, brl_chg_21d=0.01,
        price_asof="2026-07-02", weather_asof="2026-06-28",
        brl_asof="2026-07-02", crop_stage="fruit filling",
        crop_stress=0.6, crop_stress_band="watch",
        farm_gate_line="~94,808 đồng/kg", farm_gate_asof="2026-07-02",
        outlook=_stub_outlook())
    html = onepager.render_onepager(
        s, brief="Test brief sentence.",
        ledger_line="12 entries · 10 scored days",
        verification_line="3 forecasts issued, none with a closed window yet",
        updated="2026-07-04 12:00 UTC")
    for required in ["This week in one page", "stressed", "2026-07-02",
                     "2026-06-28", "fruit filling", "Test brief sentence.",
                     "CC BY 4.0", "not investment advice",
                     "scottdongkhang.github.io/ascent-agri",
                     "@media print"]:
        assert required in html, f"missing: {required!r}"
    assert "<img" not in html          # self-contained: prints without assets


def test_onepager_archive_name_fridays_only():
    from datetime import datetime, timezone
    onepager = _load_onepager()
    fri = datetime(2026, 7, 10, 22, 0, tzinfo=timezone.utc)   # a Friday
    tue = datetime(2026, 7, 7, 22, 0, tzinfo=timezone.utc)
    assert onepager.archive_name(fri) == "2026-07-10.html"
    assert onepager.archive_name(tue) is None


def test_onepager_written_by_build(built):
    latest = built / "brief" / "latest.html"
    assert latest.exists()
    assert "This week in one page" in latest.read_text()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_build_site.py -q`
Expected: FAIL — `site/onepager.py` does not exist.

- [ ] **Step 3: Create `site/onepager.py`**

```python
"""The weekly one-pager: the whole monitor on a single printable page.

Self-contained HTML (no images, print-first CSS) so it can be forwarded,
attached, or Cmd+P'd into a clean PDF. Rebuilt on every daily run;
a dated archive copy is kept on Fridays.
"""
from __future__ import annotations

import datetime as _dt
from typing import Optional

SITE_URL = "https://scottdongkhang.github.io/ascent-agri/"


def archive_name(now: "Optional[_dt.datetime]" = None) -> "Optional[str]":
    """Dated archive filename on Fridays (UTC), else None."""
    now = now or _dt.datetime.now(_dt.timezone.utc)
    return f"{now:%Y-%m-%d}.html" if now.weekday() == 4 else None


def render_onepager(s, brief: str, ledger_line: str,
                    verification_line: str, updated: str) -> str:
    pct = lambda x: f"{x:+.1%}" if x is not None else "n/a"
    o = getattr(s, "outlook", None)
    outlook_row = ""
    if o is not None:
        outlook_row = f"""
  <tr><th>Next 14 days</th><td>{o.expected_mm:.0f} mm expected vs
    {o.norm_mm:.0f} mm norm ({o.anom_z:+.1f}σ) — projected stress:
    <strong>{o.projected_band}</strong>
    <span class="dim">(Open-Meteo forecast, issued {o.issued})</span></td></tr>"""
    farm_row = (f"""
  <tr><th>Farm gate</th><td>{s.farm_gate_line}
    <span class="dim">({s.farm_gate_asof})</span></td></tr>"""
                if s.farm_gate_line else "")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Robusta Coffee Monitor — weekly one-pager</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font: 13px/1.5 -apple-system, "Segoe UI", Helvetica, Arial,
         sans-serif; color: #1a1a19; background: #fff; padding: 32px; }}
  .sheet {{ max-width: 720px; margin: 0 auto; }}
  h1 {{ font: 500 22px/1.2 Georgia, serif; }}
  .kicker {{ font-size: 10px; letter-spacing: .16em; text-transform: uppercase;
            color: #666; margin-bottom: 6px; }}
  .dim {{ color: #666; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
  th, td {{ text-align: left; padding: 7px 10px 7px 0; vertical-align: top;
           border-bottom: 1px solid #e2e0d8; }}
  th {{ width: 130px; font-weight: 600; font-size: 12px; color: #444; }}
  .brief {{ margin-top: 14px; padding: 12px 14px; background: #f6f5f0;
           border-left: 3px solid #999; }}
  footer {{ margin-top: 18px; font-size: 11px; color: #666; }}
  footer p {{ margin-top: 4px; }}
  @media print {{ body {{ padding: 0; }} .sheet {{ max-width: none; }} }}
</style>
</head>
<body>
<div class="sheet">
  <div class="kicker">ascent-agri · robusta coffee monitor · weekly brief</div>
  <h1>This week in one page</h1>
  <p class="dim">generated {updated} · live version, data files and methods:
    <a href="{SITE_URL}">{SITE_URL.replace("https://", "")}</a></p>
  <table>
  <tr><th>Regime</th><td><strong>{s.label}</strong> ({s.posture.posture}
    posture, ×{s.posture.risk_multiplier:.2f} exposure guide) —
    {max(s.dwell, 1)} sessions
    <span class="dim">(price through {s.price_asof})</span></td></tr>
  <tr><th>Price</th><td>{s.price:,.0f}¢/lb · {pct(s.chg_1w)} on the week ·
    {pct(s.chg_1m)} on the month</td></tr>
  <tr><th>Growing belt</th><td>30-day rainfall anomaly
    {f"{s.rain_z:+.1f}σ" if s.rain_z is not None else "n/a"} at Buon Ma Thuot ·
    crop stage: {s.crop_stage} · stress: {s.crop_stress_band or "n/a"}
    <span class="dim">(weather through {s.weather_asof})</span></td></tr>{outlook_row}{farm_row}
  <tr><th>Track record</th><td>{ledger_line}<br>
    <span class="dim">Forecast verification: {verification_line}</span></td></tr>
  </table>
  <div class="brief">{brief}</div>
  <footer>
    <p>Data: Yahoo Finance · Open-Meteo. Derived series (regimes, anomalies,
    briefs) CC BY 4.0 — free to reuse with attribution to ascent-agri.</p>
    <p>This is not investment advice. An open-source agricultural
    market-intelligence project — every model call is committed to a public,
    append-only ledger before the outcome is known.</p>
  </footer>
</div>
</body>
</html>
"""
```

- [ ] **Step 4: Wire into `site/build_site.py`**

4a. Below the existing `sys.path.insert(0, str(ROOT))` line add:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent))
from onepager import archive_name, render_onepager      # noqa: E402
```

4b. In `build()`, after the changes/alerts block (Task 7):

```python
    # weekly one-pager — rebuilt daily, archived on Fridays
    brief_dir = out_dir / "brief"
    brief_dir.mkdir(parents=True, exist_ok=True)
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ver_line = (forecast_ver.summary_line() if forecast_ver is not None
                else "forecast layer not yet live")
    onepager_html = render_onepager(state, brief, ledger_score.summary_line(),
                                    ver_line, updated)
    (brief_dir / "latest.html").write_text(onepager_html)
    arch = archive_name()
    if arch:
        (brief_dir / arch).write_text(onepager_html)
```

4c. In `render_html`'s Data & API `<ul>`, add:

```html
    <li><a href="brief/latest.html"><code>brief/latest.html</code></a> — the
        week on one printable page (Cmd/Ctrl+P → PDF); Friday editions are
        archived by date.</li>
```

and in the footer line that links the RSS feed, extend to:

```html
  <a href="feed.xml">RSS daily brief</a> ·
  <a href="alerts.xml">alerts feed</a> ·
  <a href="brief/latest.html">weekly one-pager</a> ·
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_build_site.py -q` — expected: all pass.

- [ ] **Step 6: Full build + suite, then commit**

Run: `python site/build_site.py && pytest -q`
Expected: build prints the brief; `site/_build/brief/latest.html` exists; suite green.

```bash
git add site/onepager.py site/build_site.py tests/test_build_site.py
git commit -m "feat: printable weekly one-pager, rebuilt daily, archived Fridays"
```

---

### Task 9: Workflow wiring, sanity checks, and docs

**Files:**
- Modify: `.github/workflows/update-site.yml`
- Modify: `README.md`
- Modify: `docs/ROADMAP-to-Nov-1.md`

**Interfaces:**
- Consumes: `python -m ascentagri.agronomy.forecast fetch` (Task 3), the new build outputs (Tasks 5–8)
- Produces: the daily workflow fetches the forecast (non-blocking), commits the snapshot ledger, and sanity-checks every new surface.

- [ ] **Step 1: Update `.github/workflows/update-site.yml`**

1a. After the "Fetch fresh data" step, add:

```yaml
      - name: Fetch 16-day forecast (optional — never publish-blocking)
        run: |
          python -m ascentagri.agronomy.forecast fetch \
            || echo "::warning::forecast fetch failed — site builds without the forward outlook"
```

1b. In the "Append today's ledger entry" step, change the `git add` line to:

```yaml
          git add data/ledger/forecasts.jsonl data/ledger/weather_forecasts.jsonl
```

1c. Extend the "Sanity-check output" step with:

```yaml
          test -s site/_build/api/changes.json
          test -s site/_build/alerts.xml
          test -s site/_build/brief/latest.html
          grep -q "This week in one page" site/_build/brief/latest.html
          python -c "import json; json.load(open('site/_build/api/changes.json'))"
```

- [ ] **Step 2: Update `README.md`**

2a. In "The public monitor" bullet list, add:

```markdown
- Forward look: a 14-day rainfall outlook for the robusta belt (Open-Meteo
  forecast × the crop phenology calendar). Every issued forecast is committed
  to an append-only snapshot ledger (`data/ledger/weather_forecasts.jsonl`)
  before the outcome is known and scored against realized rainfall once the
  window closes — skill vs climatology, published whatever it says
  (`python -m ascentagri.agronomy.forecast score` reproduces it).
- Machine surfaces for adopters: `api/latest.json` (now with the `forecast`
  block), `api/changes.json` + `alerts.xml` (fire only on regime/stress-band
  changes — poll on your schedule), and `brief/latest.html` (the week on one
  printable page, archived by date on Fridays).
```

2b. In the Layout block, change the `agronomy` area to mention the new module —
after the `macro_fetch.py` line add:

```
  agronomy/           phenology calendar + stage-weighted stress, farm-gate
                        economics, forecast.py (14-day outlook, snapshot
                        ledger, verification vs climatology)
```

- [ ] **Step 3: Update `docs/ROADMAP-to-Nov-1.md`**

In the "What exists today" list, add:

```markdown
- **Forward look + scored forecasts** — 14-day robusta-belt rainfall outlook
  with an append-only snapshot ledger; verification vs climatology appears on
  the site automatically as windows close (first table ~5 closed windows in).
- **Adoption surfaces** — `api/changes.json`, `alerts.xml` (fire-on-change),
  and the printable weekly one-pager at `brief/latest.html` — these are what
  the OUTREACH.md playbook points organizations at.
```

- [ ] **Step 4: Run the full suite**

Run: `pytest -q`
Expected: full suite green. (The workflow YAML is validated by GitHub on push;
`workflow_dispatch` in the final Verification section is the live check.)

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/update-site.yml README.md docs/ROADMAP-to-Nov-1.md
git commit -m "feat: wire forecast layer into the daily workflow + document the new surfaces"
```

---

## Verification (after all tasks)

1. `pytest -q` — full suite green (expect ~150+ tests).
2. `python -m ascentagri.agronomy.forecast fetch` (network) then
   `python site/build_site.py` — page shows "The next two weeks", brief says
   "Looking ahead", `api/latest.json` has a non-null `forecast` block,
   `api/changes.json` and `alerts.xml` parse, `brief/latest.html` renders.
3. `python -m ascentagri.agronomy.forecast score` — prints the
   "not yet scoreable" line with a first-scoreable date.
4. Push and run the `update-site.yml` workflow via `workflow_dispatch`; confirm
   green run and that `data/ledger/weather_forecasts.jsonl` gained one
   committed entry.
