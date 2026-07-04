"""Build the public Robusta Coffee Monitor — a single static page, updated daily.

Reads the data caches in data/processed/ (this script NEVER fetches — the
publish workflow fetches first, so a failed fetch can never publish a wrong
page), runs the regime engine, renders three dark charts, and writes a
self-contained site into site/_build/.

Usage:
    python site/build_site.py            # -> site/_build/index.html + assets/
    python site/build_site.py --out DIR  # custom output dir (tests)

Design: dark editorial, quiet professional. Validated dark-mode chart palette;
regime bands use the repo's posture status colors with text legends. Honest
labeling throughout: the daily price series is the ICE arabica benchmark until
enough robusta contracts are collected.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ascentagri.agronomy.economics import (                       # noqa: E402
    load_usdvnd, transmission_line, transmission_line_vi)
from ascentagri.agronomy.phenology import (                       # noqa: E402
    crop_stress_index, stage_for, stress_label)
from ascentagri.ledger import LedgerScore, read_ledger, score_ledger  # noqa: E402
from ascentagri.regime.engine import RegimeEngine                 # noqa: E402
from ascentagri.regime.features import RegimeFeatureBuilder      # noqa: E402
from ascentagri.regime.posture import compute_posture_from_regime  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
DEFAULT_OUT = ROOT / "site" / "_build"

log = logging.getLogger(__name__)

# ── palette (validated dark-mode steps + posture status colors) ────────────
SURFACE   = "#1a1a19"
PAGE_BG   = "#141413"
INK       = "#e8e6dd"
MUTED     = "#9a9890"
HAIRLINE  = "#2c2c2a"
BLUE      = "#3987e5"
AQUA      = "#199e70"
YELLOW    = "#c98500"
ORANGE    = "#d95926"
REGIME_COLORS = {   # same mapping as ascentagri/regime/posture.py
    "calm_bull": "#22c55e", "euphoric": "#84cc16", "stressed": "#f59e0b",
    "crisis": "#ef4444", "uncertain": "#64748b",
}
REGIME_WORDS = {
    "calm_bull": "calm uptrend", "euphoric": "late-cycle", "stressed": "stressed",
    "crisis": "crisis", "uncertain": "uncertain",
}

# ── Vietnamese strings (grower page) ────────────────────────────────────────
# Draft translations written for simple, plain market/agronomy Vietnamese.
# A native-speaker review is on the roadmap — wording kept short and factual
# to minimize risk until then.
REGIME_WORDS_VI = {
    "calm_bull": "xu hướng tăng ổn định", "euphoric": "giai đoạn cuối chu kỳ",
    "stressed": "căng thẳng", "crisis": "khủng hoảng",
    "uncertain": "không chắc chắn",
}
STAGE_VI = {
    "flowering & fruit set": "ra hoa và đậu quả",
    "early fruit development": "quả non phát triển",
    "fruit filling": "quả vào chắc",
    "maturation & harvest": "chín và thu hoạch",
}
STRESS_VI = {"low": "thấp", "watch": "cần theo dõi",
             "elevated": "cao", "severe": "nghiêm trọng", "": ""}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "figure.dpi": 100,
    "text.color": INK, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": HAIRLINE, "grid.color": HAIRLINE,
    "axes.grid": True, "grid.alpha": 0.6, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
})


# ── state ───────────────────────────────────────────────────────────────────

@dataclass
class MonitorState:
    close: pd.Series
    signals: pd.DataFrame
    feature_panel: pd.DataFrame
    brl: pd.Series
    weather: pd.DataFrame
    posture: object                 # RegimeSummary
    label: str
    dwell: int
    price: float
    chg_1w: float
    chg_1m: float
    rain_z: Optional[float]
    dry_frac: Optional[float]
    brl_chg_21d: Optional[float]
    price_asof: str
    weather_asof: str
    brl_asof: str
    crop_stage: str = ""              # human label, e.g. "fruit filling"
    crop_stage_note: str = ""         # one-line agronomy rationale
    crop_stress: Optional[float] = None
    crop_stress_band: str = ""        # low / watch / elevated / severe
    farm_gate_line: Optional[str] = None
    farm_gate_line_vi: Optional[str] = None
    farm_gate_asof: str = ""
    outlook: Optional[object] = None  # ForwardOutlook when the cache is fresh


def load_inputs() -> "tuple[pd.Series, pd.Series, pd.DataFrame]":
    """Load the three caches. Fails loudly — the builder never fetches."""
    price_csv = PROCESSED / "coffee_KCF_yahoo.csv"
    brl_csv = PROCESSED / "brlusd.csv"
    wx_csv = PROCESSED / "weather_central_highlands.csv"
    missing = [p.name for p in (price_csv, brl_csv, wx_csv) if not p.exists()]
    if missing:
        raise SystemExit(
            f"Missing caches: {missing}. Run `python -m ascentagri.vendor_fetch` "
            f"and `python -m ascentagri.macro_fetch` first — build_site never fetches.")
    close = (pd.read_csv(price_csv, parse_dates=["date"])
             .set_index("date").sort_index()["close"])
    brl = (pd.read_csv(brl_csv, parse_dates=["date"])
           .set_index("date").sort_index()["brl_per_usd"])
    weather = (pd.read_csv(wx_csv, parse_dates=["date"])
               .set_index("date").sort_index())
    return close, brl, weather


def compute_state(close: pd.Series, brl: pd.Series,
                  weather: pd.DataFrame) -> MonitorState:
    """Fit the regime engine and assemble everything the page needs."""
    engine = RegimeEngine()
    engine.fit(close, brl_usd=brl, weather=weather,
               run_model_selection=False, k_override=3, hmm_restarts=5)
    signals = engine.get_signal_series()
    panel = engine.get_feature_panel()

    last = signals.iloc[-1]
    label = str(last["label"])
    dwell = int(last.get("dwell_days", 0))
    prob_cols = sorted(c for c in signals.columns if c.startswith("prob_"))
    probs = {c: float(last[c]) for c in prob_cols}

    posture = compute_posture_from_regime(
        asof=str(signals.index[-1].date()),
        regime_label=label,
        probs=probs,
        days_in_regime=dwell,
    )

    price = float(close.iloc[-1])
    chg_1w = float(close.iloc[-1] / close.iloc[-6] - 1) if len(close) > 6 else 0.0
    chg_1m = float(close.iloc[-1] / close.iloc[-22] - 1) if len(close) > 22 else 0.0

    rain_z = (float(panel["rain_anom_30d"].dropna().iloc[-1])
              if "rain_anom_30d" in panel and panel["rain_anom_30d"].notna().any() else None)
    dry_frac = (float(panel["dry_frac_30d"].dropna().iloc[-1])
                if "dry_frac_30d" in panel and panel["dry_frac_30d"].notna().any() else None)
    brl_chg = (float(brl.iloc[-1] / brl.iloc[-22] - 1) if len(brl) > 22 else None)

    # agronomy: crop stage + stage-weighted stress (phenology-aware)
    today = close.index[-1]
    stage = stage_for(today)
    crop_stress = None
    if rain_z is not None:
        crop_stress = float(crop_stress_index(
            pd.Series([rain_z], index=[today])).iloc[0])

    # farm-gate economics (optional layer: robusta series + USD/VND both
    # needed; either missing → line silently absent, never a blocker)
    farm_gate_line = None
    farm_gate_line_vi = None
    farm_gate_asof = ""
    robusta_csv = ROOT / "data" / "processed" / "robusta_continuous.csv"
    vnd = load_usdvnd()
    if robusta_csv.exists() and vnd is not None and len(vnd) > 0:
        try:
            rob = (pd.read_csv(robusta_csv, parse_dates=["date"])
                   .set_index("date").sort_index()["close"])
            rob_chg_1m = (float(rob.iloc[-1] / rob.iloc[-22] - 1)
                          if len(rob) > 22 else None)
            farm_gate_line = transmission_line(
                float(rob.iloc[-1]), float(vnd.iloc[-1]), chg_1m=rob_chg_1m)
            farm_gate_line_vi = transmission_line_vi(
                float(rob.iloc[-1]), float(vnd.iloc[-1]), chg_1m=rob_chg_1m)
            farm_gate_asof = (f"robusta futures {rob.index[-1].date()} · "
                              f"USD/VND {vnd.index[-1].date()}")
        except Exception as exc:
            log.warning("farm-gate layer skipped: %s", exc)

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

    return MonitorState(
        close=close, signals=signals, feature_panel=panel, brl=brl,
        weather=weather, posture=posture, label=label, dwell=dwell,
        price=price, chg_1w=chg_1w, chg_1m=chg_1m,
        rain_z=rain_z, dry_frac=dry_frac, brl_chg_21d=brl_chg,
        price_asof=str(close.index[-1].date()),
        weather_asof=str(weather.index[-1].date()),
        brl_asof=str(brl.index[-1].date()),
        crop_stage=stage.label,
        crop_stage_note=stage.note,
        crop_stress=crop_stress,
        crop_stress_band=stress_label(crop_stress) if crop_stress is not None else "",
        farm_gate_line=farm_gate_line,
        farm_gate_line_vi=farm_gate_line_vi,
        farm_gate_asof=farm_gate_asof,
        outlook=outlook,
    )


# ── daily brief (deterministic template — no LLM) ───────────────────────────

def daily_brief(s: MonitorState) -> str:
    word = REGIME_WORDS.get(s.label, s.label)
    posture_note = (f" ({s.posture.posture} posture)"
                    if s.posture.posture != word else "")
    dir_w = "up" if s.chg_1w >= 0 else "down"
    dir_m = "up" if s.chg_1m >= 0 else "down"
    parts = [
        f"Coffee futures closed at {s.price:,.0f}¢/lb — {dir_w} "
        f"{abs(s.chg_1w):.1%} on the week and {dir_m} {abs(s.chg_1m):.1%} on "
        f"the month. The regime model reads the market as {word}"
        f"{posture_note} and has for {max(s.dwell, 1)} sessions."
    ]
    if s.rain_z is not None:
        stage_bit = (f" The crop is in its {s.crop_stage} window"
                     f"{f', and stage-weighted stress is {s.crop_stress_band}' if s.crop_stress_band else ''}."
                     if s.crop_stage else "")
        if s.rain_z <= -1.0:
            parts.append(
                f"Rainfall around Buon Ma Thuot is running well below its "
                f"seasonal norm (30-day anomaly {s.rain_z:+.1f}σ) — the classic "
                f"robusta supply-risk setup.{stage_bit}")
        elif s.rain_z >= 1.0:
            parts.append(
                f"Rainfall in the Central Highlands is running above its "
                f"seasonal norm ({s.rain_z:+.1f}σ) — favorable moisture for the "
                f"robusta belt.{stage_bit}")
        else:
            parts.append(
                f"Growing conditions look unremarkable: Central Highlands "
                f"rainfall is near its seasonal norm ({s.rain_z:+.1f}σ).{stage_bit}")
    if s.brl_chg_21d is not None:
        if s.brl_chg_21d >= 0.02:
            parts.append(
                f"A weakening Brazilian real ({s.brl_chg_21d:+.1%} in a month) "
                f"raises producers' local-currency revenue per bag — added "
                f"selling pressure.")
        elif s.brl_chg_21d <= -0.02:
            parts.append(
                f"A strengthening Brazilian real ({s.brl_chg_21d:+.1%} in a "
                f"month) squeezes producer margins — supportive for prices.")
        else:
            parts.append(
                f"The Brazilian real is little changed on the month "
                f"({s.brl_chg_21d:+.1%}) — a neutral currency backdrop.")
    if s.outlook is not None:
        o = s.outlook
        parts.append(
            f"Looking ahead: the 14-day forecast calls for "
            f"{o.expected_mm:.0f} mm of rain in the robusta belt against a "
            f"{o.norm_mm:.0f} mm seasonal norm ({o.anom_z:+.1f}σ) — projected "
            f"crop stress: {o.projected_band} (Open-Meteo forecast model, "
            f"issued {o.issued}; skill scored publicly as windows close).")
    return " ".join(parts)


def _brief_state_at(s: MonitorState, date: pd.Timestamp) -> Optional[MonitorState]:
    """Reconstruct the (small) subset of MonitorState that daily_brief needs,
    as of a historical trading date. Purely from already-computed history —
    no refits, no fetches."""
    close = s.close.loc[:date]
    if len(close) < 23:
        return None
    sig = s.signals.loc[:date]
    if sig.empty:
        return None
    last = sig.iloc[-1]
    label = str(last["label"])
    dwell = int(last.get("dwell_days", 0))
    posture = compute_posture_from_regime(
        asof=str(date.date()), regime_label=label, probs={},
        days_in_regime=dwell, min_confidence=0.0)
    panel = s.feature_panel.loc[:date]
    rain = panel["rain_anom_30d"].dropna() if "rain_anom_30d" in panel else pd.Series(dtype=float)
    brl = s.brl.loc[:date]
    rain_z = float(rain.iloc[-1]) if len(rain) else None
    stage = stage_for(date)
    stress = (float(crop_stress_index(pd.Series([rain_z], index=[date])).iloc[0])
              if rain_z is not None else None)
    return MonitorState(
        close=close, signals=sig, feature_panel=panel, brl=brl,
        weather=s.weather, posture=posture, label=label, dwell=dwell,
        price=float(close.iloc[-1]),
        chg_1w=float(close.iloc[-1] / close.iloc[-6] - 1),
        chg_1m=float(close.iloc[-1] / close.iloc[-22] - 1),
        rain_z=rain_z,
        dry_frac=None,
        brl_chg_21d=float(brl.iloc[-1] / brl.iloc[-22] - 1) if len(brl) > 22 else None,
        price_asof=str(date.date()), weather_asof="", brl_asof="",
        crop_stage=stage.label, crop_stage_note=stage.note,
        crop_stress=stress,
        crop_stress_band=stress_label(stress) if stress is not None else "",
    )


def render_feed(s: MonitorState, site_url: str, n: int = 10) -> str:
    """RSS 2.0 feed of the last `n` trading-day briefs (stable date GUIDs)."""
    from email.utils import format_datetime
    from xml.sax.saxutils import escape

    items = []
    for date in s.close.index[-n:]:
        st = _brief_state_at(s, date)
        if st is None:
            continue
        brief = escape(daily_brief(st))
        d = date.to_pydatetime().replace(hour=21, minute=30, tzinfo=timezone.utc)
        items.append(
            f"    <item>\n"
            f"      <title>Coffee monitor — {date.date()} — "
            f"{escape(REGIME_WORDS.get(st.label, st.label))}</title>\n"
            f"      <link>{site_url}</link>\n"
            f"      <guid isPermaLink=\"false\">ascent-agri-{date.date()}</guid>\n"
            f"      <pubDate>{format_datetime(d)}</pubDate>\n"
            f"      <description>{brief}</description>\n"
            f"    </item>"
        )
    now = format_datetime(datetime.now(timezone.utc))
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0">\n'
        "  <channel>\n"
        "    <title>Robusta Coffee Monitor</title>\n"
        f"    <link>{site_url}</link>\n"
        "    <description>Daily model-driven read on coffee markets and "
        "Vietnamese Central Highlands growing conditions.</description>\n"
        f"    <lastBuildDate>{now}</lastBuildDate>\n"
        + "\n".join(reversed(items)) + "\n"
        "  </channel>\n"
        "</rss>\n"
    )


API_SCHEMA_VERSION = 1
API_ATTRIBUTION = ("Regime labels, risk multipliers and anomalies computed by "
                   "ascent-agri (github.com/ScottDongKhang/ascent-agri). "
                   "Underlying inputs: Yahoo Finance, Open-Meteo (CC BY 4.0).")
API_LICENSE = "CC BY 4.0 (derived data); attribution required"


def render_api_latest(s: MonitorState, brief: str) -> str:
    """Machine-readable snapshot of today's model view. DERIVED data only —
    labels, multipliers and anomalies this project computes; no raw market
    data is redistributed."""
    import json
    payload = {
        "schema_version": API_SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of": {
            "price": s.price_asof,
            "weather": s.weather_asof,
            "fx": s.brl_asof,
        },
        "regime": {
            "label": s.label,
            "posture": s.posture.posture,
            "risk_multiplier": s.posture.risk_multiplier,
            "sessions_in_regime": max(s.dwell, 1),
        },
        "market": {
            "chg_1w": round(s.chg_1w, 5),
            "chg_1m": round(s.chg_1m, 5),
            "series": "ICE arabica benchmark (robusta series in assembly)",
        },
        "growing_conditions": {
            "location": "Buon Ma Thuot, Dak Lak, Vietnam (12.68N, 108.04E)",
            "rain_anom_30d_z": round(s.rain_z, 3) if s.rain_z is not None else None,
            "dry_frac_30d": round(s.dry_frac, 3) if s.dry_frac is not None else None,
        },
        "crop": {
            "stage": s.crop_stage,
            "stage_note": s.crop_stage_note,
            "stress_index": round(s.crop_stress, 3) if s.crop_stress is not None else None,
            "stress_band": s.crop_stress_band or None,
            "method": ("30-day rainfall anomaly weighted by the Dak Lak "
                       "robusta phenology calendar (drought-sensitive "
                       "flowering/fruit-set, wetness-sensitive harvest)"),
        },
        "farm_gate": ({
            "line": s.farm_gate_line,
            "as_of": s.farm_gate_asof,
            "note": "futures-equivalent VND/kg, before local basis",
        } if s.farm_gate_line else None),
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
        "fx": {
            "brl_usd_chg_21d": round(s.brl_chg_21d, 5)
            if s.brl_chg_21d is not None else None,
        },
        "brief": brief,
        "attribution": API_ATTRIBUTION,
        "license": API_LICENSE,
        "docs": "https://scottdongkhang.github.io/ascent-agri/#data",
    }
    return json.dumps(payload, indent=2)


def render_api_history(s: MonitorState) -> str:
    """Full daily history of the model's derived series as CSV:
    date, regime label, risk multiplier, rainfall anomaly, dry fraction."""
    df = pd.DataFrame(index=s.signals.index)
    df["label"] = s.signals["label"]
    df["risk_multiplier"] = s.signals["risk_multiplier"]
    for col in ["rain_anom_30d", "dry_frac_30d"]:
        if col in s.feature_panel.columns:
            df[col] = s.feature_panel[col].reindex(df.index).round(4)
    df.index.name = "date"
    return df.to_csv(date_format="%Y-%m-%d")


def chart_long_view(s: MonitorState, out: Path):
    """Full-history price with the 200-day average — the near-to-long-term
    trend view. Added after feedback from a green-coffee educator that the
    trade thinks in long trends, not intraday moves."""
    close = s.close
    ma200 = close.rolling(200, min_periods=150).mean()
    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    ax.plot(close.index, close.values, color=INK, lw=1.1, label="KC=F close")
    ax.plot(ma200.index, ma200.values, color=YELLOW, lw=1.8,
            label="200-day average")
    ax.set_ylabel("¢/lb")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    _save(fig, out)


FROST_RISK_TMIN_C = 6.0


def chart_brazil(wx_br: pd.DataFrame, out: Path, lookback: int = 1095):
    """Sul de Minas (arabica belt) growing conditions: 30-day rainfall
    anomaly + minimum temperature with the frost-risk line."""
    from ascentagri.research.weather_study import rain_anomaly
    anom = rain_anomaly(wx_br).iloc[-lookback:]
    tmin = wx_br["tmin_c"].astype(float).iloc[-lookback:]
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 5.2), sharex=True)
    axes[0].plot(anom.index, anom.values, color=AQUA, lw=1.3)
    axes[0].axhline(0, color=MUTED, lw=0.8)
    axes[0].axhspan(-3.5, -1.0, color=YELLOW, alpha=0.08, lw=0)
    axes[0].set_ylabel("z-score")
    axes[0].set_title("30-day rainfall anomaly vs trailing year — Varginha, "
                      "Sul de Minas", loc="left", fontsize=9.5, color=MUTED)
    axes[1].plot(tmin.index, tmin.values, color=BLUE, lw=1.0)
    axes[1].axhline(FROST_RISK_TMIN_C, color="#e66767", lw=1.2, ls="--")
    axes[1].set_ylabel("min temp (°C)")
    axes[1].set_title("Daily minimum temperature — dashed line = frost-risk "
                      "threshold (the July 2021 frost drove arabica +34% in "
                      "five sessions)", loc="left", fontsize=9.5, color=MUTED)
    axes[1].xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    fig.tight_layout()
    _save(fig, out)


def render_long_view_section(s: MonitorState) -> str:
    return f"""
<section>
  <h2>The long view</h2>
  <p class="asof">full price history through {s.price_asof}</p>
  <figure>
    <img src="assets/long_view.png" alt="Full coffee price history with 200-day average">
    <figcaption>The market most of the trade actually discusses: price
    against its 200-day average, full history. Green-coffee buyers hedge and
    reprice off moves like these, not intraday noise — this view exists
    because an importer's education team told us that's the horizon they
    teach.</figcaption>
  </figure>
</section>
"""


def render_brazil_section(brazil_asof: str) -> str:
    return f"""
<section>
  <h2>Growing conditions — Sul de Minas, Brazil</h2>
  <p class="asof">weather data through {brazil_asof} · Varginha, Minas Gerais
  (−21.55, −45.43) · the arabica side of the market</p>
  <figure>
    <img src="assets/brazil.png" alt="Rainfall anomaly and minimum temperature at Varginha, Sul de Minas">
    <figcaption>Brazil dominates arabica, and Sul de Minas is its largest
    belt. Two risks matter here: dry spells (the same anomaly logic as the
    Vietnam panel) and winter frost — the dashed line marks the frost-risk
    threshold on daily minimum temperature. Most of the specialty trade buys
    arabica, so this panel pairs with the benchmark price above.</figcaption>
  </figure>
</section>
"""


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


LEDGER_CHART_MIN_DAYS = 10


def chart_ledger(score: LedgerScore, out: Path) -> bool:
    """Scored track record chart. Returns False (no chart) while the ledger
    is younger than LEDGER_CHART_MIN_DAYS scored days."""
    if score.n_scored_days < LEDGER_CHART_MIN_DAYS:
        return False
    strat_eq = (1 + score.strategy_daily).cumprod()
    bh_eq = (1 + score.bh_daily).cumprod()
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    ax.plot(strat_eq.index, strat_eq.values, color=BLUE, lw=1.6,
            label="model (public ledger, 1-day delay)")
    ax.plot(bh_eq.index, bh_eq.values, color=AQUA, lw=1.6,
            label="buy & hold")
    ax.set_ylabel("growth of $1")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    _save(fig, out)
    return True


def render_ledger_section(score: LedgerScore, has_chart: bool) -> str:
    raw_url = ("https://github.com/ScottDongKhang/ascent-agri/blob/main/"
               "data/ledger/forecasts.jsonl")
    if score.n_entries == 0:
        body = "<p class=\"asof\">The ledger opens with the next daily run.</p>"
    elif score.n_scored_days == 0:
        body = (f'<p class="asof">Ledger opened {score.start} · '
                f'{score.n_entries} entr{"y" if score.n_entries == 1 else "ies"} '
                f'so far — the scored track record appears here automatically '
                f'once enough days accumulate.</p>')
    else:
        chart_html = ('<figure><img src="assets/ledger.png" '
                      'alt="Ledger track record vs buy and hold"></figure>'
                      if has_chart else "")
        body = (f'<p class="asof">{score.n_entries} entries · '
                f'{score.n_scored_days} scored days ({score.start} → {score.end}) · '
                f'model {score.strategy_return:+.2%} vs buy-and-hold '
                f'{score.bh_return:+.2%} · mean exposure '
                f'{score.mean_exposure:.2f}</p>{chart_html}')
    return f"""
<section class="methods">
  <h2>The ledger — the model in public</h2>
  <p>Every weekday the pipeline writes down what the model believes —
  regime call and target exposure — <em>before</em> the outcome is known,
  and commits it to the repository. Entries are never edited or deleted;
  scoring uses only the prices recorded in the ledger itself, with a 1-day
  execution delay. If the model is wrong, this section says so forever.</p>
  {body}
  <ul><li><a href="{raw_url}">Inspect the raw ledger on GitHub</a></li></ul>
</section>
"""


def daily_brief_vi(s: MonitorState) -> str:
    """Vietnamese daily brief — same deterministic template logic as the
    English one, written independently (not word-for-word translated) so it
    reads naturally."""
    word = REGIME_WORDS_VI.get(s.label, s.label)
    dir_w = "tăng" if s.chg_1w >= 0 else "giảm"
    dir_m = "tăng" if s.chg_1m >= 0 else "giảm"
    parts = [
        f"Giá cà phê kỳ hạn (chuẩn arabica) chốt ở {s.price:,.0f}¢/lb, "
        f"{dir_w} {abs(s.chg_1w):.1%} trong tuần và {dir_m} "
        f"{abs(s.chg_1m):.1%} trong tháng. Mô hình đánh giá thị trường đang "
        f"ở trạng thái {word}, duy trì {max(s.dwell, 1)} phiên."
    ]
    if s.rain_z is not None:
        stage_vi = STAGE_VI.get(s.crop_stage, s.crop_stage)
        band_vi = STRESS_VI.get(s.crop_stress_band, s.crop_stress_band)
        stage_bit = (f" Cây cà phê đang trong giai đoạn {stage_vi}; mức rủi ro "
                     f"thời tiết theo giai đoạn: {band_vi}." if stage_vi else "")
        if s.rain_z <= -1.0:
            parts.append(
                f"Lượng mưa quanh Buôn Ma Thuột thấp hơn nhiều so với mức "
                f"bình thường theo mùa (dị thường 30 ngày {s.rain_z:+.1f}σ), "
                f"kịch bản rủi ro nguồn cung robusta điển hình.{stage_bit}")
        elif s.rain_z >= 1.0:
            parts.append(
                f"Lượng mưa Tây Nguyên cao hơn mức bình thường theo mùa "
                f"({s.rain_z:+.1f}σ), độ ẩm thuận lợi cho vùng "
                f"robusta.{stage_bit}")
        else:
            parts.append(
                f"Lượng mưa Tây Nguyên gần mức bình thường theo mùa "
                f"({s.rain_z:+.1f}σ).{stage_bit}")
    if s.brl_chg_21d is not None:
        if s.brl_chg_21d >= 0.02:
            parts.append(
                f"Đồng real Brazil yếu đi ({s.brl_chg_21d:+.1%} trong một "
                f"tháng) làm tăng doanh thu nội tệ của nông dân Brazil, "
                f"thêm áp lực bán ra thị trường thế giới.")
        elif s.brl_chg_21d <= -0.02:
            parts.append(
                f"Đồng real Brazil mạnh lên ({s.brl_chg_21d:+.1%} trong một "
                f"tháng), yếu tố hỗ trợ giá cà phê.")
        else:
            parts.append(
                f"Đồng real Brazil ít thay đổi trong tháng "
                f"({s.brl_chg_21d:+.1%}), bối cảnh tiền tệ trung tính.")
    if s.outlook is not None:
        o = s.outlook
        parts.append(
            f"Dự báo 14 ngày tới: khoảng {o.expected_mm:.0f} mm mưa so với "
            f"mức trung bình mùa vụ {o.norm_mm:.0f} mm ({o.anom_z:+.1f}σ) — "
            f"mức căng thẳng dự kiến cho cây: "
            f"{STRESS_VI.get(o.projected_band, o.projected_band)}.")
    return " ".join(parts)


def render_html_vi(s: MonitorState, brief_vi: str) -> str:
    """The grower page — a focused Vietnamese page with what a Tây Nguyên
    grower cares about: weather and crop-stage risk, market direction in
    plain words, and the farm-gate đồng/kg line. Deliberately NOT a mirror
    of the full English page."""
    p = s.posture
    accent = "#" + p.posture_color
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    stage_vi = STAGE_VI.get(s.crop_stage, s.crop_stage)
    band_vi = STRESS_VI.get(s.crop_stress_band, s.crop_stress_band)
    word_vi = REGIME_WORDS_VI.get(s.label, s.label)
    rain_line = (f"{s.rain_z:+.1f}σ so với mức bình thường theo mùa"
                 if s.rain_z is not None else "không có dữ liệu")
    stress_line = (f"{band_vi} ({s.crop_stress:.2f})"
                   if s.crop_stress is not None else "không có dữ liệu")
    farm_gate_html = ""
    if s.farm_gate_line_vi:
        farm_gate_html = (
            f'<div class="stat"><div class="k">Giá quy đổi tại vườn</div>'
            f'<div class="v" style="font-size:15px">{s.farm_gate_line_vi}'
            f' <span style="opacity:.6">({s.farm_gate_asof})</span></div></div>')

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Theo dõi Cà phê Robusta — thời tiết Tây Nguyên và thị trường</title>
<meta name="description" content="Trang miễn phí, cập nhật mỗi ngày: điều kiện thời tiết vùng trồng robusta Tây Nguyên, giai đoạn sinh trưởng của cây cà phê, và diễn biến thị trường cà phê thế giới.">
<style>
  :root {{ --bg: {PAGE_BG}; --surface: {SURFACE}; --ink: {INK};
          --muted: {MUTED}; --hairline: {HAIRLINE}; --accent: {accent}; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--ink);
    font: 16px/1.7 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }}
  .wrap {{ max-width: 820px; margin: 0 auto; padding: 0 24px 80px; }}
  header {{ padding: 48px 0 24px; border-bottom: 1px solid var(--hairline); }}
  .kicker {{ font-size: 12px; letter-spacing: .16em; text-transform: uppercase;
             color: var(--muted); }}
  .lang {{ float: right; font-size: 13px; color: var(--muted); }}
  .lang a {{ color: var(--ink); text-decoration-color: var(--muted); }}
  h1 {{ font-family: Georgia, serif; font-weight: 500; font-size: 30px;
        margin-top: 10px; line-height: 1.25; }}
  .sub {{ color: var(--muted); margin-top: 10px; max-width: 640px; }}
  .hero {{ margin: 32px 0; padding: 26px; background: var(--surface);
           border: 1px solid var(--hairline); border-left: 3px solid var(--accent); }}
  .stats {{ display: flex; flex-wrap: wrap; gap: 24px; margin-top: 6px; }}
  .stat .k {{ font-size: 12px; letter-spacing: .1em; text-transform: uppercase;
              color: var(--muted); }}
  .stat .v {{ font-size: 18px; margin-top: 2px; }}
  .brief {{ margin-top: 18px; }}
  section {{ margin-top: 44px; }}
  h2 {{ font-family: Georgia, serif; font-weight: 500; font-size: 21px;
        padding-bottom: 10px; border-bottom: 1px solid var(--hairline); }}
  .asof {{ color: var(--muted); font-size: 13px; margin: 8px 0 14px; }}
  figure img {{ width: 100%; height: auto; border: 1px solid var(--hairline); }}
  figcaption {{ color: var(--muted); font-size: 13.5px; margin-top: 10px; }}
  footer {{ margin-top: 60px; padding-top: 20px;
            border-top: 1px solid var(--hairline); color: var(--muted);
            font-size: 13px; }}
  footer p {{ margin-top: 8px; }}
  footer a {{ color: var(--ink); text-decoration-color: var(--muted); }}
</style>
</head>
<body>
<div class="wrap">

<header>
  <div class="lang"><a href="../">English</a> · Tiếng Việt</div>
  <div class="kicker">ascent-agri · dành cho người trồng cà phê</div>
  <h1>Theo dõi Cà phê Robusta</h1>
  <p class="sub">Trang miễn phí, tự cập nhật mỗi ngày làm việc: thời tiết
  vùng trồng robusta Tây Nguyên, giai đoạn sinh trưởng của cây, và diễn biến
  thị trường thế giới, viết bằng ngôn ngữ dễ hiểu. Toàn bộ mã nguồn và dữ
  liệu đều mở.</p>
</header>

<div class="hero">
  <div class="stats">
    <div class="stat"><div class="k">Giai đoạn của cây</div>
      <div class="v">{stage_vi}</div></div>
    <div class="stat"><div class="k">Rủi ro thời tiết theo giai đoạn</div>
      <div class="v">{stress_line}</div></div>
    <div class="stat"><div class="k">Lượng mưa 30 ngày</div>
      <div class="v">{rain_line}</div></div>
    <div class="stat"><div class="k">Thị trường</div>
      <div class="v">{word_vi}</div></div>
    {farm_gate_html}
  </div>
  <p class="brief">{brief_vi}</p>
</div>

<section>
  <h2>Thời tiết vùng trồng — Buôn Ma Thuột, Đắk Lắk</h2>
  <p class="asof">dữ liệu thời tiết đến {s.weather_asof}</p>
  <figure>
    <img src="../assets/weather.png" alt="Dị thường lượng mưa và chuỗi ngày khô tại Buôn Ma Thuột">
    <figcaption>Biểu đồ trên: lượng mưa 30 ngày so với chính địa phương này
    trong một năm gần nhất (dưới dải màu là thiếu mưa). Biểu đồ dưới: tỷ lệ
    ngày gần như không mưa trong 30 ngày qua. Cùng một mức thiếu mưa nhưng
    tác động khác nhau tùy giai đoạn: thiếu nước lúc ra hoa (tháng 1–3) gây
    hại nặng nhất, còn mưa nhiều vào vụ thu hoạch (tháng 10–12) lại là rủi
    ro phơi sấy.</figcaption>
  </figure>
</section>

<section>
  <h2>Thị trường thế giới</h2>
  <p class="asof">dữ liệu giá đến {s.price_asof} · chuẩn arabica KC=F (chuỗi
  robusta liên tục đang được xây dựng từ các hợp đồng riêng lẻ)</p>
  <figure>
    <img src="../assets/price_regime.png" alt="Giá cà phê kỳ hạn với trạng thái thị trường">
    <figcaption>Giá cà phê kỳ hạn, tô màu theo trạng thái thị trường mà mô
    hình nhận diện (vùng không tô = xu hướng tăng ổn định; vàng = căng
    thẳng; đỏ = khủng hoảng).</figcaption>
  </figure>
</section>

<footer>
  <p><strong style="color:var(--ink)">Đây không phải lời khuyên đầu tư.</strong>
  Dự án giáo dục, mã nguồn mở, phi lợi nhuận. Dữ liệu: Yahoo Finance,
  Open-Meteo. Bản dịch tiếng Việt đang được hoàn thiện — góp ý xin gửi
  <a href="https://github.com/ScottDongKhang/ascent-agri/issues">tại đây</a>.</p>
  <p>Trang đầy đủ (nghiên cứu, API dữ liệu, phương pháp):
  <a href="../">bản tiếng Anh</a> · Scott Dong · cập nhật {updated}</p>
</footer>

</div>
<script data-goatcounter="https://ascent-agri.goatcounter.com/count"
        async src="//gc.zgo.at/count.js"></script>
</body>
</html>
"""


# ── charts ──────────────────────────────────────────────────────────────────

def _save(fig, path: Path):
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def chart_price_regime(s: MonitorState, out: Path, lookback: int = 504):
    """Price with regime shading. Calm uptrend is the UNSHADED default state —
    only departures from calm (stressed / crisis / euphoric / uncertain) get a
    soft band, so the chart stays quiet."""
    import matplotlib.patches as mpatches

    close = s.close.iloc[-lookback:]
    sig = s.signals.reindex(close.index, method="ffill")
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    lbl = sig["label"].fillna("uncertain")
    blocks = (lbl != lbl.shift()).cumsum()
    shaded = set()
    for _, seg in lbl.groupby(blocks):
        name = str(seg.iloc[0])
        if name == "calm_bull":
            continue
        ax.axvspan(seg.index[0], seg.index[-1],
                   color=REGIME_COLORS.get(name, MUTED), alpha=0.10, lw=0)
        shaded.add(name)
    ax.plot(close.index, close.values, color=INK, lw=1.5)
    ax.set_ylabel("¢/lb")

    handles = [mpatches.Patch(facecolor=SURFACE, edgecolor=HAIRLINE,
                              label="calm uptrend (unshaded)")]
    handles += [mpatches.Patch(color=REGIME_COLORS[n], alpha=0.45,
                               label=REGIME_WORDS.get(n, n))
                for n in ["euphoric", "stressed", "crisis", "uncertain"]
                if n in shaded]
    ax.legend(handles=handles, loc="upper left", ncol=min(len(handles), 4),
              fontsize=8.5)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    _save(fig, out)


def chart_weather(s: MonitorState, out: Path, lookback: int = 756):
    panel = s.feature_panel.iloc[-lookback:]
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 5.2), sharex=True)
    if "rain_anom_30d" in panel:
        axes[0].plot(panel.index, panel["rain_anom_30d"], color=AQUA, lw=1.3)
    axes[0].axhline(0, color=MUTED, lw=0.8)
    axes[0].axhspan(-3.5, -1.0, color=YELLOW, alpha=0.08, lw=0)
    axes[0].set_ylabel("z-score")
    axes[0].set_title("30-day rainfall anomaly vs trailing year — below the "
                      "shaded band = dry-side supply risk", loc="left",
                      fontsize=9.5, color=MUTED)
    if "dry_frac_30d" in panel:
        axes[1].plot(panel.index, panel["dry_frac_30d"], color=YELLOW, lw=1.3)
    axes[1].set_ylabel("fraction of dry days")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Dry-spell intensity — share of near-rainless days, past 30",
                      loc="left", fontsize=9.5, color=MUTED)
    axes[1].xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    fig.tight_layout()
    _save(fig, out)


def chart_brl(s: MonitorState, out: Path, lookback_days: int = 730):
    brl = s.brl[s.brl.index >= s.brl.index[-1] - pd.Timedelta(days=lookback_days)]
    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    ax.plot(brl.index, brl.values, color=ORANGE, lw=1.4)
    ax.set_ylabel("BRL per USD")
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    _save(fig, out)


# ── page ────────────────────────────────────────────────────────────────────

def render_html(s: MonitorState, brief: str, ledger_html: str = "",
                long_view_html: str = "", brazil_html: str = "",
                forecast_html: str = "") -> str:
    p = s.posture
    accent = "#" + p.posture_color
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    chg_fmt = lambda x: f"{x:+.1%}"
    rain_line = (f"{s.rain_z:+.1f}σ vs seasonal norm" if s.rain_z is not None
                 else "n/a")
    dry_line = (f"{s.dry_frac:.0%} of the last 30 days near-rainless"
                if s.dry_frac is not None else "n/a")
    brl_line = (f"{s.brl_chg_21d:+.1%} over the past month"
                if s.brl_chg_21d is not None else "n/a")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Robusta Coffee Monitor — regimes, weather, and the Vietnamese crop</title>
<meta name="description" content="A daily, model-driven read on coffee markets and Vietnamese robusta growing conditions: market regime, Central Highlands rainfall anomalies, and the Brazilian real.">
<meta property="og:title" content="Robusta Coffee Monitor">
<meta property="og:description" content="Daily market regime, Central Highlands weather anomalies, and currency pressure — an open agricultural market-intelligence project.">
<link rel="alternate" type="application/rss+xml" title="Robusta Coffee Monitor — daily brief" href="feed.xml">
<style>
  :root {{
    --bg: {PAGE_BG}; --surface: {SURFACE}; --ink: {INK}; --muted: {MUTED};
    --hairline: {HAIRLINE}; --accent: {accent};
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg); color: var(--ink);
    font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ max-width: 880px; margin: 0 auto; padding: 0 24px 96px; }}
  header {{ padding: 56px 0 28px; border-bottom: 1px solid var(--hairline); }}
  .kicker {{
    font-size: 12px; letter-spacing: .18em; text-transform: uppercase;
    color: var(--muted);
  }}
  h1 {{
    font-family: Georgia, "Times New Roman", serif; font-weight: 500;
    font-size: 34px; line-height: 1.2; margin-top: 10px;
  }}
  .sub {{ color: var(--muted); margin-top: 10px; max-width: 620px; }}
  .hero {{
    margin: 36px 0; padding: 28px; background: var(--surface);
    border: 1px solid var(--hairline); border-left: 3px solid var(--accent);
  }}
  .posture {{
    font-family: Georgia, serif; font-size: 26px; text-transform: capitalize;
  }}
  .posture small {{ color: var(--muted); font-size: 15px; font-family: inherit; }}
  .stats {{ display: flex; flex-wrap: wrap; gap: 28px; margin-top: 18px; }}
  .stat .k {{ font-size: 12px; letter-spacing: .12em; text-transform: uppercase;
             color: var(--muted); }}
  .stat .v {{ font-size: 19px; margin-top: 2px; }}
  .brief {{ margin-top: 20px; color: var(--ink); max-width: 720px; }}
  section {{ margin-top: 56px; }}
  h2 {{
    font-family: Georgia, serif; font-weight: 500; font-size: 22px;
    padding-bottom: 10px; border-bottom: 1px solid var(--hairline);
  }}
  .asof {{ color: var(--muted); font-size: 13px; margin: 8px 0 16px; }}
  figure img {{ width: 100%; height: auto; border: 1px solid var(--hairline); }}
  figcaption {{ color: var(--muted); font-size: 13.5px; margin-top: 10px;
               max-width: 720px; }}
  .methods p, .methods li {{ color: var(--muted); font-size: 15px; }}
  .methods ul {{ margin: 12px 0 0 20px; }}
  .methods a, footer a {{ color: var(--ink); text-decoration-color: var(--muted); }}
  footer {{
    margin-top: 72px; padding-top: 24px; border-top: 1px solid var(--hairline);
    color: var(--muted); font-size: 13.5px;
  }}
  footer p {{ margin-top: 8px; }}
</style>
</head>
<body>
<div class="wrap">

<header>
  <div style="float:right;font-size:13px;color:var(--muted)">English ·
    <a href="vi/" style="color:var(--ink);text-decoration-color:var(--muted)">Tiếng Việt</a></div>
  <div class="kicker">ascent-agri · daily agricultural market intelligence</div>
  <h1>Robusta Coffee Monitor</h1>
  <p class="sub">A model-driven read on coffee markets and the Vietnamese crop:
  market regime from a hidden-Markov engine, rainfall anomalies in the Central
  Highlands robusta belt, and producer-currency pressure from the Brazilian
  real. Rebuilt daily; everything open source.</p>
</header>

<div class="hero">
  <div class="posture">{p.posture}
    <small>{"· regime: " + REGIME_WORDS.get(s.label, s.label) + " "
            if REGIME_WORDS.get(s.label, s.label) != p.posture else ""}· {max(s.dwell,1)} sessions ·
    exposure guide ×{p.risk_multiplier:.2f}</small>
  </div>
  <div class="stats">
    <div class="stat"><div class="k">Coffee (benchmark)</div>
      <div class="v">{s.price:,.0f}¢/lb</div></div>
    <div class="stat"><div class="k">1 week</div>
      <div class="v">{chg_fmt(s.chg_1w)}</div></div>
    <div class="stat"><div class="k">1 month</div>
      <div class="v">{chg_fmt(s.chg_1m)}</div></div>
    <div class="stat"><div class="k">Highlands rainfall</div>
      <div class="v">{rain_line}</div></div>
    <div class="stat"><div class="k">BRL / USD</div>
      <div class="v">{brl_line}</div></div>
  </div>
  <p class="brief">{brief}</p>
</div>

<section>
  <h2>Market regime</h2>
  <p class="asof">price data through {s.price_asof}</p>
  <figure>
    <img src="assets/price_regime.png" alt="Coffee futures price with regime shading">
    <figcaption>Coffee futures (ICE arabica KC=F — the compliant daily
    benchmark; the hand-built roll-adjusted robusta series joins the page as
    contract history is assembled). Shading is the regime engine's smoothed
    state: a 3-state Gaussian HMM over price, currency and weather features,
    with asymmetric hysteresis so defensive transitions trigger faster than
    upgrades.</figcaption>
  </figure>
</section>

{long_view_html}

<section>
  <h2>Growing conditions — Central Highlands, Vietnam</h2>
  <p class="asof">weather data through {s.weather_asof} · Buon Ma Thuot, Dak Lak
  (12.68 N, 108.04 E) · {dry_line}</p>
  <p class="asof"><strong style="color:var(--ink)">crop stage:
  {s.crop_stage}</strong>{f" · stage-weighted stress: {s.crop_stress_band} ({s.crop_stress:.2f})" if s.crop_stress is not None else ""}
  — {s.crop_stage_note}</p>
  {f'<p class="asof">{s.farm_gate_line} <span style="opacity:.7">({s.farm_gate_asof})</span></p>' if s.farm_gate_line else ""}
  <figure>
    <img src="assets/weather.png" alt="Rainfall anomaly and dry-spell intensity at Buon Ma Thuot">
    <figcaption>Vietnam grows most of the world's robusta, and most of that in
    the Central Highlands. Each series is compared only to its own trailing
    year, so the anomaly is seasonal-aware and fully causal — and the stress
    read is phenology-weighted: the same rainfall deficit matters far more
    during dry-season flowering (Jan–Mar), when a failed blossom cannot be
    recovered, than during harvest (Oct–Dec), when wetness, not drought, is
    the risk. Persistent dry anomalies here have preceded the major robusta
    squeezes.</figcaption>
  </figure>
</section>

{forecast_html}

{brazil_html}

<section>
  <h2>Currency driver — the Brazilian real</h2>
  <p class="asof">FX data through {s.brl_asof}</p>
  <figure>
    <img src="assets/brl.png" alt="BRL per USD">
    <figcaption>Brazil is the largest coffee producer; growers sell dollars
    and spend reais. A weakening real raises local-currency revenue per bag
    and historically adds selling pressure to world prices — one of the
    regime model's inputs.</figcaption>
  </figure>
</section>

{ledger_html}

<section class="methods" id="data">
  <h2>Data &amp; API</h2>
  <p>Organizations and builders can consume this monitor programmatically —
  everything below is regenerated with each daily run, versioned, and free
  under CC BY 4.0 (attribution required). Derived series only: regime labels,
  risk multipliers and weather anomalies computed by this project.</p>
  <ul>
    <li><a href="api/latest.json"><code>api/latest.json</code></a> — today's
        regime, posture, exposure guide, growing-condition anomalies, and the
        daily brief (stable schema, versioned).</li>
    <li><a href="api/history.csv"><code>api/history.csv</code></a> — the full
        daily history of regime labels, risk multipliers and rainfall
        anomalies.</li>
    <li><a href="feed.xml"><code>feed.xml</code></a> — the daily brief as RSS.</li>
    <li><a href="https://github.com/ScottDongKhang/ascent-agri/blob/main/data/ledger/forecasts.jsonl">the
        public ledger</a> — the model's append-only, scoreable track record.</li>
  </ul>
  <p>Using this in your shop or newsletter? <a
  href="https://github.com/ScottDongKhang/ascent-agri/issues">Open an issue</a>
  or email — custom growing regions can be added on request.</p>
</section>

<section class="methods">
  <h2>Research</h2>
  <p>The question behind this page — <em>do growing-region weather anomalies
  actually predict coffee futures returns?</em> — is tested properly in the
  project's working paper: an event study with a-priori definitions,
  permutation inference, and a built-in placebo structure (each region's
  weather vs the <em>other</em> region's crop). Headline: Brazilian dry
  spells precede arabica rallies with the right sign at every horizon
  (+5.1pp over 5 days, n=4, uncorrected p=0.04); the placebo is null; the
  Vietnam→robusta test is still data-limited — and the famous 2021 frost
  exposed a real flaw in threshold-based event definitions.</p>
  <ul>
    <li><a href="assets/weather-and-coffee-returns.pdf">Read the paper
        (PDF)</a> · <a href="https://github.com/ScottDongKhang/ascent-agri/blob/main/docs/research/weather-and-coffee-returns.md">markdown
        + reproduction commands</a></li>
  </ul>
</section>

<section class="methods">
  <h2>Methods, honestly</h2>
  <p>This monitor is the public face of an open research project: a
  single-instrument port of a multi-agent equity platform, rebuilt for
  agricultural markets.</p>
  <ul>
    <li>Regime: 3-state Gaussian HMM on trailing price, BRL/USD and weather
        features; entropy filter, asymmetric hysteresis, and a rule-based
        crisis override.</li>
    <li>Weather: Open-Meteo daily observations at Buon Ma Thuot, z-scored
        against their own trailing year (causal — no future data).</li>
    <li>Everything above is descriptive. The project's own walk-forward
        backtest of a trading strategy on these signals is deliberately
        published with its unflattering result (negative walk-forward
        efficiency) — the point is evaluation machinery that makes lying to
        yourself hard.</li>
    <li>Code, tests (120), and the full backtest notebook:
        <a href="https://github.com/ScottDongKhang/ascent-agri">github.com/ScottDongKhang/ascent-agri</a>.</li>
  </ul>
</section>

<footer>
  <p><strong style="color:var(--ink)">Data</strong> — prices: Yahoo Finance
  (KC=F, BRL=X) · weather: Open-Meteo · regime &amp; anomalies: computed by
  this project. Panels carry their own as-of dates; the page rebuilds only
  after a fully successful data refresh, so it can go stale but not wrong.</p>
  <p><strong style="color:var(--ink)">This is not investment advice.</strong>
  An educational, open-source agricultural market-intelligence project.
  Code MIT-licensed; derived data (regime labels, anomalies, briefs)
  CC BY 4.0 — free to use with attribution.</p>
  <p>Built by Scott Dong ·
  <a href="feed.xml">RSS daily brief</a> ·
  <a href="https://github.com/ScottDongKhang/ascent-agri/issues">suggest a
  feature or report something wrong</a> · last updated {updated}</p>
</footer>

</div>
<script data-goatcounter="https://ascent-agri.goatcounter.com/count"
        async src="//gc.zgo.at/count.js"></script>
</body>
</html>
"""


# ── build ───────────────────────────────────────────────────────────────────

def build(out_dir: Path = DEFAULT_OUT) -> Path:
    logging.basicConfig(level=logging.ERROR)
    warnings.filterwarnings("ignore")

    close, brl, weather = load_inputs()
    print(f"[site] inputs: price→{close.index[-1].date()} "
          f"brl→{brl.index[-1].date()} weather→{weather.index[-1].date()}")

    state = compute_state(close, brl, weather)
    print(f"[site] regime: {state.label} (posture {state.posture.posture}, "
          f"risk ×{state.posture.risk_multiplier})")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    assets = out_dir / "assets"
    assets.mkdir(parents=True)

    chart_price_regime(state, assets / "price_regime.png")
    chart_weather(state, assets / "weather.png")
    chart_brl(state, assets / "brl.png")

    brief = daily_brief(state)

    ledger_score = score_ledger(read_ledger())
    has_ledger_chart = chart_ledger(ledger_score, assets / "ledger.png")
    ledger_html = render_ledger_section(ledger_score, has_ledger_chart)

    # long-view trend chart (always available — same close series)
    chart_long_view(state, assets / "long_view.png")
    long_view_html = render_long_view_section(state)

    # Brazil / arabica panel — optional layer, never publish-blocking
    brazil_html = ""
    try:
        from ascentagri.macro_fetch import load_weather_brazil
        wx_br = load_weather_brazil()
        if wx_br is not None and len(wx_br) > 400:
            chart_brazil(wx_br, assets / "brazil.png")
            brazil_html = render_brazil_section(str(wx_br.index[-1].date()))
    except Exception as exc:
        log.warning("Brazil panel skipped: %s", exc)

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

    (out_dir / "index.html").write_text(render_html(
        state, brief, ledger_html, long_view_html, brazil_html,
        forecast_html=forecast_html))
    (out_dir / ".nojekyll").write_text("")

    vi_dir = out_dir / "vi"
    vi_dir.mkdir(parents=True, exist_ok=True)
    (vi_dir / "index.html").write_text(render_html_vi(state, daily_brief_vi(state)))

    site_url = "https://scottdongkhang.github.io/ascent-agri/"
    (out_dir / "feed.xml").write_text(render_feed(state, site_url))

    api_dir = out_dir / "api"
    api_dir.mkdir(parents=True, exist_ok=True)
    (api_dir / "latest.json").write_text(render_api_latest(state, brief))
    (api_dir / "history.csv").write_text(render_api_history(state))

    paper = ROOT / "docs" / "research" / "weather-and-coffee-returns.pdf"
    if paper.exists():
        shutil.copy(paper, assets / paper.name)

    print(f"[site] brief: {brief}")
    print(f"[site] wrote {out_dir / 'index.html'} (+feed.xml, paper={paper.exists()})")
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    build(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
