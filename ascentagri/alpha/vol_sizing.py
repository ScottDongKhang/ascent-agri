"""ascentagri/alpha/vol_sizing.py — exposure overlays for a single series.

Port of Ascent Capital's portfolio/exposure.py `ma_filter_scale` and
`vol_target_scale`. Changes: the overlays operate on the coffee series
itself instead of SPY, and the VIX confirmation gate is dropped (no
options-implied vol series exists for robusta) — the MA cut fires on the
moving average alone, as in the source's legacy behavior.

All computations are causal: only data strictly before (vol) or up to (MA)
each decision date is used.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MA_WINDOW          = 200
MA_MIN_PERIODS     = 150
MA_CUT_MULTIPLIER  = 0.70
VOL_TARGET         = 0.20    # robusta runs ~30% annualized vol
VOL_LOOKBACK       = 21
VOL_FLOOR          = 0.25
VOL_CAP            = 1.00


def ma_filter_scale(
    close: pd.Series,
    dates: pd.Index,
    ma_window: int = MA_WINDOW,
    ma_min_periods: int = MA_MIN_PERIODS,
    multiplier: float = MA_CUT_MULTIPLIER,
) -> pd.Series:
    """Per-date exposure multiplier from the price-vs-own-MA filter.

    Returns a Series indexed by `dates`: `multiplier` where the close is
    below its trailing MA, else 1.0. Dates before the MA has enough history
    default to 1.0 (no cut on unknowable information).
    """
    close = close.sort_index()
    close = close[~close.index.duplicated(keep="last")]
    ma = close.rolling(ma_window, min_periods=ma_min_periods).mean()
    below = close < ma

    aligned = below.reindex(dates, method="ffill").fillna(False)
    return pd.Series(np.where(aligned, multiplier, 1.0), index=dates)


def vol_target_scale(
    close: pd.Series,
    dates: pd.Index,
    target_vol: float = VOL_TARGET,
    lookback: int = VOL_LOOKBACK,
    floor: float = VOL_FLOOR,
    cap: float = VOL_CAP,
) -> pd.Series:
    """Per-date exposure multiplier targeting `target_vol` annualized, using
    trailing `lookback`-day realized vol of the series itself.

    scale(d) = clip(target_vol / realized_vol(d), floor, cap), where
    realized_vol(d) uses returns strictly before d (fully causal).
    Dates with <5 trailing observations get scale 1.0.
    """
    close = close.sort_index()
    close = close[~close.index.duplicated(keep="last")]
    rets = close.pct_change().dropna()

    # Vectorized causal computation: rolling vol shifted one day so that the
    # value at d uses returns strictly before d.
    trailing = (rets.rolling(lookback, min_periods=5).std() * np.sqrt(252)).shift(1)
    trailing = trailing.reindex(dates, method="ffill")

    scales = (target_vol / trailing.replace(0, np.nan)).clip(lower=floor, upper=cap)
    return scales.fillna(1.0)


def apply_exposure_overlays(
    positions: pd.Series,
    close: pd.Series,
    rebalance_only: bool = True,
    target_vol: float = VOL_TARGET,
    vol_floor: float = VOL_FLOOR,
    vol_cap: float = VOL_CAP,
    ma_multiplier: float = MA_CUT_MULTIPLIER,
    ma_window: int = MA_WINDOW,
    vol_targeting_enabled: bool = True,
) -> "tuple[pd.Series, dict]":
    """Apply MA filter then vol targeting to a per-date exposure Series.

    With rebalance_only=True (default) the combined scale is computed on
    dates where the underlying position changes and held constant between
    them — matching live behavior, where exposure is set at rebalance and
    not adjusted daily. Returns (scaled_positions, meta).
    """
    if positions.empty:
        return positions, {"ma_cut_dates": 0, "mean_vol_scale": 1.0}

    dates = positions.index

    ma_scale = ma_filter_scale(close, dates, ma_window=ma_window,
                               multiplier=ma_multiplier)
    if vol_targeting_enabled:
        v_scale = vol_target_scale(close, dates, target_vol=target_vol,
                                   floor=vol_floor, cap=vol_cap)
    else:
        v_scale = pd.Series(1.0, index=dates)

    combined = ma_scale * v_scale

    if rebalance_only and len(positions) > 1:
        changed = positions.diff().abs() > 1e-12
        changed.iloc[0] = True
        combined = combined.where(changed).ffill().fillna(1.0)

    scaled = positions * combined
    meta = {
        "ma_cut_dates": int((ma_scale < 1.0).sum()),
        "mean_vol_scale": round(float(v_scale.mean()), 4),
        "min_vol_scale": round(float(v_scale.min()), 4),
    }
    return scaled, meta
