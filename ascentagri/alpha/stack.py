"""ascentagri/alpha/stack.py — minimal alpha combiner (full rewrite).

Ascent Capital's stack.py blends 5+ sleeves across a 900-symbol universe
with IC gating, sparse-panel handling and regime reweighting. The
single-series analog keeps exactly the pieces that survive N=1:

  1. build features from the close series (same indicator definitions
     as the source's feature_defs.py, on Series instead of DataFrames)
  2. score two sleeves (trend + meanrev), each TS-z-normalized
  3. combine with regime-adjusted sleeve weights (per-date)
  4. map the combined score to a LONG-ONLY position in [0, max_exposure]
  5. scale by the regime risk multiplier, then the vol-target and
     200d-MA exposure overlays

The long-only mapping is deliberate: this is an agricultural systems
case study, not a trading pitch — no short coffee positions.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import AgriConfig
from .trend import trend_alpha
from .meanrev import meanrev_alpha
from .vol_sizing import apply_exposure_overlays

log = logging.getLogger(__name__)


# ── feature computation (single-series versions of feature_defs.py) ───────

def build_features(close: pd.Series) -> Dict[str, pd.Series]:
    """Compute the indicator set consumed by the trend and meanrev sleeves.
    All features use only past data — safe for walk-forward use."""
    close = close.sort_index()
    features: Dict[str, pd.Series] = {}

    # Momentum
    for w in [5, 21, 63, 126, 252]:
        features[f"mom_{w}d"] = close.pct_change(w)
    # Skip-last-month momentum (11-1): 12m return minus last month
    features["mom_skip1m"] = features["mom_252d"] - features["mom_21d"]

    # Mean reversion inputs
    rolling_mean = close.rolling(20, min_periods=10).mean()
    rolling_std = close.rolling(20, min_periods=10).std()
    features["zscore_20d"] = (close - rolling_mean) / rolling_std.replace(0, np.nan)

    sma = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = sma + 2.0 * std
    lower = sma - 2.0 * std
    features["bb_pct_20d"] = (close - lower) / (upper - lower).replace(0, np.nan)

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    features["rsi_14"] = 100 - (100 / (1 + rs))

    # Trend indicators
    sma_fast = close.rolling(10).mean()
    sma_slow = close.rolling(50).mean()
    features["sma_cross_10_50"] = sma_fast / sma_slow.replace(0, np.nan) - 1

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=9, adjust=False).mean()
    features["macd_hist"] = (macd - sig) / close.replace(0, np.nan)

    return features


# ── sleeve weighting ───────────────────────────────────────────────────────

def sleeve_weight_frame(
    dates: pd.Index,
    base_weights: Dict[str, float],
    regime_signal_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Per-date sleeve weights (dates × sleeves), regime-adjusted and
    renormalized. Falls back to base weights on dates without a signal."""
    w = pd.DataFrame(
        {s: float(bw) for s, bw in base_weights.items()},
        index=dates,
    )
    if regime_signal_df is None or regime_signal_df.empty:
        return w

    adj_cols = {s: f"sleeve_{s}" for s in base_weights}
    available = [c for c in adj_cols.values() if c in regime_signal_df.columns]
    if not available:
        return w

    # As-of alignment: each date uses the latest signal at or before it
    adj = regime_signal_df[available].reindex(dates, method="ffill").fillna(0.0)
    for sleeve, col in adj_cols.items():
        if col in adj.columns:
            w[sleeve] = (w[sleeve] + adj[col]).clip(lower=0.0)

    # Renormalize each row; rows that collapse to zero fall back to base
    row_sum = w.sum(axis=1)
    collapsed = row_sum <= 1e-12
    w = w.div(row_sum.replace(0, np.nan), axis=0)
    if collapsed.any():
        for s, bw in base_weights.items():
            w.loc[collapsed, s] = bw
    return w


# ── long-only score → position mapping ─────────────────────────────────────

def score_to_position(
    score: pd.Series,
    score_cap: float = 1.5,
    max_exposure: float = 1.0,
) -> pd.Series:
    """Map a z-scored composite signal to a long-only exposure.

    position = max_exposure * clip(score, 0, score_cap) / score_cap

    Negative scores go flat (never short); a score of +score_cap or better
    is fully invested. Monotonic and easy to explain.
    """
    clipped = score.clip(lower=0.0, upper=score_cap)
    return (clipped / score_cap) * max_exposure


# ── main entry point ───────────────────────────────────────────────────────

def build_positions(
    close: pd.Series,
    regime_signal_df: Optional[pd.DataFrame] = None,
    config: Optional[AgriConfig] = None,
    sleeve_weight_override: Optional[Dict[str, float]] = None,
) -> Tuple[pd.Series, Dict]:
    """Build the daily target-exposure series for the coffee series.

    close            : continuous close prices (chronological)
    regime_signal_df : output of RegimeEngine.get_signal_series() — used for
                       per-date sleeve reweighting and the risk multiplier.
                       None → base weights, no regime scaling.
    config           : AgriConfig (defaults if None)
    sleeve_weight_override : replaces config.sleeve_weights when given
                       (e.g. from the meta-learner)

    Returns (positions in [0, max_exposure], diagnostics dict). Diagnostics
    include the per-sleeve score series for charts and IC analysis.
    """
    cfg = config or AgriConfig()
    base_weights = dict(sleeve_weight_override or cfg.sleeve_weights)

    features = build_features(close)
    sleeves: Dict[str, pd.Series] = {}
    if base_weights.get("trend", 0) > 0:
        sleeves["trend"] = trend_alpha(
            features, z_window=cfg.ts_z_window, z_min_periods=cfg.ts_z_min_periods)
    if base_weights.get("meanrev", 0) > 0:
        sleeves["meanrev"] = meanrev_alpha(
            features, z_window=cfg.ts_z_window, z_min_periods=cfg.ts_z_min_periods)

    loaded = {s: v for s, v in sleeves.items() if v is not None and not v.empty}
    if not loaded:
        log.warning("alpha.stack: no sleeves loaded — flat positions")
        flat = pd.Series(0.0, index=close.index)
        return flat, {"sleeves": {}, "note": "no sleeves loaded"}

    # Only blend sleeves that loaded; renormalize their base weights
    active_base = {s: base_weights[s] for s in loaded}
    total = sum(active_base.values())
    active_base = {s: w / total for s, w in active_base.items()}

    weights = sleeve_weight_frame(close.index, active_base, regime_signal_df)
    composite = sum(loaded[s].reindex(close.index).fillna(0.0) * weights[s]
                    for s in loaded)

    positions = score_to_position(
        composite, score_cap=cfg.score_cap, max_exposure=cfg.max_exposure)

    # Regime risk multiplier (as-of aligned, causal)
    if regime_signal_df is not None and "risk_multiplier" in regime_signal_df.columns:
        risk_mult = (regime_signal_df["risk_multiplier"]
                     .reindex(close.index, method="ffill").fillna(1.0))
        positions = (positions * risk_mult).clip(lower=0.0)

    # Exposure overlays: 200d MA filter + vol targeting
    positions, overlay_meta = apply_exposure_overlays(
        positions, close,
        target_vol=cfg.vol_target,
        vol_floor=cfg.vol_floor,
        vol_cap=cfg.vol_cap,
        ma_multiplier=cfg.ma_multiplier,
        ma_window=cfg.ma_window,
        rebalance_only=False,   # engine handles rebalance discretization
    )
    positions = positions.clip(lower=0.0, upper=cfg.max_exposure)

    diagnostics = {
        "sleeves": loaded,
        "composite_score": composite,
        "sleeve_weights_used": active_base,
        "overlay": overlay_meta,
    }
    return positions, diagnostics
