"""ascentagri/regime/integration.py — regime signals → position sizing.

Moderate rewrite of Ascent Capital's regime/integration.py: sleeve names
remapped to this project's two scored sleeves ({trend, meanrev}), and the
multi-name portfolio helpers (sector caps, max-weight redistribution,
covariance half-life) reduced to their single-instrument analogs.

API contract (all pure functions):
  regime_scale_exposure(exposure, signal)        -> scaled exposure
  regime_adjust_sleeve_weights(base, signal)     -> adjusted sleeve weights
  regime_signal_threshold(base, signal)          -> score threshold bump
  regime_rebalance_band(base, signal)            -> rebalance tolerance band
  get_signal_for_date(signal_df, date)           -> RegimeSignal (causal)
  build_regime_series(signal_df)                 -> pd.Series of labels
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from .types import RegimeLabel, RegimeSignal

log = logging.getLogger(__name__)

# ── Base alpha sleeve weights (must sum to 1) ─────────────────────────────
_BASE_SLEEVE_WEIGHTS: Dict[str, float] = {
    "trend": 0.75,
    "meanrev": 0.25,
}


# ── A. Gross exposure control ─────────────────────────────────────────────

def regime_scale_exposure(
    exposure: Union[float, pd.Series],
    signal: Optional[RegimeSignal],
) -> Union[float, pd.Series]:
    """Apply the regime risk multiplier to a target exposure (scalar or
    per-date Series). Long-only constraint enforced (no short positions)."""
    if signal is None:
        return exposure

    mult = signal.risk_multiplier
    scaled = exposure * mult
    if isinstance(scaled, pd.Series):
        scaled = scaled.clip(lower=0.0)
    else:
        scaled = max(0.0, scaled)

    if mult != 1.0:
        log.info(
            f"regime.integration: gross exposure scaled by {mult:.2f} "
            f"(regime={signal.label.value})"
        )
    return scaled


# ── B. Alpha sleeve reweighting ───────────────────────────────────────────

def regime_adjust_sleeve_weights(
    base_sleeve_weights: Optional[Dict[str, float]] = None,
    signal: Optional[RegimeSignal] = None,
    adjustment_scale: float = 1.0,
) -> Dict[str, float]:
    """Return regime-adjusted alpha sleeve weights, normalized to sum to 1.

    adjustment_scale scales the deltas (0 = no change, 1 = full adjustment).
    Falls back to base weights when the signal carries no adjustments or the
    adjusted weights collapse to zero.
    """
    base = dict(base_sleeve_weights or _BASE_SLEEVE_WEIGHTS)

    if signal is None or not signal.sleeve_adjustments:
        return base

    adjusted = {}
    for sleeve, base_w in base.items():
        delta = signal.sleeve_adjustments.get(sleeve, 0.0) * adjustment_scale
        adjusted[sleeve] = max(0.0, base_w + delta)

    # Renormalize
    total = sum(adjusted.values())
    if total <= 0:
        log.warning("regime.integration: sleeve weights summed to zero — using base")
        return base

    normalized = {k: v / total for k, v in adjusted.items()}
    log.debug(f"regime.integration: sleeve weights adjusted for {signal.label.value}: {normalized}")
    return normalized


def adjust_sleeve_weights_for_label(
    label: str,
    base_sleeve_weights: Optional[Dict[str, float]] = None,
    sleeve_adjustments: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, float]:
    """Vector-friendly variant: adjust sleeve weights from a plain label
    string plus an adjustment map (as stored in the signal-cache DataFrame)."""
    from .types import REGIME_CONFIG_DEFAULTS
    base = dict(base_sleeve_weights or _BASE_SLEEVE_WEIGHTS)
    adj_map = sleeve_adjustments or REGIME_CONFIG_DEFAULTS["regime_sleeve_adjustments"]
    deltas = adj_map.get(str(label).lower(), {})
    adjusted = {s: max(0.0, w + deltas.get(s, 0.0)) for s, w in base.items()}
    total = sum(adjusted.values())
    if total <= 0:
        return base
    return {s: w / total for s, w in adjusted.items()}


# ── C. Signal threshold and rebalance band widening ───────────────────────

def regime_signal_threshold(
    base_threshold: float = 0.0,
    signal: Optional[RegimeSignal] = None,
) -> float:
    """Regime-adjusted signal threshold: in stressed / crisis regimes,
    raise the bar to reduce noise-driven trades."""
    if signal is None:
        return base_threshold

    bump: Dict[str, float] = {
        RegimeLabel.CALM_BULL.value: 0.0,
        RegimeLabel.EUPHORIC.value: 0.05,
        RegimeLabel.STRESSED.value: 0.10,
        RegimeLabel.CRISIS.value: 0.15,
        RegimeLabel.UNCERTAIN.value: 0.08,
    }
    return base_threshold + bump.get(signal.label.value, 0.0)


def regime_rebalance_band(
    base_band: float = 0.02,
    signal: Optional[RegimeSignal] = None,
) -> float:
    """Regime-adjusted rebalance tolerance band: widen in stressed regimes
    to reduce excessive turnover."""
    if signal is None:
        return base_band

    multiplier: Dict[str, float] = {
        RegimeLabel.CALM_BULL.value: 1.0,
        RegimeLabel.EUPHORIC.value: 1.2,
        RegimeLabel.STRESSED.value: 1.5,
        RegimeLabel.CRISIS.value: 2.0,
        RegimeLabel.UNCERTAIN.value: 1.3,
    }
    return base_band * multiplier.get(signal.label.value, 1.0)


# ── Helpers over the signal-cache DataFrame ───────────────────────────────

def build_regime_series(signal_df: pd.DataFrame) -> pd.Series:
    """Given RegimeDecisionEngine.process_to_frame() output, return a simple
    pd.Series of RegimeLabel values indexed by date."""
    if "label" not in signal_df.columns:
        return pd.Series(dtype=str)
    return signal_df["label"].map(RegimeLabel.from_str)


def get_signal_for_date(
    signal_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> Optional[RegimeSignal]:
    """Retrieve the most recent RegimeSignal available as of as_of_date from
    a pre-computed signal DataFrame. Uses only data up to as_of_date."""
    idx = signal_df.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    available = signal_df.loc[idx <= pd.Timestamp(as_of_date).tz_localize(None)]
    if available.empty:
        log.warning(f"regime.integration: no signal available as of {pd.Timestamp(as_of_date).date()}")
        return None

    row = available.iloc[-1]
    k_cols = [c for c in signal_df.columns if c.startswith("prob_")]
    probs = np.array([row[c] for c in sorted(k_cols)]) if k_cols else np.array([1.0])

    # Reconstruct sleeve_adjustments from columns
    sleeve_cols = [c for c in signal_df.columns if c.startswith("sleeve_")]
    sleeve_adj = {c.replace("sleeve_", ""): float(row[c]) for c in sleeve_cols}

    return RegimeSignal(
        date=row.name,
        probs=probs,
        label=RegimeLabel.from_str(str(row["label"])),
        entropy=float(row.get("entropy", 0.5)),
        transition_flag=bool(row.get("transition_flag", False)),
        risk_multiplier=float(row.get("risk_multiplier", 1.0)),
        sleeve_adjustments=sleeve_adj,
        dwell_days=int(row.get("dwell_days", 0)),
    )
