"""ascentagri/alpha/meanrev.py — short-term reversal alpha for a single series.

Ported from Ascent Capital's alpha/meanrev.py with the same structural change
as trend.py: cross-sectional z-scores (zero at N=1) replaced by rolling
time-series z-scores. Signal composition and weights are unchanged.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

from .normalize import ts_normalize


def meanrev_alpha(
    features: Dict[str, pd.Series],
    z_window: int = 252,
    z_min_periods: int = 60,
) -> pd.Series:
    """Mean reversion alpha: buy oversold, fade overbought.
    Inverts short-term momentum signals. Returns a Series in z-units."""
    def _norm(s: pd.Series) -> pd.Series:
        return ts_normalize(s, window=z_window, min_periods=z_min_periods)

    components = []
    weights = []

    # Short-term reversal (5-day): negative momentum = buy signal
    if "mom_5d" in features:
        components.append(_norm(-features["mom_5d"]))
        weights.append(0.35)

    # Z-score: buy low z-score (below mean)
    if "zscore_20d" in features:
        components.append(_norm(-features["zscore_20d"]))
        weights.append(0.35)

    # RSI: buy oversold
    if "rsi_14" in features:
        rsi_signal = -(features["rsi_14"] - 50) / 50  # <50 = buy signal
        components.append(_norm(rsi_signal))
        weights.append(0.15)

    # Bollinger %B: buy at lower band
    if "bb_pct_20d" in features:
        bb = -(features["bb_pct_20d"] - 0.5)  # <0.5 = buy
        components.append(_norm(bb))
        weights.append(0.15)

    if not components:
        return pd.Series(dtype=float)

    total_w = sum(weights)
    alpha = sum(c * (w / total_w) for c, w in zip(components, weights))
    return ts_normalize(alpha, window=z_window, min_periods=z_min_periods)
