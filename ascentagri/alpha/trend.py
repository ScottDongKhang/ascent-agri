"""ascentagri/alpha/trend.py — trend-following alpha for a single series.

Ported from Ascent Capital's alpha/trend.py with one structural change:
cross-sectional z-score normalization (which silently zeroes out at N=1)
is replaced by a rolling time-series z-score (see alpha/normalize.py).
Signal composition and weights are unchanged.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

from .normalize import ts_normalize


def trend_alpha(
    features: Dict[str, pd.Series],
    z_window: int = 252,
    z_min_periods: int = 60,
) -> pd.Series:
    """Composite trend-following alpha.
    Combines momentum, MACD, and SMA crossover signals.
    Returns: Series (dates) of alpha scores in z-units."""
    def _norm(s: pd.Series) -> pd.Series:
        return ts_normalize(s, window=z_window, min_periods=z_min_periods)

    components = []
    weights = []

    # Medium-term momentum (highest weight)
    if "mom_63d" in features:
        components.append(_norm(features["mom_63d"]))
        weights.append(0.30)

    # Skip-last-month momentum (11-1): 12m minus last month — avoids
    # short-term reversal. Preferred over raw 252d when available.
    if "mom_skip1m" in features:
        components.append(_norm(features["mom_skip1m"]))
        weights.append(0.20)
    elif "mom_126d" in features:
        components.append(_norm(features["mom_126d"]))
        weights.append(0.20)

    # 6-month momentum (secondary long-term signal)
    if "mom_126d" in features and "mom_skip1m" in features:
        components.append(_norm(features["mom_126d"]))
        weights.append(0.10)

    # Short-term momentum (lower weight — more noise)
    if "mom_21d" in features:
        components.append(_norm(features["mom_21d"]))
        weights.append(0.15)

    # MACD histogram
    if "macd_hist" in features:
        components.append(_norm(features["macd_hist"]))
        weights.append(0.20)

    # SMA crossover
    if "sma_cross_10_50" in features:
        components.append(_norm(features["sma_cross_10_50"]))
        weights.append(0.15)

    if not components:
        return pd.Series(dtype=float)

    # Weighted combination
    total_w = sum(weights)
    alpha = sum(c * (w / total_w) for c, w in zip(components, weights))
    return alpha
