"""Rolling time-series z-score normalization.

THE key rewrite in the alpha port. Ascent Capital's sleeves normalized
cross-sectionally (z-score across symbols per date). With a single
instrument (N=1) that computation silently returns 0 for every date —
std across one value is NaN, filled to 0 — so every ported sleeve would
emit all-zero scores. Here every signal is normalized against its own
trailing history instead.

Causality: mean and std at date t use only observations up to and
including t (a plain trailing window) — no future data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ts_normalize(
    s: pd.Series,
    window: int = 252,
    min_periods: int = 60,
    clip: float = 3.0,
) -> pd.Series:
    """Rolling time-series z-score of a signal against its own trailing
    `window`-day distribution. Clipped to ±clip, NaN → 0 (no signal)."""
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std()
    z = (s - mean) / std.replace(0, np.nan)
    return z.clip(-clip, clip).fillna(0.0)
