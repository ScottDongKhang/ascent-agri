"""ascentagri/regime/breaks.py — ported near-verbatim from Ascent Capital.

Structural break / change-point detection. DIAGNOSTIC ONLY:
  • Flags dates where the latent regime model may be stale.
  • Recommends re-fit triggers.
  • Never drives live regime decisions directly.

Uses the `ruptures` library (Pelt or Binseg). Gracefully skips if unavailable
(ruptures is an optional dependency — everything else works without it).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .types import BreakResult

log = logging.getLogger(__name__)

try:
    import ruptures as rpt
    _HAS_RUPTURES = True
except ImportError:
    _HAS_RUPTURES = False
    log.info("regime.breaks: ruptures not installed — break detection disabled")


# ── helpers ───────────────────────────────────────────────────────────────

def _clean_series(s: pd.Series, min_points: int = 63) -> Optional[np.ndarray]:
    """Return cleaned numpy array or None if insufficient data."""
    cleaned = s.dropna()
    if len(cleaned) < min_points:
        return None
    return cleaned.values.reshape(-1, 1)


def _penalty_from_series(s: np.ndarray, multiplier: float = 2.0) -> float:
    """Heuristic BIC-like penalty: multiplier * log(n) * sigma^2."""
    n = len(s)
    sigma2 = float(np.var(s))
    return multiplier * np.log(n) * sigma2


# ── main detector ─────────────────────────────────────────────────────────

class BreakDetector:
    """Detects structural breaks in one or more feature series.

    min_size           : minimum samples between breakpoints (ruptures param)
    penalty_multiplier : scale factor for the automatic penalty
    algorithm          : 'pelt' (exact) or 'binseg' (approximate, faster)
    """

    def __init__(
        self,
        min_size: int = 63,
        penalty_multiplier: float = 2.0,
        algorithm: str = "pelt",
    ):
        self.min_size = min_size
        self.penalty_multiplier = penalty_multiplier
        self.algorithm = algorithm

    def detect(
        self,
        feature_panel: pd.DataFrame,
        target_columns: Optional[List[str]] = None,
    ) -> Dict[str, BreakResult]:
        """Run break detection on selected feature columns.
        Returns dict mapping feature_name -> BreakResult."""
        if not _HAS_RUPTURES:
            log.debug("regime.breaks: ruptures unavailable — returning empty results")
            return {}

        cols = target_columns or list(feature_panel.columns)
        results: Dict[str, BreakResult] = {}

        for col in cols:
            if col not in feature_panel.columns:
                continue
            series = feature_panel[col]
            arr = _clean_series(series, min_points=self.min_size * 2)
            if arr is None:
                log.debug(f"regime.breaks: insufficient data for column '{col}'")
                continue

            try:
                result = self._detect_one(col, arr, series.dropna().index)
                if result is not None:
                    results[col] = result
            except Exception as exc:
                log.warning(f"regime.breaks: detection failed for '{col}': {exc}")

        log.info(
            f"regime.breaks: detected breaks in {len(results)}/{len(cols)} features"
        )
        return results

    def _detect_one(
        self, col: str, arr: np.ndarray, date_index: pd.DatetimeIndex
    ) -> Optional[BreakResult]:
        penalty = _penalty_from_series(arr, self.penalty_multiplier)
        penalty = max(penalty, 1e-6)

        if self.algorithm == "pelt":
            algo = rpt.Pelt(model="rbf", min_size=self.min_size, jump=5).fit(arr)
        else:
            algo = rpt.Binseg(model="rbf", min_size=self.min_size, jump=5).fit(arr)

        breakpoints = algo.predict(pen=penalty)

        # ruptures returns indices (1-indexed end positions), last is len(arr)
        bp_indices = [bp - 1 for bp in breakpoints if bp < len(arr)]

        if not bp_indices:
            return BreakResult(
                feature_name=col,
                break_dates=[],
                n_breaks=0,
                confidence=0.0,
            )

        bp_dates = [date_index[i] for i in bp_indices if i < len(date_index)]

        # Confidence proxy: ratio of between-segment variance to total variance
        segments = np.split(arr.ravel(), [bp + 1 for bp in bp_indices])
        seg_means = np.array([np.mean(s) for s in segments if len(s) > 0])
        between_var = float(np.var(seg_means)) if len(seg_means) > 1 else 0.0
        total_var = float(np.var(arr.ravel())) + 1e-12
        confidence = min(1.0, between_var / total_var)

        return BreakResult(
            feature_name=col,
            break_dates=bp_dates,
            n_breaks=len(bp_dates),
            confidence=confidence,
        )

    def detect_aggregate(
        self,
        feature_panel: pd.DataFrame,
        target_columns: Optional[List[str]] = None,
    ) -> pd.Series:
        """Return a single time series of 'break intensity' — the count of
        features that experienced a break on or near each date."""
        results = self.detect(feature_panel, target_columns)
        if not results:
            return pd.Series(0.0, index=feature_panel.index)

        intensity = pd.Series(0.0, index=feature_panel.index)
        for result in results.values():
            for bd in result.break_dates:
                if bd in intensity.index:
                    intensity[bd] += result.confidence

        return intensity

    def latest_zscore(
        self,
        feature_panel: pd.DataFrame,
        lookback_recent: int = 5,
        lookback_baseline: int = 63,
    ) -> float:
        """Continuous anomaly signal: mean absolute z-score of the recent window
        relative to the historical baseline, averaged across features.
        Returns 0.0 if feature_panel is None or history is insufficient."""
        if feature_panel is None:
            return 0.0
        if len(feature_panel) < lookback_baseline + lookback_recent + 10:
            return 0.0

        try:
            recent = feature_panel.iloc[-lookback_recent:].values
            baseline = feature_panel.iloc[
                -(lookback_baseline + lookback_recent):-lookback_recent
            ].values

            zscores: List[float] = []
            for col_idx in range(feature_panel.shape[1]):
                rec_col = recent[:, col_idx].astype(float)
                base_col = baseline[:, col_idx].astype(float)
                # Skip if too many NaNs
                if np.sum(np.isfinite(base_col)) < 10:
                    continue
                base_mean = float(np.nanmean(base_col))
                base_std = float(np.nanstd(base_col))
                if base_std < 1e-9:
                    continue
                rec_mean = float(np.nanmean(rec_col[np.isfinite(rec_col)]))
                z = abs(rec_mean - base_mean) / base_std
                zscores.append(z)

            if not zscores:
                return 0.0
            return float(np.mean(zscores))
        except Exception as exc:
            log.warning("regime.breaks.latest_zscore failed: %s", exc)
            return 0.0

    def should_refit(
        self,
        feature_panel: pd.DataFrame,
        as_of_date: pd.Timestamp,
        lookback_days: int = 63,
        min_break_confidence: float = 0.3,
        min_break_features: int = 2,
    ) -> bool:
        """Heuristic: recommend a model re-fit if there are ≥ min_break_features
        features with a structural break in the last lookback_days.
        Diagnostic recommendation only — never drives a hard switch."""
        if not _HAS_RUPTURES:
            return False

        recent_panel = feature_panel.loc[:as_of_date].tail(lookback_days + 63)
        results = self.detect(recent_panel)

        n_recent_breaks = 0
        for res in results.values():
            for bd in res.break_dates:
                lookback_start = as_of_date - pd.Timedelta(days=lookback_days)
                if bd >= lookback_start and res.confidence >= min_break_confidence:
                    n_recent_breaks += 1
                    break  # count feature once even if multiple breaks

        recommend = n_recent_breaks >= min_break_features
        if recommend:
            log.info(
                f"regime.breaks: refit recommended as of {as_of_date.date()} — "
                f"{n_recent_breaks} features show recent structural breaks"
            )
        return recommend
