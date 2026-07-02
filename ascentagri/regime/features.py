"""ascentagri/regime/features.py — regime feature panel builder.

Moderate rewrite of Ascent Capital's regime/features.py for a single
agricultural instrument:
  * keeps the univariate market-state features (returns, vol, drawdown,
    MA distance, skew/kurt, jump frequency) computed on the coffee series
  * drops the cross-sectional group (breadth/dispersion need a universe)
  * replaces the equity stress proxies (VIX, HYG/LQD credit, TLT/IEF curve)
    with the two drivers relevant to robusta:
      - BRL/USD (Brazil is the dominant producer; a weakening BRL raises
        Brazilian producers' local-currency revenue per bag — bearish coffee)
      - Central Highlands weather anomalies (rainfall / temperature at
        Buon Ma Thuot — supply-side shock proxy for Vietnamese robusta)

Design principles preserved from the source:
  • Every feature is computed with TRAILING windows only — no future leakage.
  • Missing optional data (FX, weather) is handled gracefully.
  • All outputs are aligned on the SAME DatetimeIndex as the input prices.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── helpers ───────────────────────────────────────────────────────────────

def _safe_rolling(s: pd.Series, window: int, func: str = "mean",
                  min_periods: Optional[int] = None) -> pd.Series:
    """Apply a rolling function with sane min_periods fallback."""
    mp = min_periods if min_periods is not None else max(1, window // 2)
    return getattr(s.rolling(window, min_periods=mp), func)()


def _log_return(prices: pd.Series) -> pd.Series:
    """Compute daily log returns."""
    return np.log(prices / prices.shift(1))


def _trailing_vol(returns: pd.Series, window: int) -> pd.Series:
    """Annualized trailing realized vol."""
    return _safe_rolling(returns, window, "std") * np.sqrt(252)


def _downside_vol(returns: pd.Series, window: int, threshold: float = 0.0) -> pd.Series:
    """Annualized downside semi-volatility."""
    downside = returns.clip(upper=threshold)
    return _safe_rolling(downside, window, "std") * np.sqrt(252)


def _rolling_drawdown(prices: pd.Series, window: int) -> pd.Series:
    """Rolling drawdown from peak within last `window` days."""
    peak = prices.rolling(window, min_periods=1).max()
    return (prices - peak) / peak


def _skew(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window, min_periods=window // 2).skew()


def _kurt(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window, min_periods=window // 2).kurt()


def _trailing_anomaly_z(s: pd.Series, window: int, baseline: int = 365) -> pd.Series:
    """Z-score of the rolling `window`-day aggregate against its own trailing
    `baseline`-day distribution. Fully causal: mean/std are computed on data
    strictly before each date (shifted by one day)."""
    agg = s.rolling(window, min_periods=max(3, window // 2)).mean()
    base_mean = agg.shift(1).rolling(baseline, min_periods=baseline // 3).mean()
    base_std = agg.shift(1).rolling(baseline, min_periods=baseline // 3).std()
    return (agg - base_mean) / base_std.replace(0, np.nan)


# ── main builder ──────────────────────────────────────────────────────────

class RegimeFeatureBuilder:
    """Assembles the regime feature DataFrame for the coffee series.

    Parameters
    ----------
    prices : pd.Series
        Daily close prices for the continuous coffee series.
    brl_usd : pd.Series, optional
        Daily BRL-per-USD rate. Rising = weakening BRL.
    weather : pd.DataFrame, optional
        Daily weather at the growing region, columns [rain_mm, temp_c].
    lookbacks : tuple of int
        Rolling window sizes (days) for most features.
    jump_threshold : float
        Daily |log return| above this counts as a jump.
    """

    def __init__(
        self,
        prices: pd.Series,
        brl_usd: Optional[pd.Series] = None,
        weather: Optional[pd.DataFrame] = None,
        lookbacks: Tuple[int, ...] = (5, 21, 63),
        jump_threshold: float = 0.03,
    ):
        self.prices = prices.copy()
        self.brl = brl_usd.copy() if brl_usd is not None else None
        self.weather = weather.copy() if weather is not None else None
        self.lookbacks = sorted(lookbacks)
        self.jump_threshold = jump_threshold
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if not isinstance(self.prices.index, pd.DatetimeIndex):
            raise TypeError("prices must have a DatetimeIndex")
        if not self.prices.index.is_monotonic_increasing:
            self.prices = self.prices.sort_index()
            log.warning("regime.features: prices sorted ascending")

    # ── Group A: market-state features (univariate, kept from source) ────

    def _build_market_features(self) -> pd.DataFrame:
        ret = _log_return(self.prices)
        features: Dict[str, pd.Series] = {}

        for lb in self.lookbacks:
            features[f"px_ret_{lb}d"] = self.prices.pct_change(lb)

        for lb in self.lookbacks:
            features[f"px_rvol_{lb}d"] = _trailing_vol(ret, lb)

        features["px_downvol_21d"] = _downside_vol(ret, 21)

        for lb in [63, 126]:
            features[f"px_dd_{lb}d"] = _rolling_drawdown(self.prices, lb)

        for ma in [50, 200]:
            ma_val = self.prices.rolling(ma, min_periods=ma // 2).mean()
            features[f"px_dist_ma{ma}"] = (self.prices - ma_val) / ma_val

        features["px_skew_63d"] = _skew(ret, 63)
        features["px_kurt_63d"] = _kurt(ret, 63)

        features["px_jump_freq_21d"] = (
            (ret.abs() > self.jump_threshold).astype(float)
            .rolling(21, min_periods=10).mean()
        )

        return pd.DataFrame(features, index=self.prices.index)

    # ── Group B: FX driver (replaces equity credit/curve group) ──────────

    def _build_fx_features(self) -> pd.DataFrame:
        if self.brl is None:
            log.info("regime.features: no BRL/USD — skipping FX group")
            return pd.DataFrame(index=self.prices.index)

        brl = self.brl.sort_index().reindex(self.prices.index, method="ffill")
        brl_ret = _log_return(brl)
        features: Dict[str, pd.Series] = {
            "brl_chg_21d": brl.pct_change(21),
            "brl_chg_63d": brl.pct_change(63),
            "brl_rvol_21d": _trailing_vol(brl_ret, 21),
        }
        ma = brl.rolling(63, min_periods=30).mean()
        features["brl_dist_ma63"] = (brl - ma) / ma
        log.info("regime.features: BRL/USD features included")
        return pd.DataFrame(features, index=self.prices.index)

    # ── Group C: growing-region weather (replaces VIX stress group) ──────

    def _build_weather_features(self) -> pd.DataFrame:
        if self.weather is None or self.weather.empty:
            log.info("regime.features: no weather data — skipping weather group")
            return pd.DataFrame(index=self.prices.index)

        wx = self.weather.sort_index()
        features: Dict[str, pd.Series] = {}

        if "rain_mm" in wx.columns:
            rain = wx["rain_mm"].astype(float)
            # 30-day rainfall anomaly vs the location's own trailing year —
            # persistent dryness in the Central Highlands is the classic
            # robusta supply shock
            rain_anom = _trailing_anomaly_z(rain, window=30, baseline=365)
            features["rain_anom_30d"] = rain_anom.reindex(
                self.prices.index, method="ffill")
            # dry-spell intensity: fraction of near-zero-rain days, past 30
            dry = (rain < 1.0).astype(float).rolling(30, min_periods=15).mean()
            features["dry_frac_30d"] = dry.reindex(self.prices.index, method="ffill")

        if "temp_c" in wx.columns:
            temp = wx["temp_c"].astype(float)
            temp_anom = _trailing_anomaly_z(temp, window=21, baseline=365)
            features["temp_anom_21d"] = temp_anom.reindex(
                self.prices.index, method="ffill")

        if features:
            log.info("regime.features: weather features included (%s)",
                     ", ".join(features))
        return pd.DataFrame(features, index=self.prices.index)

    # ── public API ───────────────────────────────────────────────────────

    def build(self) -> pd.DataFrame:
        """Build and return the full regime feature panel.

        Returns a DataFrame with DatetimeIndex (same as prices) and one
        column per feature. NaNs are allowed — the model layer handles
        missing data through min_periods and imputation.
        """
        parts = [
            self._build_market_features(),
            self._build_fx_features(),
            self._build_weather_features(),
        ]
        panel = pd.concat([p for p in parts if not p.empty], axis=1)
        panel = panel.reindex(self.prices.index)
        panel = panel.replace([np.inf, -np.inf], np.nan)

        n_features = panel.shape[1]
        n_complete = panel.dropna().shape[0]
        log.info(
            f"regime.features: built {n_features} features, "
            f"{n_complete}/{len(panel)} rows fully complete"
        )
        return panel

    def get_core_features(self) -> pd.DataFrame:
        """Return only the minimal always-available price-feature subset —
        fallback when the optional FX/weather inputs are missing."""
        panel = self.build()
        core_cols = [c for c in panel.columns if c.startswith("px_")]
        return panel[core_cols]
