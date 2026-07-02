"""ascentagri/regime/engine.py — top-level RegimeEngine.

Moderate rewrite of Ascent Capital's regime/engine.py for a single
agricultural instrument. Kept: feature building → optional walk-forward K
selection → model fit → hysteresis decision layer → signal caching, plus the
rule-based crisis override on top of the HMM. Dropped: equity-only machinery
(VIX confirmation, SPY/TLT emergency triggers, AI blend, particle filter).

The crisis override is re-tuned for coffee: instead of "VIX > 30 AND SPY
5d < -7%", it fires when the coffee series' own 21d realized vol exceeds a
threshold AND the 5-day return crashes — same idea (a fast rule layer that
does not wait for the HMM), volatility-confirmed by the instrument itself.

Usage:
    engine = RegimeEngine(config_overrides)
    engine.fit(prices, brl_usd, weather)
    signal = engine.get_signal(as_of_date)      # causal
    frame  = engine.get_signal_series()         # full cache for backtests
"""
from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .features import RegimeFeatureBuilder
from .model import RegimeModel, walk_forward_model_select
from .breaks import BreakDetector
from .decision import RegimeDecisionEngine
from .integration import get_signal_for_date
from .types import RegimeLabel, RegimeScorecard, RegimeSignal, REGIME_CONFIG_DEFAULTS

log = logging.getLogger(__name__)


def apply_crisis_override(
    signal_df: pd.DataFrame,
    prices: pd.Series,
    ret_5d_threshold: float = -0.10,
    vol_threshold: float = 0.45,
) -> pd.DataFrame:
    """Force label='crisis' and risk_multiplier=0.40 on any day where the
    5-day return is below ret_5d_threshold AND 21d annualized realized vol
    exceeds vol_threshold.

    Adds boolean column 'crisis_override' — True only on rule-overridden rows,
    not on HMM-native crisis days, so rule vs model can be audited downstream.
    """
    if signal_df.empty or prices is None:
        if "crisis_override" not in signal_df.columns:
            signal_df = signal_df.copy()
            signal_df["crisis_override"] = False
        return signal_df

    df = signal_df.copy()
    if "crisis_override" not in df.columns:
        df["crisis_override"] = False

    ret_5d = prices.pct_change(5).reindex(df.index, method="ffill")
    rvol_21d = (
        prices.pct_change().rolling(21, min_periods=10).std() * np.sqrt(252)
    ).reindex(df.index, method="ffill")

    trigger_mask = (
        (ret_5d.fillna(0) < ret_5d_threshold)
        & (rvol_21d.fillna(0) > vol_threshold)
        & (df["label"] != "crisis")   # don't double-flag HMM-native crisis
    )

    n_triggered = int(trigger_mask.sum())
    if n_triggered > 0:
        df.loc[trigger_mask, "label"] = "crisis"
        df.loc[trigger_mask, "risk_multiplier"] = 0.40
        df.loc[trigger_mask, "crisis_override"] = True
        log.warning(
            "regime.engine: crisis override triggered on %d days "
            "(5d ret < %.0f%% AND 21d vol > %.0f%%)",
            n_triggered, ret_5d_threshold * 100, vol_threshold * 100,
        )

    return df


class RegimeEngine:
    """Regime engine for the continuous coffee series.

    Parameters
    ----------
    config : dict, optional
        Override dict for any REGIME_CONFIG_DEFAULTS keys.
    """

    def __init__(self, config: Optional[Dict] = None):
        self._cfg = dict(REGIME_CONFIG_DEFAULTS)
        if config:
            self._cfg.update(config)

        self._model: Optional[RegimeModel] = None
        self._decision_engine: Optional[RegimeDecisionEngine] = None
        self._break_detector = BreakDetector(
            min_size=self._cfg["regime_break_min_size"],
            penalty_multiplier=self._cfg["regime_break_penalty_multiplier"],
        )
        self._signal_cache: Optional[pd.DataFrame] = None
        self._feature_panel: Optional[pd.DataFrame] = None
        self._best_k: int = 3
        self._scorecard: Optional[RegimeScorecard] = None
        self._last_fit_date: Optional[pd.Timestamp] = None
        self._fitted = False
        self._prices: Optional[pd.Series] = None

    # ── public fit API ────────────────────────────────────────────────────

    def fit(
        self,
        prices: pd.Series,
        brl_usd: Optional[pd.Series] = None,
        weather: Optional[pd.DataFrame] = None,
        run_model_selection: bool = True,
        k_override: Optional[int] = None,
        hmm_restarts: int = 10,
    ) -> None:
        """Build features and fit the regime model.

        prices   : coffee close prices (chronological, DatetimeIndex)
        brl_usd  : BRL-per-USD series (optional, graceful fallback)
        weather  : growing-region weather frame (optional)
        run_model_selection : if True and enough data, run walk-forward K
            selection before the final fit; else use k_override or the middle
            candidate.
        """
        log.info("regime.engine: starting fit (%d rows)", len(prices))

        # 1. Build feature panel
        builder = RegimeFeatureBuilder(
            prices=prices,
            brl_usd=brl_usd,
            weather=weather,
            lookbacks=tuple(self._cfg["regime_feature_lookbacks"]),
            jump_threshold=self._cfg["regime_jump_threshold"],
        )
        self._feature_panel = builder.build()

        # 2. Walk-forward model selection (optional)
        min_days = self._cfg["regime_training_min_days"]
        selection_feasible = (
            len(self._feature_panel)
            >= self._cfg["regime_wf_train_days"] + self._cfg["regime_wf_test_days"]
        )
        if k_override is not None:
            self._best_k = int(k_override)
            log.info("regime.engine: K=%d fixed by caller", self._best_k)
        elif run_model_selection and selection_feasible:
            ret = prices.pct_change().reindex(self._feature_panel.index)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                best_k, scorecard = walk_forward_model_select(
                    feature_panel=self._feature_panel,
                    benchmark_returns=ret,
                    candidate_k=self._cfg["regime_n_candidates"],
                    train_days=self._cfg["regime_wf_train_days"],
                    test_days=self._cfg["regime_wf_test_days"],
                    step_days=self._cfg["regime_wf_step_days"],
                    min_train=min_days,
                )
            self._best_k = best_k
            self._scorecard = scorecard
            log.info(
                f"regime.engine: model selection chose K={best_k}, "
                f"composite_score={scorecard.composite_score:.4f}"
            )
        else:
            cands = self._cfg["regime_n_candidates"]
            self._best_k = cands[len(cands) // 2]
            log.info(f"regime.engine: skipping model selection, using K={self._best_k} default")

        # 3. Fit production model
        self._model = RegimeModel(k_regimes=self._best_k, hmm_restarts=hmm_restarts)
        success = self._model.fit(self._feature_panel)
        if not success:
            log.warning("regime.engine: model fit failed — using uniform probs")

        # 4. Compute probability series and run decision layer
        self._decision_engine = RegimeDecisionEngine(
            state_labels=self._model.state_labels if self._model else {},
            enter_threshold=self._cfg["regime_enter_threshold"],
            exit_threshold=self._cfg["regime_exit_threshold"],
            min_dwell_days=self._cfg["regime_min_dwell_days"],
            entropy_uncertain_threshold=self._cfg["regime_entropy_uncertain_threshold"],
            downgrade_threshold=self._cfg.get("regime_downgrade_threshold", 0.40),
            upgrade_threshold=self._cfg.get("regime_upgrade_threshold", 0.70),
            risk_multipliers=self._cfg["regime_risk_multiplier"],
            sleeve_adjustments=self._cfg["regime_sleeve_adjustments"],
        )

        if self._model and self._model._fitted:
            prob_df = self._model.predict_probs(self._feature_panel)
        else:
            k = self._best_k
            prob_df = pd.DataFrame(
                np.full((len(self._feature_panel), k), 1.0 / k),
                index=self._feature_panel.index,
                columns=list(range(k)),
            )

        self._signal_cache = self._decision_engine.process_to_frame(prob_df)
        self._signal_cache = apply_crisis_override(
            self._signal_cache,
            prices,
            ret_5d_threshold=self._cfg["regime_crisis_ret_5d_threshold"],
            vol_threshold=self._cfg["regime_crisis_vol_threshold"],
        )
        self._last_fit_date = self._feature_panel.index[-1]
        self._fitted = True
        self._prices = prices

        # Diagnostic: regime occupancy
        if self._signal_cache is not None and "label" in self._signal_cache.columns:
            occ = self._signal_cache["label"].value_counts().to_dict()
            total = len(self._signal_cache)
            pct = {k: f"{v} ({100 * v / total:.1f}%)" for k, v in sorted(occ.items())}
            log.info(
                f"REGIME ENGINE COMPLETE best_k={self._best_k} "
                f'backend={self._model.active_backend if self._model else "none"} '
                f"dates={total} occupancy={pct}"
            )

    # ── public signal API ─────────────────────────────────────────────────

    def get_signal(self, as_of_date: pd.Timestamp) -> Optional[RegimeSignal]:
        """Return the regime signal for as_of_date.
        Uses only data available up to that date (causal)."""
        if not self._fitted or self._signal_cache is None:
            log.warning("regime.engine: get_signal called before fit — returning None")
            return None
        return get_signal_for_date(self._signal_cache, as_of_date)

    def predict_signal_frame(self, feature_panel: pd.DataFrame,
                             prices: Optional[pd.Series] = None) -> pd.DataFrame:
        """Score NEW feature rows with the already-fitted model + decision
        layer (used by the walk-forward runner to score test windows with a
        train-fitted model — fully causal).

        Note the hysteresis machine is re-run over the provided panel from its
        first row, so pass train+test features together and slice afterwards
        if warm-started smoothing is desired.
        """
        if not self._fitted or self._model is None or self._decision_engine is None:
            raise RuntimeError("RegimeEngine.fit() must be called first")
        if self._model._fitted:
            prob_df = self._model.predict_probs(feature_panel)
        else:
            k = self._best_k
            prob_df = pd.DataFrame(
                np.full((len(feature_panel), k), 1.0 / k),
                index=feature_panel.index, columns=list(range(k)),
            )
        frame = self._decision_engine.process_to_frame(prob_df)
        px = prices if prices is not None else self._prices
        if px is not None:
            frame = apply_crisis_override(
                frame, px,
                ret_5d_threshold=self._cfg["regime_crisis_ret_5d_threshold"],
                vol_threshold=self._cfg["regime_crisis_vol_threshold"],
            )
        return frame

    def get_signal_series(self) -> pd.DataFrame:
        """Return the full signal DataFrame (for backtesting / diagnostics)."""
        return self._signal_cache.copy() if self._signal_cache is not None else pd.DataFrame()

    def get_feature_panel(self) -> pd.DataFrame:
        return self._feature_panel.copy() if self._feature_panel is not None else pd.DataFrame()

    # ── refit trigger ─────────────────────────────────────────────────────

    def should_refit(self, as_of_date: pd.Timestamp) -> bool:
        """True if a model refit is recommended: scheduled interval elapsed,
        or the structural break detector recommends one."""
        if not self._fitted or self._last_fit_date is None:
            return True

        days_since_fit = (as_of_date - self._last_fit_date).days
        if days_since_fit >= self._cfg["regime_refit_every_days"]:
            log.info(
                f"regime.engine: scheduled refit triggered ({days_since_fit} days since last fit)"
            )
            return True

        if self._feature_panel is not None:
            return self._break_detector.should_refit(
                self._feature_panel,
                as_of_date=as_of_date,
            )

        return False

    # ── break detection (diagnostic) ──────────────────────────────────────

    def detect_breaks(self, target_columns: Optional[List[str]] = None) -> Dict:
        """Run structural break detection on the feature panel (diagnostic)."""
        if self._feature_panel is None:
            return {}
        return self._break_detector.detect(self._feature_panel, target_columns)

    def break_intensity_series(self) -> pd.Series:
        """Time series of aggregate break intensity (diagnostic)."""
        if self._feature_panel is None:
            return pd.Series(dtype=float)
        return self._break_detector.detect_aggregate(self._feature_panel)

    # ── scorecard access ──────────────────────────────────────────────────

    @property
    def scorecard(self) -> Optional[RegimeScorecard]:
        return self._scorecard

    @property
    def best_k(self) -> int:
        return self._best_k

    @property
    def state_labels(self) -> Dict[int, RegimeLabel]:
        if self._model:
            return self._model.state_labels
        return {}
