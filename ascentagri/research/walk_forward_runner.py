"""ascentagri/research/walk_forward_runner.py — FULL REWRITE.

Ascent Capital's walk-forward runner is built around a point-in-time equity
universe: get_universe_on_date() per fold, sector joins, and symbol pivots
inlined throughout. None of that survives a single-instrument port, so this
is a ground-up rewrite as a single-series fold loop with no universe concept.

What each fold does (all causal):
  1. Fit the regime engine on the TRAINING slice only (fixed K, no per-fold
     model selection — parameters are not re-optimized per fold).
  2. Score train+test features with the train-fitted model; the hysteresis
     layer warm-starts on train so test-window smoothing carries no cold
     start. Only test-window signals are consumed out-of-sample.
  3. Build alpha positions from the close series up to test_end. Features
     and z-scores use trailing windows only, so pre-train history is legal.
  4. Run the backtest engine on the TEST window → OOS net returns.
  5. Run the engine on the TRAIN window with the same fold state → IS
     returns, kept only for the Walk-Forward Efficiency ratio.

Failed folds are visible: each failure is recorded with fold id, stage and
exception type, printed in the fold summary, and never silently zeroed.

The purge gap between train end and test start protects against label
leakage from multi-day features crossing the boundary.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..alpha.stack import build_positions
from ..backtest.engine import BacktestEngine
from ..config import AgriConfig
from ..regime.engine import RegimeEngine
from ..regime.features import RegimeFeatureBuilder
from ..regime.types import REGIME_CONFIG_DEFAULTS
from .evaluation import (
    FoldResult,
    arithmetic_sharpe,
    compute_all_metrics,
    walk_forward_efficiency,
)
from .splits import SplitWindow, walk_forward_splits

log = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    oos_returns: pd.Series                     # stitched OOS net returns
    benchmark_returns: pd.Series               # buy-and-hold over same dates
    positions: pd.Series                       # stitched OOS held exposure
    fold_results: List[FoldResult]             # per-fold IS Sharpe + OOS returns
    fold_summaries: List[Dict]                 # per-fold diagnostics (incl. failures)
    report: Dict = field(default_factory=dict)

    @property
    def n_failed_folds(self) -> int:
        return sum(1 for f in self.fold_summaries if f.get("status") == "failed")

    @property
    def equity_curve(self) -> pd.Series:
        return (1 + self.oos_returns).cumprod()


def _series_to_frame(s: pd.Series, name: str) -> pd.DataFrame:
    return s.to_frame(name)


def _run_engine_window(
    positions: pd.Series,
    close: pd.Series,
    open_: pd.Series,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    cfg: AgriConfig,
) -> pd.Series:
    """Run the backtest engine over [window_start, window_end] and return
    net daily returns."""
    mask = (close.index >= window_start) & (close.index <= window_end)
    dates = close.index[mask]
    engine = BacktestEngine(
        initial_capital=cfg.initial_capital,
        spread_bps=cfg.spread_bps,
        impact_bps=cfg.impact_bps,
        rebalance_freq_days=cfg.rebalance_freq_days,
        execution_delay=cfg.execution_delay,
    )
    result = engine.run(
        target_weights=_series_to_frame(positions.reindex(dates).fillna(0.0), cfg.series_name),
        close_prices=_series_to_frame(close.loc[dates], cfg.series_name),
        open_prices=_series_to_frame(open_.loc[dates], cfg.series_name),
    )
    return result.portfolio_returns


def run_walk_forward(
    close: pd.Series,
    open_prices: Optional[pd.Series] = None,
    brl_usd: Optional[pd.Series] = None,
    weather: Optional[pd.DataFrame] = None,
    config: Optional[AgriConfig] = None,
    verbose: bool = True,
) -> WalkForwardResult:
    """Honest walk-forward evaluation of the coffee strategy.

    close       : continuous close prices (chronological DatetimeIndex)
    open_prices : opens if available; None → close is used (rebalance trades
                  then execute at the t+1 close — disclosed approximation)
    brl_usd     : BRL-per-USD series for regime features (optional)
    weather     : growing-region weather frame (optional)
    config      : AgriConfig (defaults if None)
    """
    cfg = config or AgriConfig()
    close = close.sort_index()
    close = close[~close.index.duplicated(keep="last")]
    open_ = (open_prices.sort_index() if open_prices is not None else close).reindex(
        close.index).fillna(close)

    splits = walk_forward_splits(
        close.index,
        train_days=cfg.wf_train_days,
        test_days=cfg.wf_test_days,
        step_days=cfg.wf_step_days,
        purge_days=cfg.wf_purge_days,
        min_train_days=cfg.wf_min_train_days,
    )
    if not splits:
        raise ValueError(
            f"No walk-forward folds possible: {len(close)} rows < "
            f"train {cfg.wf_train_days} + purge {cfg.wf_purge_days} + test window. "
            f"Reduce wf_train_days/wf_test_days or supply more history."
        )
    if verbose:
        print(f"[WF] {len(splits)} folds | train={cfg.wf_train_days}d "
              f"test={cfg.wf_test_days}d step={cfg.wf_step_days}d "
              f"purge={cfg.wf_purge_days}d | {close.index[0].date()} → {close.index[-1].date()}")

    regime_cfg = dict(REGIME_CONFIG_DEFAULTS)
    regime_cfg.update(cfg.regime_overrides)

    oos_chunks: List[pd.Series] = []
    pos_chunks: List[pd.Series] = []
    fold_results: List[FoldResult] = []
    fold_summaries: List[Dict] = []

    t0 = time.time()
    for split in splits:
        summary: Dict = {"fold_id": split.fold_id,
                         "train": f"{split.train_start.date()}→{split.train_end.date()}",
                         "test": f"{split.test_start.date()}→{split.test_end.date()}"}
        stage = "init"
        try:
            # ── 1. regime: fit on training slice only ────────────────────
            stage = "regime_fit"
            train_close = close.loc[split.train_start:split.train_end]
            engine = RegimeEngine(config=regime_cfg)
            engine.fit(
                prices=train_close,
                brl_usd=brl_usd.loc[:split.train_end] if brl_usd is not None else None,
                weather=weather.loc[:split.train_end] if weather is not None else None,
                run_model_selection=False,
                k_override=cfg.wf_regime_k,
                hmm_restarts=cfg.wf_hmm_restarts,
            )

            # ── 2. score train+test with the train-fitted model ──────────
            stage = "regime_predict"
            span_close = close.loc[split.train_start:split.test_end]
            builder = RegimeFeatureBuilder(
                prices=span_close,
                brl_usd=brl_usd.loc[:split.test_end] if brl_usd is not None else None,
                weather=weather.loc[:split.test_end] if weather is not None else None,
                lookbacks=tuple(regime_cfg["regime_feature_lookbacks"]),
                jump_threshold=regime_cfg["regime_jump_threshold"],
            )
            span_panel = builder.build()
            signal_frame = engine.predict_signal_frame(span_panel, prices=span_close)

            # ── 3. alpha positions (trailing-only features) ───────────────
            stage = "alpha"
            positions, _diag = build_positions(
                close.loc[:split.test_end],
                regime_signal_df=signal_frame,
                config=cfg,
            )

            # ── 4. OOS backtest on the test window ────────────────────────
            stage = "backtest_oos"
            oos_ret = _run_engine_window(
                positions, close, open_, split.test_start, split.test_end, cfg)

            # ── 5. IS backtest on the train window (for WFE only) ────────
            stage = "backtest_is"
            is_ret = _run_engine_window(
                positions, close, open_, split.train_start, split.train_end, cfg)
            is_sharpe = arithmetic_sharpe(is_ret)

            oos_chunks.append(oos_ret)
            pos_chunks.append(positions.loc[split.test_start:split.test_end])
            fold_results.append(FoldResult(
                fold_id=split.fold_id, is_sharpe=is_sharpe, oos_returns=oos_ret))
            regime_label = (signal_frame["label"].loc[split.test_start:split.test_end]
                            .mode().iloc[0] if not signal_frame.empty else "n/a")
            summary.update({
                "status": "ok",
                "is_sharpe": round(is_sharpe, 3),
                "oos_sharpe": round(arithmetic_sharpe(oos_ret), 3),
                "oos_days": int(len(oos_ret)),
                "dominant_regime": regime_label,
                "mean_exposure": round(float(
                    positions.loc[split.test_start:split.test_end].mean()), 3),
            })
            if verbose:
                print(f"[WF] fold {split.fold_id:>2} ok   "
                      f"IS Sharpe {is_sharpe:+.2f} | OOS Sharpe "
                      f"{arithmetic_sharpe(oos_ret):+.2f} | regime {regime_label} | "
                      f"mean exp {summary['mean_exposure']:.2f}")
        except Exception as exc:
            # Integrity: failed folds are visible, never silently zeroed
            summary.update({
                "status": "failed",
                "stage": stage,
                "error_type": type(exc).__name__,
                "error": str(exc)[:300],
            })
            log.exception("WF fold %d failed at stage=%s", split.fold_id, stage)
            if verbose:
                print(f"[WF] fold {split.fold_id:>2} FAILED at {stage}: "
                      f"{type(exc).__name__}: {exc}")
        fold_summaries.append(summary)

    if not oos_chunks:
        raise RuntimeError(
            f"All {len(splits)} walk-forward folds failed — see fold summaries")

    oos_returns = pd.concat(oos_chunks).sort_index()
    oos_returns = oos_returns[~oos_returns.index.duplicated(keep="first")]
    positions_stitched = pd.concat(pos_chunks).sort_index()
    positions_stitched = positions_stitched[~positions_stitched.index.duplicated(keep="first")]

    bh_returns = close.pct_change().reindex(oos_returns.index).fillna(0.0)

    report = compute_all_metrics(oos_returns, benchmark_returns=bh_returns)
    report["wfe"] = walk_forward_efficiency(fold_results)
    report["n_folds"] = len(splits)
    report["n_failed_folds"] = sum(
        1 for f in fold_summaries if f.get("status") == "failed")
    report["n_oos_days"] = int(len(oos_returns))
    report["oos_start"] = str(oos_returns.index[0].date())
    report["oos_end"] = str(oos_returns.index[-1].date())
    report["elapsed_sec"] = round(time.time() - t0, 1)

    if verbose:
        n_ok = sum(1 for f in fold_summaries if f.get("status") == "ok")
        print(f"[WF] complete: {n_ok} folds succeeded, "
              f"{report['n_failed_folds']} failed, "
              f"{report['n_oos_days']} OOS days in {report['elapsed_sec']}s")

    return WalkForwardResult(
        oos_returns=oos_returns,
        benchmark_returns=bh_returns,
        positions=positions_stitched,
        fold_results=fold_results,
        fold_summaries=fold_summaries,
        report=report,
    )


def print_wf_report(result: WalkForwardResult) -> None:
    """Print the OOS performance report in the source project's format."""
    r = result.report
    print("=" * 55)
    print("  WALK-FORWARD OOS PERFORMANCE REPORT")
    print("=" * 55)
    print(f"  OOS Window        : {r['oos_start']} → {r['oos_end']}")
    print(f"  OOS Trading Days  : {r['n_oos_days']}")
    print(f"  Folds             : {r['n_folds']}  ({r['n_failed_folds']} failed)")
    print(f"  CAGR              : {r['cagr'] * 100:+.2f}%")
    print(f"  Volatility        : {r['volatility'] * 100:.2f}%")
    print(f"  Sharpe Ratio      : {r['arithmetic_sharpe']:.3f}")
    print(f"  Sortino Ratio     : {r['sortino']:.3f}")
    print(f"  Max Drawdown      : {r['max_drawdown'] * 100:.2f}%")
    print(f"  Win Rate          : {r['hit_rate'] * 100:.1f}%")
    print("-" * 55)
    print(f"  Buy&Hold CAGR     : {r.get('benchmark_return', float('nan')) * 100:+.2f}%")
    print(f"  Alpha vs B&H      : {r.get('alpha', float('nan')) * 100:+.2f}%")
    print(f"  Beta vs B&H       : {r.get('beta', float('nan')):.3f}")
    print("-" * 55)
    wfe = r["wfe"]
    if np.isfinite(wfe):
        label = "acceptable" if wfe >= 0.5 else "OVERFIT — do not trade"
        print(f"  Walk-Forward Eff. : {wfe:.3f}  ({label})")
    else:
        print("  Walk-Forward Eff. : n/a (no folds with positive IS Sharpe)")
    print("=" * 55)
