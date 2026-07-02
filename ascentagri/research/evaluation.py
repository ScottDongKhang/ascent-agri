"""ascentagri/research/evaluation.py — performance metrics.

Ported verbatim from Ascent Capital's research/evaluation.py, plus the
Walk-Forward Efficiency machinery from its WF framework (FoldResult +
arithmetic Sharpe + WFE), so research/ stays at the planned three modules.

WFE interpretation
------------------
> 1.0 : OOS beats IS — unusual, check for data leakage
0.5–1.0: Normal degradation — strategy is tradeable
< 0.5 : Significant overfitting — do not trade live
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate."""
    total = (1 + returns).prod()
    n_periods = len(returns)
    if n_periods == 0 or total <= 0:
        return 0.0
    return total ** (periods_per_year / n_periods) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized standard deviation of returns."""
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio (CAGR / vol convention)."""
    vol = annualized_volatility(returns, periods_per_year)
    if vol == 0:
        return 0.0
    ret = annualized_return(returns, periods_per_year)
    return (ret - rf_annual) / vol


def arithmetic_sharpe(returns: pd.Series, rf_annual: float = 0.0,
                      periods_per_year: int = 252) -> float:
    """Arithmetic Sharpe: sqrt(252) * mean(daily_excess) / std(daily_excess).
    Industry-standard convention; used for WFE fold ratios."""
    if len(returns) == 0:
        return 0.0
    rf_daily = rf_annual / periods_per_year
    excess = returns - rf_daily
    std = excess.std()
    if std < 1e-10:
        m = excess.mean()
        return np.inf if m > 0 else (-np.inf if m < 0 else 0.0)
    return float(np.sqrt(periods_per_year) * excess.mean() / std)


def sortino_ratio(returns: pd.Series, rf_annual: float = 0.0, periods_per_year: int = 252) -> float:
    """Sortino ratio using downside deviation."""
    downside = returns[returns < 0]
    if len(downside) == 0:
        return float("inf") if annualized_return(returns) > 0 else 0.0
    downside_vol = downside.std() * np.sqrt(periods_per_year)
    if downside_vol == 0:
        return 0.0
    ret = annualized_return(returns, periods_per_year)
    return (ret - rf_annual) / downside_vol


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown (as negative fraction)."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """CAGR / |max drawdown|."""
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return 0.0
    return annualized_return(returns, periods_per_year) / mdd


def turnover(weights: pd.DataFrame) -> pd.Series:
    """Daily turnover: sum of absolute weight changes."""
    return weights.diff().abs().sum(axis=1)


def average_turnover(weights: pd.DataFrame) -> float:
    """Average daily one-way turnover."""
    t = turnover(weights)
    return t.mean() / 2  # one-way


def hit_rate(returns: pd.Series) -> float:
    """Fraction of positive return days."""
    if len(returns) == 0:
        return 0.0
    return (returns > 0).mean()


def profit_factor(returns: pd.Series) -> float:
    """Sum of gains / sum of losses."""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


# ── Walk-Forward Efficiency ────────────────────────────────────────────────

@dataclass
class FoldResult:
    """Per-fold IS Sharpe and OOS returns for WFE computation."""
    fold_id: int
    is_sharpe: float
    oos_returns: pd.Series


def walk_forward_efficiency(fold_results: "list[FoldResult]",
                            min_is_sharpe: float = 0.10) -> float:
    """WFE = mean(OOS_Sharpe_fold / IS_Sharpe_fold) across folds where
    IS_Sharpe >= min_is_sharpe. Returns NaN if no valid folds.

    The source excluded folds with IS Sharpe <= 0; the floor is raised to
    min_is_sharpe here because a near-zero positive IS Sharpe makes the
    ratio explode (an IS Sharpe of +0.01 would multiply the OOS Sharpe
    400-fold) — those folds carry no information about IS→OOS degradation.

    Infinite OOS Sharpe (constant positive returns, zero variance) is capped
    at 3.0 so the ratio stays finite and comparable across folds.
    """
    _INF_CAP = 3.0
    ratios = []
    for fold in fold_results:
        if fold.is_sharpe < min_is_sharpe:
            continue
        oos_sharpe = arithmetic_sharpe(fold.oos_returns)
        if oos_sharpe == np.inf:
            oos_sharpe = _INF_CAP
        elif oos_sharpe == -np.inf:
            oos_sharpe = -_INF_CAP
        if np.isfinite(oos_sharpe):
            ratios.append(oos_sharpe / fold.is_sharpe)
    return float(np.mean(ratios)) if ratios else float("nan")


# ── Comprehensive report ───────────────────────────────────────────────────

def compute_all_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    weights: Optional[pd.DataFrame] = None,
) -> dict:
    """Compute comprehensive performance metrics."""
    metrics = {
        "total_return": (1 + returns).prod() - 1,
        "cagr": annualized_return(returns),
        "volatility": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "arithmetic_sharpe": arithmetic_sharpe(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns),
        "hit_rate": hit_rate(returns),
        "profit_factor": profit_factor(returns),
        "n_days": len(returns),
        "best_day": returns.max(),
        "worst_day": returns.min(),
        "avg_daily_return": returns.mean(),
        "skewness": returns.skew(),
        "kurtosis": returns.kurtosis(),
    }

    if benchmark_returns is not None and len(benchmark_returns) > 0:
        # Align
        common = returns.index.intersection(benchmark_returns.index)
        r = returns.reindex(common)
        b = benchmark_returns.reindex(common)
        excess = r - b

        metrics["benchmark_return"] = annualized_return(b)
        metrics["alpha"] = annualized_return(r) - annualized_return(b)
        metrics["excess_sharpe"] = sharpe_ratio(excess) if len(excess) > 1 else 0.0

        # Beta
        if b.var() > 0:
            metrics["beta"] = r.cov(b) / b.var()
        else:
            metrics["beta"] = 0.0

    if weights is not None:
        metrics["avg_turnover"] = average_turnover(weights)
        metrics["avg_positions"] = (weights > 0.001).sum(axis=1).mean()

    return metrics


def format_metrics(metrics: dict) -> str:
    """Pretty-print metrics."""
    lines = [
        "=" * 60,
        "  PERFORMANCE REPORT",
        "=" * 60,
        f"  Total Return:     {metrics.get('total_return', 0) * 100:+.2f}%",
        f"  CAGR:             {metrics.get('cagr', 0) * 100:+.2f}%",
        f"  Volatility:       {metrics.get('volatility', 0) * 100:.2f}%",
        f"  Sharpe Ratio:     {metrics.get('sharpe', 0):.3f}",
        f"  Sortino Ratio:    {metrics.get('sortino', 0):.3f}",
        f"  Max Drawdown:     {metrics.get('max_drawdown', 0) * 100:.2f}%",
        f"  Calmar Ratio:     {metrics.get('calmar', 0):.3f}",
        f"  Hit Rate:         {metrics.get('hit_rate', 0) * 100:.1f}%",
        f"  Profit Factor:    {metrics.get('profit_factor', 0):.2f}",
        "-" * 60,
        f"  Trading Days:     {metrics.get('n_days', 0)}",
        f"  Best Day:         {metrics.get('best_day', 0) * 100:+.2f}%",
        f"  Worst Day:        {metrics.get('worst_day', 0) * 100:+.2f}%",
    ]

    if "benchmark_return" in metrics:
        lines += [
            "-" * 60,
            f"  Benchmark CAGR:   {metrics['benchmark_return'] * 100:+.2f}%",
            f"  Alpha:            {metrics.get('alpha', 0) * 100:+.2f}%",
            f"  Beta:             {metrics.get('beta', 0):.3f}",
            f"  Excess Sharpe:    {metrics.get('excess_sharpe', 0):.3f}",
        ]

    if "avg_turnover" in metrics:
        lines += [
            "-" * 60,
            f"  Avg Turnover:     {metrics['avg_turnover'] * 100:.2f}% per day",
        ]

    lines.append("=" * 60)
    return "\n".join(lines)
