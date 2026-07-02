"""Evaluation metrics on known series + WFE behavior."""
import numpy as np
import pandas as pd
import pytest

from ascentagri.research.evaluation import (
    FoldResult,
    annualized_return,
    arithmetic_sharpe,
    compute_all_metrics,
    max_drawdown,
    sharpe_ratio,
    walk_forward_efficiency,
)


def _series(vals, start="2024-01-01"):
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)))


def test_annualized_return_known_value():
    # +1% per day for 252 days
    r = _series([0.01] * 252)
    assert annualized_return(r) == pytest.approx(1.01 ** 252 - 1, rel=1e-9)


def test_max_drawdown_known_value():
    r = _series([0.10, -0.50, 0.10])
    assert max_drawdown(r) == pytest.approx(-0.50)


def test_sharpe_zero_vol_is_zero():
    r = _series([0.0] * 100)
    assert sharpe_ratio(r) == 0.0


def test_arithmetic_sharpe_sign():
    rng = np.random.default_rng(0)
    up = _series(rng.normal(0.001, 0.01, 500))
    down = _series(rng.normal(-0.001, 0.01, 500))
    assert arithmetic_sharpe(up) > 0 > arithmetic_sharpe(down)


def test_wfe_perfect_no_degradation():
    oos = _series(np.random.default_rng(1).normal(0.001, 0.01, 100))
    s = arithmetic_sharpe(oos)
    folds = [FoldResult(0, is_sharpe=s, oos_returns=oos)]
    assert walk_forward_efficiency(folds) == pytest.approx(1.0)


def test_wfe_excludes_tiny_is_sharpe():
    """A fold with IS Sharpe +0.01 must not blow the ratio up 400x."""
    bad_oos = _series(np.random.default_rng(2).normal(-0.004, 0.01, 100))
    folds = [FoldResult(0, is_sharpe=0.01, oos_returns=bad_oos)]
    assert np.isnan(walk_forward_efficiency(folds))


def test_wfe_degradation_below_one():
    oos = _series(np.random.default_rng(3).normal(0.0002, 0.01, 200))
    folds = [FoldResult(0, is_sharpe=2.0, oos_returns=oos)]
    wfe = walk_forward_efficiency(folds)
    assert np.isfinite(wfe) and wfe < 1.0


def test_compute_all_metrics_with_benchmark():
    rng = np.random.default_rng(4)
    r = _series(rng.normal(0.0008, 0.012, 400))
    b = _series(rng.normal(0.0004, 0.012, 400))
    m = compute_all_metrics(r, benchmark_returns=b)
    for key in ["cagr", "sharpe", "arithmetic_sharpe", "max_drawdown",
                "alpha", "beta", "hit_rate", "n_days"]:
        assert key in m
    assert m["n_days"] == 400
