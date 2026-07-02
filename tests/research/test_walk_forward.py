"""Walk-forward runner: end-to-end on synthetic data, causality, fold visibility."""
import logging

import numpy as np
import pandas as pd
import pytest

from ascentagri.config import get_config
from ascentagri.research.walk_forward_runner import (
    WalkForwardResult,
    print_wf_report,
    run_walk_forward,
)

logging.disable(logging.WARNING)


@pytest.fixture(scope="module")
def close():
    rng = np.random.default_rng(17)
    n = 800
    dates = pd.bdate_range("2020-01-01", periods=n)
    rets = np.concatenate([
        rng.normal(0.0010, 0.012, n // 2),     # calm bull
        rng.normal(-0.0006, 0.026, n - n // 2) # stressed
    ])
    return pd.Series(2800 * np.exp(np.cumsum(rets)), index=dates)


@pytest.fixture(scope="module")
def fast_cfg():
    return get_config(wf_train_days=300, wf_test_days=60, wf_step_days=60,
                      wf_min_train_days=252, wf_regime_k=2, wf_hmm_restarts=2)


@pytest.fixture(scope="module")
def result(close, fast_cfg):
    return run_walk_forward(close, config=fast_cfg, verbose=False)


def test_runs_end_to_end(result):
    assert isinstance(result, WalkForwardResult)
    assert result.report["n_folds"] > 3
    assert result.n_failed_folds == 0
    assert len(result.oos_returns) > 100


def test_oos_dates_never_in_train(close, fast_cfg, result):
    """Every OOS return date lies strictly after its fold's train window —
    the core no-look-ahead property."""
    from ascentagri.research.splits import walk_forward_splits
    splits = walk_forward_splits(close.index,
                                 train_days=fast_cfg.wf_train_days,
                                 test_days=fast_cfg.wf_test_days,
                                 step_days=fast_cfg.wf_step_days,
                                 purge_days=fast_cfg.wf_purge_days,
                                 min_train_days=fast_cfg.wf_min_train_days)
    for fold, split in zip(result.fold_results, splits):
        assert fold.oos_returns.index.min() >= split.test_start
        assert fold.oos_returns.index.max() <= split.test_end
        assert fold.oos_returns.index.min() > split.train_end


def test_positions_long_only(result):
    assert (result.positions >= 0).all()
    assert (result.positions <= 1.0 + 1e-9).all()


def test_report_fields(result):
    for key in ["cagr", "arithmetic_sharpe", "max_drawdown", "wfe",
                "n_folds", "n_failed_folds", "n_oos_days", "alpha", "beta"]:
        assert key in result.report


def test_benchmark_aligned(result):
    assert result.benchmark_returns.index.equals(result.oos_returns.index)


def test_failed_folds_are_visible_not_silent(fast_cfg):
    """A pathological series (constant price → degenerate features) must
    surface failures/flat folds in summaries, never invent returns."""
    dates = pd.bdate_range("2020-01-01", periods=500)
    flat = pd.Series(100.0, index=dates)
    res = run_walk_forward(flat, config=fast_cfg, verbose=False)
    # every fold accounted for: ok or failed with stage + error type
    assert len(res.fold_summaries) == res.report["n_folds"]
    for s in res.fold_summaries:
        assert s["status"] in {"ok", "failed"}
        if s["status"] == "failed":
            assert "stage" in s and "error_type" in s
    # flat price + long-only → strategy cannot fabricate PnL beyond costs
    assert res.oos_returns.abs().max() < 0.01


def test_too_short_series_raises():
    dates = pd.bdate_range("2023-01-01", periods=100)
    close = pd.Series(np.linspace(100, 110, 100), index=dates)
    with pytest.raises(ValueError, match="No walk-forward folds"):
        run_walk_forward(close, config=get_config(), verbose=False)


def test_print_report_smoke(result, capsys):
    print_wf_report(result)
    out = capsys.readouterr().out
    assert "WALK-FORWARD OOS PERFORMANCE REPORT" in out
    assert "Walk-Forward Eff." in out
