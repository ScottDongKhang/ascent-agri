"""Backtest engine: costs, execution delay, no look-ahead."""
import numpy as np
import pandas as pd
import pytest

from ascentagri.backtest.engine import BacktestEngine


def _frames(prices, name="C"):
    df = prices.to_frame(name)
    return df, df  # close, open (close-only convention)


def test_flat_prices_zero_gross_costs_only():
    dates = pd.bdate_range("2024-01-01", periods=30)
    close = pd.Series(100.0, index=dates)
    weights = pd.DataFrame({"C": 1.0}, index=dates)
    eng = BacktestEngine(rebalance_freq_days=10, spread_bps=5, impact_bps=5)
    res = eng.run(weights, *_frames(close))
    # gross return must be exactly zero; net = -costs at rebalances
    assert abs(res.gross_returns().sum()) < 1e-12
    assert res.total_cost > 0
    assert (res.portfolio_returns <= 0).all()


def test_day_one_stays_in_cash_no_lookahead():
    """Day 1 is a rebalance date but has no valid delayed signal — the engine
    must stay in cash rather than trade on a signal that doesn't exist yet."""
    dates = pd.bdate_range("2024-01-01", periods=10)
    close = pd.Series(np.linspace(100, 200, 10), index=dates)  # violent rally
    weights = pd.DataFrame({"C": 1.0}, index=dates)
    eng = BacktestEngine(rebalance_freq_days=100, execution_delay=1)
    res = eng.run(weights, *_frames(close))
    assert res.portfolio_returns.iloc[0] == 0.0
    assert (res.held_weights.iloc[0] == 0).all()


def test_execution_delay_uses_prior_signal():
    """Weights change at t; the rebalance at t must apply the t-1 signal."""
    dates = pd.bdate_range("2024-01-01", periods=6)
    close = pd.Series([100, 100, 100, 110, 110, 110.0], index=dates)
    # signal: flat until day 3 (index 3), then fully long
    w = pd.DataFrame({"C": [0, 0, 0, 1.0, 1.0, 1.0]}, index=dates)
    eng = BacktestEngine(rebalance_freq_days=1, execution_delay=1,
                         spread_bps=0, impact_bps=0)
    res = eng.run(w, *_frames(close))
    # day 3: rebalance applies day-2 signal (0) → misses the +10% jump
    assert res.portfolio_returns.iloc[3] == pytest.approx(0.0, abs=1e-12)
    # day 4: rebalance applies day-3 signal (1.0) → holds through flat close
    assert (res.held_weights.iloc[4] == 1.0).all()


def test_buy_and_hold_tracks_underlying():
    rng = np.random.default_rng(8)
    dates = pd.bdate_range("2024-01-01", periods=120)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, 120))),
                      index=dates)
    weights = pd.DataFrame({"C": 1.0}, index=dates)
    eng = BacktestEngine(rebalance_freq_days=1, execution_delay=1,
                         spread_bps=0, impact_bps=0)
    res = eng.run(weights, *_frames(close))
    # day 0: cash (no delayed signal). day 1: enters at close (open==close
    # convention → zero intraday PnL). day 2 onward tracks the underlying.
    underlying = close.pct_change()
    got = res.portfolio_returns.iloc[2:]
    np.testing.assert_allclose(got.values, underlying.iloc[2:].values, atol=1e-10)


def test_costs_proportional_to_turnover():
    dates = pd.bdate_range("2024-01-01", periods=4)
    close = pd.Series(100.0, index=dates)
    w = pd.DataFrame({"C": [0.0, 1.0, 1.0, 1.0]}, index=dates)
    eng = BacktestEngine(rebalance_freq_days=1, execution_delay=1,
                         spread_bps=10, impact_bps=10)
    res = eng.run(w, *_frames(close))
    # day 2 rebalance: 0→1 exposure = 0.5 two-sided turnover × 20bps
    expected_cost = 0.5 * 20 / 10_000
    assert res.costs.iloc[2] == pytest.approx(expected_cost)


def test_fractional_exposure_does_not_snap_to_full():
    """Regression: with a single asset at 50% exposure, the cash bucket must
    drift too. The ported renormalization dropped cash, snapping any nonzero
    weight to 1.0 the day after — full exposure regardless of signal."""
    dates = pd.bdate_range("2024-01-01", periods=40)
    rng = np.random.default_rng(3)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 40))), index=dates)
    w = pd.DataFrame({"C": 0.5}, index=dates)
    eng = BacktestEngine(rebalance_freq_days=10, execution_delay=1,
                         spread_bps=0, impact_bps=0)
    res = eng.run(w, *_frames(close))
    held = res.held_weights["C"].iloc[11:]      # after first valid rebalance
    assert held.max() < 0.60                     # drift stays near 0.5
    assert held.min() > 0.40
    # daily portfolio return ≈ half the underlying's return
    got = res.portfolio_returns.iloc[12:]
    expected = 0.5 * close.pct_change().iloc[12:]
    np.testing.assert_allclose(got.values, expected.values, rtol=0.15)


def test_exact_drift_math_single_asset():
    """Weight 0.5, price doubles overnight: position worth 1.0, cash 0.5 →
    portfolio 1.5 → drifted weight must be 2/3, not renormalized to 1.0."""
    dates = pd.bdate_range("2024-01-01", periods=4)
    close = pd.Series([100.0, 100.0, 200.0, 200.0], index=dates)
    w = pd.DataFrame({"C": [0.5, 0.5, 0.5, 0.5]}, index=dates)
    eng = BacktestEngine(rebalance_freq_days=100, execution_delay=1,
                         spread_bps=0, impact_bps=0)
    # only rebalance day is day 0 (no valid signal → cash); force entry via day-1 rebal
    eng2 = BacktestEngine(rebalance_freq_days=1, execution_delay=1,
                          spread_bps=0, impact_bps=0)
    res = eng2.run(w, *_frames(close))
    # day 1: rebalance to 0.5. day 2: price doubles → drift to 2/3 before
    # that day's rebalance resets it to 0.5 using the day-1 signal.
    # capture drift by looking at the day-2 return: 0.5 × 100% = 50%
    assert res.portfolio_returns.iloc[2] == pytest.approx(0.5)
    # and day-3 held weight is back at the 0.5 target
    assert res.held_weights["C"].iloc[3] == pytest.approx(0.5)


def test_drawdown_series_nonpositive():
    rng = np.random.default_rng(12)
    dates = pd.bdate_range("2024-01-01", periods=200)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 200))), index=dates)
    w = pd.DataFrame({"C": 0.8}, index=dates)
    res = BacktestEngine(rebalance_freq_days=5).run(w, *_frames(close))
    dd = res.drawdown_series()
    assert (dd <= 1e-12).all()
