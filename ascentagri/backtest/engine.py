"""ascentagri/backtest/engine.py — ported near-verbatim from Ascent Capital.

Portfolio-native backtest with realistic costs and execution delay.

Key design decisions (unchanged from the source):
- Signal computed at date t close
- Execution at date t+1 open (1-day delay)
- Costs modeled per rebalance
- Weights held constant between rebalances
- Returns computed from actual price movements
- Day 1 has no valid delayed signal → stays in cash (no look-ahead)

Single-series note: callers with close-only data (the robusta continuous
series has no open column) pass open_prices=close_prices. Then the
overnight leg is the full close-to-close move and the intraday leg is 0 —
i.e. rebalance trades execute at the t+1 CLOSE. Conservative and honest;
disclosed in the README.
"""
from __future__ import annotations
import pandas as pd
from typing import Optional

from .costs import flat_cost_model


class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        spread_bps: float = 5.0,
        impact_bps: float = 5.0,
        rebalance_freq_days: int = 21,
        execution_delay: int = 1,
    ):
        self.initial_capital = initial_capital
        self.cost_bps = spread_bps + impact_bps
        self.rebalance_freq_days = rebalance_freq_days
        self.execution_delay = execution_delay

    def run(
        self,
        target_weights: pd.DataFrame,
        close_prices: pd.DataFrame,
        open_prices: pd.DataFrame,
        benchmark_prices: Optional[pd.Series] = None,
    ) -> "BacktestResult":
        # Align all data
        common_dates = target_weights.index.intersection(close_prices.index)
        common_dates = common_dates.intersection(open_prices.index)
        common_dates = common_dates.sort_values()

        symbols = target_weights.columns.intersection(close_prices.columns)

        tw     = target_weights.reindex(index=common_dates, columns=symbols).fillna(0)
        close  = close_prices.reindex(index=common_dates, columns=symbols)
        open_  = open_prices.reindex(index=common_dates, columns=symbols)

        daily_returns     = close.pct_change().fillna(0)
        prev_close        = close.shift(1)
        overnight_returns = (open_ / prev_close - 1).fillna(0)
        intraday_returns  = (close / open_ - 1).fillna(0)

        rebal_dates = self._get_rebalance_dates(common_dates)

        n_dates           = len(common_dates)
        portfolio_returns = pd.Series(0.0, index=common_dates)
        held_weights      = pd.DataFrame(0.0, index=common_dates, columns=symbols)
        turnover_series   = pd.Series(0.0, index=common_dates)
        cost_series       = pd.Series(0.0, index=common_dates)

        daily_rows = []

        prev_weights   = pd.Series(0.0, index=symbols)
        prev_end_value = float(self.initial_capital)

        for i in range(n_dates):
            dt          = common_dates[i]
            start_value = prev_end_value

            # Drift weights from previous day.
            # The source divided by drifted.sum() alone — correct only when
            # weights span the whole portfolio (multi-asset, fully invested).
            # With fractional exposure the cash bucket must drift too, or a
            # 0.10 weight snaps to 1.0 the next day (cash silently dropped).
            if i > 0:
                prev_dt = common_dates[i - 1]
                ret     = daily_returns.loc[prev_dt]
                drifted = prev_weights * (1 + ret)
                cash    = max(0.0, 1.0 - float(prev_weights.sum()))
                total   = drifted.sum() + cash
                current_weights = drifted / total if total > 0 else prev_weights.copy()
            else:
                current_weights = prev_weights.copy()

            signal_date     = pd.NaT
            is_rebalance    = dt in rebal_dates
            turn            = 0.0
            cost_rate       = 0.0
            valid_rebalance = False  # only True when a real delayed signal is applied

            if is_rebalance:
                delay_idx = i - self.execution_delay

                if delay_idx < 0:
                    # No valid delayed signal exists yet — stay in cash.
                    # (Using tw.loc[dt] here would let day 1 earn intraday PnL
                    # on a signal that hadn't been generated yet: look-ahead.)
                    signal_date     = pd.NaT
                    valid_rebalance = False
                else:
                    signal_date = common_dates[delay_idx]
                    new_target  = tw.loc[signal_date]

                    turn      = float((new_target - current_weights).abs().sum() / 2)
                    cost_rate = float(flat_cost_model(turn, self.cost_bps))
                    turnover_series.loc[dt] = turn
                    cost_series.loc[dt]     = cost_rate

                    prev_weights_before_rebal = current_weights.copy()
                    current_weights           = new_target.copy()
                    valid_rebalance           = True

            # Return calculation
            if valid_rebalance:
                # Old weights earn overnight, new weights earn intraday
                overnight_gross = float((prev_weights_before_rebal * overnight_returns.loc[dt]).sum())
                intraday_gross  = float((current_weights * intraday_returns.loc[dt]).sum())
                gross_return    = overnight_gross + intraday_gross
            else:
                # No rebalance (or no valid signal): full close-to-close on held weights
                gross_return = float((current_weights * daily_returns.loc[dt]).sum())

            net_return = gross_return - cost_rate

            portfolio_returns.loc[dt] = net_return
            held_weights.loc[dt]      = current_weights
            prev_weights              = current_weights.copy()

            gross_pnl      = start_value * gross_return
            cost_dollars   = start_value * cost_rate
            net_pnl        = start_value * net_return
            end_value      = start_value + net_pnl
            prev_end_value = end_value
            positions      = int((current_weights.abs() > 1e-12).sum())

            daily_rows.append({
                "date":         dt,
                "start_value":  start_value,
                "gross_return": gross_return,
                "gross_pnl":    gross_pnl,
                "turnover":     turn,
                "cost_rate":    cost_rate,
                "cost_dollars": cost_dollars,
                "net_return":   net_return,
                "net_pnl":      net_pnl,
                "end_value":    end_value,
                "positions":    positions,
                "is_rebalance": is_rebalance,
                "signal_date":  signal_date,
            })

        benchmark_returns = None
        if benchmark_prices is not None:
            bm = benchmark_prices.reindex(common_dates)
            benchmark_returns = bm.pct_change().fillna(0)

        equity       = self.initial_capital * (1 + portfolio_returns).cumprod()
        daily_ledger = pd.DataFrame(daily_rows).set_index("date")

        return BacktestResult(
            portfolio_returns=portfolio_returns,
            equity_curve=equity,
            held_weights=held_weights,
            turnover=turnover_series,
            costs=cost_series,
            benchmark_returns=benchmark_returns,
            initial_capital=self.initial_capital,
            daily_ledger=daily_ledger,
        )

    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> set:
        rebal = set()
        count = 0
        for dt in dates:
            if count % self.rebalance_freq_days == 0:
                rebal.add(dt)
            count += 1
        return rebal


class BacktestResult:
    def __init__(
        self,
        portfolio_returns: pd.Series,
        equity_curve: pd.Series,
        held_weights: pd.DataFrame,
        turnover: pd.Series,
        costs: pd.Series,
        benchmark_returns: Optional[pd.Series],
        initial_capital: float,
        daily_ledger: Optional[pd.DataFrame] = None,
    ):
        self.portfolio_returns = portfolio_returns
        self.equity_curve      = equity_curve
        self.held_weights      = held_weights
        self.turnover          = turnover
        self.costs             = costs
        self.benchmark_returns = benchmark_returns
        self.initial_capital   = initial_capital
        self.daily_ledger      = daily_ledger

    @property
    def total_return(self) -> float:
        return (1 + self.portfolio_returns).prod() - 1

    @property
    def total_cost(self) -> float:
        return self.costs.sum()

    @property
    def avg_daily_turnover(self) -> float:
        return self.turnover.mean()

    def gross_returns(self) -> pd.Series:
        return self.portfolio_returns + self.costs

    def drawdown_series(self) -> pd.Series:
        cum  = (1 + self.portfolio_returns).cumprod()
        peak = cum.cummax()
        return (cum - peak) / peak

    def summary(self) -> dict:
        from ..research.evaluation import compute_all_metrics
        return compute_all_metrics(
            self.portfolio_returns,
            benchmark_returns=self.benchmark_returns,
            weights=self.held_weights,
        )
