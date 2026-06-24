"""Tests for proportional (ratio) back-adjustment in roll.build_continuous.

Synthetic fixture design (so every number is hand-checkable):
- One shared 'true' underlying price path P(t) (a small random walk, fixed seed).
- Contract i quotes  close_i(t) = P(t) * (1 + step)**i   -> constant `step` contango.
- 4 contracts (RMK24, RMN24, RMU24, RMX24) -> 3 rolls.

Consequences we assert:
- ratio at each roll == (1 + step) exactly (same P in the window cancels).
- cumulative factor for the oldest contract == (1 + step)**3.
- after back-adjustment every segment collapses to P(t)*(1+step)**3 -> a single
  clean series proportional to the true underlying => returns preserved, no jumps.
- the most recent contract is left unadjusted (factor 1).
- the NAIVE spliced series still contains the ~step jump at each boundary.
"""
import numpy as np
import pandas as pd
import pytest

from ascentagri.contracts import parse_contract
from ascentagri.roll import build_continuous


STEP = 0.03
SYMBOLS = ["RMK24", "RMN24", "RMU24", "RMX24"]


def make_fixture(step=STEP, seed=0):
    clist = [parse_contract(s) for s in SYMBOLS]
    dates = pd.bdate_range("2024-01-02", "2024-11-15")
    rng = np.random.default_rng(seed)
    P = pd.Series(2000.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))), index=dates)
    contracts = {c: pd.DataFrame({"close": P * (1 + step) ** i}) for i, c in enumerate(clist)}
    return clist, P, contracts


def test_ratio_equals_contango_step_at_every_roll():
    clist, P, contracts = make_fixture()
    res = build_continuous(contracts, roll_offset_bd=5, window=4)
    assert len(res.roll_table) == 3
    assert np.allclose(res.roll_table["ratio"].values, 1 + STEP, atol=1e-9)


def test_cumulative_factor_for_oldest_is_step_cubed():
    clist, P, contracts = make_fixture()
    res = build_continuous(contracts, roll_offset_bd=5, window=4)
    # roll_table cumulative_factor is the factor applied to the expiring leg + everything older.
    oldest_factor = res.roll_table.iloc[0]["cumulative_factor"]
    assert oldest_factor == pytest.approx((1 + STEP) ** 3)


def test_most_recent_contract_is_unadjusted():
    clist, P, contracts = make_fixture()
    res = build_continuous(contracts, roll_offset_bd=5, window=4)
    last = clist[-1]
    last_dates = res.contract_map.index[res.contract_map["contract"] == last.symbol]
    # adjusted == raw close of the last contract on its own segment
    raw_last = contracts[last]["close"].reindex(last_dates)
    assert np.allclose(res.adjusted["close"].reindex(last_dates).values, raw_last.values)


def test_back_adjusted_series_collapses_to_clean_underlying():
    # returns-preservation: adjusted close == P(t) * (1+step)**3 on every date.
    clist, P, contracts = make_fixture()
    res = build_continuous(contracts, roll_offset_bd=5, window=4)
    expected = (P * (1 + STEP) ** 3).reindex(res.adjusted.index)
    assert np.allclose(res.adjusted["close"].values, expected.values, rtol=1e-9)
    # and the adjusted daily returns equal the true underlying returns
    adj_ret = res.adjusted["close"].pct_change().dropna()
    true_ret = P.reindex(res.adjusted.index).pct_change().dropna()
    assert np.allclose(adj_ret.values, true_ret.values, atol=1e-9)


def test_naive_spliced_still_contains_the_fake_jump():
    # guard: we must not "fix" the naive series. At EACH roll boundary the naive
    # return carries the curve jump while the adjusted return does not, so
    # (naive - adjusted) at the boundary should equal the contango step.
    clist, P, contracts = make_fixture()
    res = build_continuous(contracts, roll_offset_bd=5, window=4)
    raw = res.raw_spliced["close"]
    adj = res.adjusted["close"]
    for rd in res.roll_table["roll_date"]:
        prior = raw.index[raw.index < rd]
        after = raw.index[raw.index >= rd]
        d_prev, d_roll = prior[-1], after[0]
        naive_ret = raw[d_roll] / raw[d_prev] - 1
        adj_ret = adj[d_roll] / adj[d_prev] - 1
        # the artificial component removed by adjustment ~ the 3% contango step
        assert (naive_ret - adj_ret) == pytest.approx(STEP, abs=2e-3)
    # and the adjusted boundary returns are just the true underlying moves (small)
    for rd in res.roll_table["roll_date"]:
        prior = adj.index[adj.index < rd]
        after = adj.index[adj.index >= rd]
        d_prev, d_roll = prior[-1], after[0]
        true_ret = P[d_roll] / P[d_prev] - 1
        assert (adj[d_roll] / adj[d_prev] - 1) == pytest.approx(true_ret, abs=1e-9)


def test_intermediates_are_all_present_and_inspectable():
    clist, P, contracts = make_fixture()
    res = build_continuous(contracts, roll_offset_bd=5, window=4)
    for col in ("roll_date", "expiring", "next", "avg_exp", "avg_next", "ratio", "cumulative_factor"):
        assert col in res.roll_table.columns
    assert "close" in res.raw_spliced.columns
    assert "close" in res.adjusted.columns
    assert "contract" in res.contract_map.columns
    # spliced and adjusted cover the same dates
    assert res.raw_spliced.index.equals(res.adjusted.index)
