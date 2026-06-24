"""Proportional (ratio) back-adjustment of individual futures contracts into a
single continuous front-month series.

The most recent contract is left unadjusted; every earlier segment is scaled by
the cumulative product of roll ratios at/after its roll. This preserves
percentage returns across each splice exactly, which is what a returns-driven
regime model needs.

Nothing here is a black box: `build_continuous` returns every intermediate
(naive spliced series, per-roll ratios, cumulative factors, source map) so the
result can be inspected and validated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import pandas as pd

from .contracts import Contract, roll_date


@dataclass
class RollResult:
    raw_spliced: pd.DataFrame   # naive front-month concat (CONTAINS the fake jumps)
    roll_table: pd.DataFrame    # one row per roll: dates, contracts, avgs, ratio, factor
    adjusted: pd.DataFrame      # final ratio back-adjusted continuous series
    contract_map: pd.DataFrame  # which contract sources each date


def _pre_roll_avg(close: pd.Series, roll_dt: pd.Timestamp, window: int) -> float:
    """Mean close over the `window` most recent available days strictly before roll_dt."""
    prior = close[close.index < roll_dt].dropna()
    if prior.empty:
        raise ValueError(
            f"No data before roll date {roll_dt.date()} to compute the ratio window "
            f"(chain break or missing overlap)."
        )
    return float(prior.tail(window).mean())


def build_continuous(
    contracts: Dict[Contract, pd.DataFrame],
    roll_offset_bd: int = 5,
    window: int = 4,
    holidays: Optional[Sequence] = None,
) -> RollResult:
    """Splice individual contracts into a ratio back-adjusted continuous series.

    contracts: {Contract: DataFrame indexed by date with a 'close' column}.
    """
    if len(contracts) < 1:
        raise ValueError("Need at least one contract.")

    # Chronological order by delivery date.
    clist = sorted(contracts, key=lambda c: (c.year, c.month))
    frames = {c: contracts[c].sort_index() for c in clist}
    for c, df in frames.items():
        if "close" not in df.columns:
            raise ValueError(f"Contract {c.symbol} frame has no 'close' column.")
        if not isinstance(df.index, pd.DatetimeIndex):
            frames[c] = df.set_index(pd.to_datetime(df.index))
    n = len(clist)

    # Roll dates for the n-1 expiring contracts.
    rolls = [pd.Timestamp(roll_date(clist[i], roll_offset_bd, holidays)) for i in range(n - 1)]

    # --- ratios + cumulative factors -------------------------------------------------
    ratios = []
    roll_rows = []
    for i in range(n - 1):
        exp_c, next_c = clist[i], clist[i + 1]
        rd = rolls[i]
        avg_exp = _pre_roll_avg(frames[exp_c]["close"], rd, window)
        avg_next = _pre_roll_avg(frames[next_c]["close"], rd, window)
        ratio = avg_next / avg_exp
        ratios.append(ratio)
        roll_rows.append(
            {
                "roll_index": i,
                "roll_date": rd,
                "expiring": exp_c.symbol,
                "next": next_c.symbol,
                "avg_exp": avg_exp,
                "avg_next": avg_next,
                "ratio": ratio,
            }
        )

    # cumulative factor applied to contract i (and everything older) = product of ratios[i:].
    cum_factor = [1.0] * n
    running = 1.0
    for i in range(n - 1, -1, -1):
        cum_factor[i] = running
        if i > 0:  # ratios[i-1] is the roll that lifts contract i-1 up to contract i
            running *= ratios[i - 1]
    # attach the expiring-leg cumulative factor to each roll row (= cum_factor of expiring i)
    for row in roll_rows:
        row["cumulative_factor"] = cum_factor[row["roll_index"]]
    _ROLL_COLS = ["roll_index", "roll_date", "expiring", "next",
                  "avg_exp", "avg_next", "ratio", "cumulative_factor"]
    roll_table = pd.DataFrame(roll_rows, columns=_ROLL_COLS)  # keep schema even with 0 rolls

    # --- naive splice + adjusted ------------------------------------------------------
    raw_parts, adj_parts, map_parts = [], [], []
    for i, c in enumerate(clist):
        close = frames[c]["close"]
        start = None if i == 0 else rolls[i - 1]
        end = rolls[i] if i < n - 1 else None
        seg = close
        if start is not None:
            seg = seg[seg.index >= start]
        if end is not None:
            seg = seg[seg.index < end]
        seg = seg.dropna()
        raw_parts.append(seg)
        adj_parts.append(seg * cum_factor[i])
        map_parts.append(pd.Series(c.symbol, index=seg.index))

    raw_spliced = pd.concat(raw_parts).sort_index().to_frame("close")
    adjusted = pd.concat(adj_parts).sort_index().to_frame("close")
    contract_map = pd.concat(map_parts).sort_index().to_frame("contract")

    # Guard against overlapping segments producing duplicate dates.
    raw_spliced = raw_spliced[~raw_spliced.index.duplicated(keep="first")]
    adjusted = adjusted[~adjusted.index.duplicated(keep="first")]
    contract_map = contract_map[~contract_map.index.duplicated(keep="first")]

    return RollResult(
        raw_spliced=raw_spliced,
        roll_table=roll_table,
        adjusted=adjusted,
        contract_map=contract_map,
    )
