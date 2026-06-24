"""Validation diagnostics for the roll-adjusted series.

The point of these is to let a human confirm, visually and numerically, that the
splice no longer contains artificial jumps -- and to measure how sensitive the
result is to the roll-offset choice before committing a default for real data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import pandas as pd

from .contracts import Contract
from .roll import RollResult, build_continuous

PathLike = Union[str, Path]


def _boundary_return(series: pd.Series, roll_dt: pd.Timestamp):
    """(return across the roll boundary, prev_date, roll_date)."""
    before = series.index[series.index < roll_dt]
    after = series.index[series.index >= roll_dt]
    if len(before) == 0 or len(after) == 0:
        return float("nan"), None, None
    d_prev, d_roll = before[-1], after[0]
    return series[d_roll] / series[d_prev] - 1.0, d_prev, d_roll


def roll_return_report(result: RollResult) -> pd.DataFrame:
    """Per roll: the boundary return before vs after adjustment, and the
    artificial jump removed (naive - adjusted)."""
    raw = result.raw_spliced["close"]
    adj = result.adjusted["close"]
    rows = []
    for _, r in result.roll_table.iterrows():
        rd = pd.Timestamp(r["roll_date"])
        naive_ret, d_prev, d_roll = _boundary_return(raw, rd)
        adj_ret, _, _ = _boundary_return(adj, rd)
        rows.append(
            {
                "roll_date": rd,
                "expiring": r["expiring"],
                "next": r["next"],
                "boundary_prev_date": d_prev,
                "boundary_roll_date": d_roll,
                "naive_return": naive_ret,
                "adjusted_return": adj_ret,
                "artificial_jump_removed": naive_ret - adj_ret,
            }
        )
    return pd.DataFrame(rows)


def offset_sensitivity(
    contracts: Dict[Contract, pd.DataFrame],
    offsets: Sequence[int] = (3, 5, 10),
    window: int = 4,
    holidays: Optional[Sequence] = None,
) -> pd.DataFrame:
    """Run the splice at several roll offsets and stack the roll tables, so you
    can see how ratios and cumulative factors move with the offset choice."""
    frames = []
    for off in offsets:
        res = build_continuous(contracts, roll_offset_bd=off, window=window, holidays=holidays)
        t = res.roll_table.copy()
        t.insert(0, "roll_offset_bd", off)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def offset_sensitivity_summary(sens: pd.DataFrame) -> pd.DataFrame:
    """Collapse the long sensitivity table to one row per roll showing the spread
    of the ratio across offsets (max - min) -- a quick robustness read."""
    g = sens.groupby(["expiring", "next"])
    out = g.agg(
        ratio_min=("ratio", "min"),
        ratio_max=("ratio", "max"),
        ratio_mean=("ratio", "mean"),
    ).reset_index()
    out["ratio_spread"] = out["ratio_max"] - out["ratio_min"]
    out["ratio_spread_bps"] = out["ratio_spread"] * 1e4
    return out


def plot_rolls(result: RollResult, out_path: PathLike) -> Path:
    """Plot naive spliced vs ratio-adjusted series with roll dates marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(result.raw_spliced.index, result.raw_spliced["close"],
            label="naive splice (has roll jumps)", lw=1.0, alpha=0.7)
    ax.plot(result.adjusted.index, result.adjusted["close"],
            label="ratio back-adjusted", lw=1.4)
    for rd in result.roll_table["roll_date"]:
        ax.axvline(pd.Timestamp(rd), color="grey", ls="--", lw=0.7, alpha=0.6)
    ax.set_title("Robusta continuous front-month: naive vs ratio-adjusted")
    ax.set_ylabel("price (USD/tonne)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
