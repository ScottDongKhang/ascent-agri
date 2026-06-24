"""Worked example + offset-sensitivity check on SYNTHETIC data.

Run:  python -m ascentagri.worked_example

No real data required. Proves the roll-adjustment pipeline end to end before any
Barchart CSV is downloaded:

  Section A - clean constant 3% contango. Every number is exactly checkable: the
              Mar->May ratio is 1.0300 and the cumulative factor is 1.03**4.
  Section B - realistic *time-varying* contango. The roll offset is swept over
              {3, 5, 10} business days so you can see whether the choice is robust
              or sensitive on a curve whose shape actually moves.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import parse_contract
from .roll import build_continuous
from .validate import (
    roll_return_report,
    offset_sensitivity,
    offset_sensitivity_summary,
    plot_rolls,
)

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda x: f"{x:,.6f}")

# H24(Mar) K24(May) N24(Jul) U24(Sep) X24(Nov) -> 5 contracts, 4 rolls.
# First roll is H24->K24 = the March->May roll referenced in the brief.
SYMBOLS = ["RMH24", "RMK24", "RMN24", "RMU24", "RMX24"]
DATES = pd.bdate_range("2023-11-01", "2024-11-15")


def _spot(seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(2000.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(DATES)))), index=DATES)


def constant_contango_fixture(step: float = 0.03):
    """close_i(t) = spot(t) * (1+step)**i  -> exact, offset-invariant ratios."""
    spot = _spot()
    clist = [parse_contract(s) for s in SYMBOLS]
    return clist, spot, {c: pd.DataFrame({"close": spot * (1 + step) ** i}) for i, c in enumerate(clist)}


def time_varying_contango_fixture():
    """Spread between adjacent contracts drifts over time (+ mild cycle), so the
    4-day window location -- and thus the ratio -- depends on the roll offset.
    close_i(t) = spot(t) * (1 + spread(t) * i)."""
    spot = _spot()
    t = np.linspace(0, 1, len(DATES))
    spread = 0.02 + 0.04 * t + 0.01 * np.sin(2 * np.pi * t)  # ~2% -> ~6%, drifting
    spread = pd.Series(spread, index=DATES)
    clist = [parse_contract(s) for s in SYMBOLS]
    return clist, spot, {c: pd.DataFrame({"close": spot * (1 + spread * i)}) for i, c in enumerate(clist)}


def _hr(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def section_a() -> None:
    _hr("SECTION A - clean constant 3% contango (every number exactly checkable)")
    clist, spot, contracts = constant_contango_fixture(step=0.03)
    res = build_continuous(contracts, roll_offset_bd=5, window=4)

    print("\nRoll table (avg over the 4 trading days before each roll date):")
    print(res.roll_table.to_string(index=False))

    first = res.roll_table.iloc[0]
    print(
        f"\nWorked Mar->May roll ({first['expiring']} -> {first['next']}) at {pd.Timestamp(first['roll_date']).date()}:"
        f"\n    avg_exp (4d) = {first['avg_exp']:.4f}"
        f"\n    avg_next(4d) = {first['avg_next']:.4f}"
        f"\n    ratio        = avg_next / avg_exp = {first['ratio']:.6f}   (expected 1.030000)"
    )
    ratios = res.roll_table["ratio"].tolist()
    print("\nBackward propagation (cumulative product of ratios, newest contract = 1.0):")
    print(f"    ratios per roll (old->new): {[round(r, 6) for r in ratios]}")
    print(f"    oldest contract cumulative factor = prod(all ratios) = {np.prod(ratios):.6f}"
          f"   (expected 1.03**4 = {1.03**4:.6f})")
    print(f"    => every historical price on {clist[0].symbol} is scaled UP by {np.prod(ratios):.4f}x")

    print("\nReturns around each roll (naive vs adjusted):")
    rep = roll_return_report(res)
    print(rep[["roll_date", "expiring", "next", "naive_return",
               "adjusted_return", "artificial_jump_removed"]].to_string(index=False))
    print(f"\n    artificial jump removed per roll ~ {rep['artificial_jump_removed'].mean():.4f} "
          f"(= the 3% contango that was NOT a real market move)")


def section_b() -> Path:
    _hr("SECTION B - realistic time-varying contango: roll-offset sensitivity (3 / 5 / 10)")
    clist, spot, contracts = time_varying_contango_fixture()

    sens = offset_sensitivity(contracts, offsets=(3, 5, 10), window=4)
    print("\nPer-roll ratio + cumulative factor at each offset:")
    print(sens[["roll_offset_bd", "expiring", "next", "roll_date",
                "ratio", "cumulative_factor"]].to_string(index=False))

    print("\nRobustness summary (how much each roll's ratio moves across offsets 3-10):")
    summ = offset_sensitivity_summary(sens)
    print(summ.to_string(index=False))

    # cumulative factor on the OLDEST contract at each offset (the most-amplified number)
    oldest_sym = clist[0].symbol
    print(f"\nOldest-contract ({oldest_sym}) cumulative factor by offset "
          f"(this multiplies the whole earliest segment):")
    for off in (3, 5, 10):
        res = build_continuous(contracts, roll_offset_bd=off, window=4)
        # oldest cumulative factor = product of all ratios
        fac = float(np.prod(res.roll_table["ratio"].values))
        print(f"    offset {off:>2} bd: {fac:.6f}")

    res5 = build_continuous(contracts, roll_offset_bd=5, window=4)
    out = plot_rolls(res5, Path(__file__).resolve().parents[1] / "outputs" / "worked_example_rolls.png")
    print(f"\nPlot (naive vs adjusted, offset=5) written to: {out}")
    return out


def main() -> None:
    section_a()
    section_b()
    _hr("DONE - synthetic pipeline verified. Drop real Barchart CSVs in data/raw/ to run for real.")


if __name__ == "__main__":
    main()
