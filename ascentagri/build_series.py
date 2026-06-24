"""Build the roll-adjusted continuous robusta series from REAL data in data/raw/.

Run (after dropping Barchart CSVs into data/raw/):
    python -m ascentagri.build_series [--offset 5] [--window 4]

Writes:
    data/interim/robusta_naive_spliced.csv     (un-adjusted, for inspection)
    data/processed/robusta_continuous.csv       (ratio back-adjusted, the deliverable)
    data/processed/robusta_roll_table.csv       (ratios + cumulative factors)
    outputs/robusta_rolls.png                    (naive vs adjusted plot)

Prints the data-quality report: history span, #contracts, chain gaps, NaNs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .contracts import chain_gaps
from .loader import load_raw_dir
from .roll import build_continuous
from .validate import roll_return_report, offset_sensitivity, offset_sensitivity_summary, plot_rolls

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offset", type=int, default=5, help="roll_offset_bd (business days before FND)")
    ap.add_argument("--window", type=int, default=4, help="pre-roll averaging window (trading days)")
    args = ap.parse_args()

    contracts = load_raw_dir(RAW)
    if not contracts:
        print(f"No contract CSVs found in {RAW}.")
        print("See data/raw/README.md for how to download and name them (e.g. RMK24.csv).")
        return 1

    clist = list(contracts)
    gaps = chain_gaps(clist)

    print("=" * 78)
    print("DATA REPORT")
    print("=" * 78)
    print(f"Contracts loaded : {len(clist)}  ({clist[0].symbol} ... {clist[-1].symbol})")
    spans = []
    for c in clist:
        df = contracts[c]
        spans.append((c.symbol, df.index.min().date(), df.index.max().date(), len(df), int(df['close'].isna().sum())))
    first_date = min(s[1] for s in spans)
    last_date = max(s[2] for s in spans)
    years = (last_date - first_date).days / 365.25
    print(f"History span     : {first_date} -> {last_date}  (~{years:.1f} years)")
    print(f"Chain gaps       : {gaps if gaps else 'none (consecutive)'}")
    if gaps:
        print("  WARNING: gaps break the front-month splice at those rolls. Download the")
        print("           missing contracts, or the adjusted series will skip those rolls.")
    print("\nPer-contract:")
    print(f"  {'symbol':<8}{'start':<12}{'end':<12}{'rows':>6}{'close_NaN':>11}")
    for sym, s0, s1, n, nan in spans:
        print(f"  {sym:<8}{str(s0):<12}{str(s1):<12}{n:>6}{nan:>11}")

    if len(clist) < 2:
        print("\n" + "=" * 78)
        print("FORMAT OK — but only 1 contract loaded, so there are no rolls to compute.")
        print("The loader parsed it cleanly (see span/rows/NaNs above). Add a 2nd "
              "consecutive contract\nto start building the continuous series, then re-run.")
        print("=" * 78)
        return 0

    res = build_continuous(contracts, roll_offset_bd=args.offset, window=args.window)

    INTERIM.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    res.raw_spliced.to_csv(INTERIM / "robusta_naive_spliced.csv")
    res.adjusted.to_csv(PROCESSED / "robusta_continuous.csv")
    res.roll_table.to_csv(PROCESSED / "robusta_roll_table.csv", index=False)

    print("\n" + "=" * 78)
    print(f"ROLL TABLE (offset={args.offset} bd, window={args.window} d)")
    print("=" * 78)
    print(res.roll_table.to_string(index=False))

    print("\nReturns around each roll (naive vs adjusted):")
    print(roll_return_report(res).to_string(index=False))

    print("\nROLL-OFFSET SENSITIVITY (3 / 5 / 10):")
    sens = offset_sensitivity(contracts, offsets=(3, 5, 10), window=args.window)
    print(offset_sensitivity_summary(sens).to_string(index=False))
    print("Oldest-contract cumulative factor by offset:")
    for off in (3, 5, 10):
        r = build_continuous(contracts, roll_offset_bd=off, window=args.window)
        print(f"    offset {off:>2} bd: {float(np.prod(r.roll_table['ratio'].values)):.6f}")

    out = plot_rolls(res, OUTPUTS / "robusta_rolls.png")
    print(f"\nWrote:\n  {PROCESSED/'robusta_continuous.csv'}\n  {INTERIM/'robusta_naive_spliced.csv'}\n  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
