"""Fetch a free continuous commodity series from Yahoo (yfinance) as a tidy,
loader-compatible CSV.

DEV STAND-IN ONLY. The default symbol is KC=F = ICE **Arabica** coffee, used as a
development dataset because a free daily *robusta* continuous series does not exist
in any compliant programmatic form (Yahoo has no downloadable robusta; Stooq is
JS-gated; Barchart robusta is paywalled past ~1 download/day). Arabica lets us build
and tune the regime/debate pipeline now; the real hand-built robusta series swaps in
later with no code change (same tidy columns).

Usage:
    python -m ascentagri.vendor_fetch                 # KC=F arabica -> data/processed/
    python -m ascentagri.vendor_fetch --symbol KC=F --start 2018-01-01
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
_OHLCV = ["open", "high", "low", "close", "volume"]


def tidy_yahoo(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance download frame -> tidy [date, open, high, low, close, volume], sorted."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):          # single-ticker (Price, Ticker) cols
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().lower() for c in df.columns]
    cols = [c for c in _OHLCV if c in df.columns]
    if "close" not in cols:
        raise ValueError(f"No close column in Yahoo frame; got {list(df.columns)}")
    out = df[cols].copy()
    out.insert(0, "date", pd.to_datetime(df.index).date)
    return out.sort_values("date").reset_index(drop=True)


# --- network layer (not unit-tested; exercised live) ---------------------------------

def fetch_yahoo(symbol: str = "KC=F", start: str = "2018-01-01",
                end: Optional[str] = None) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        raise SystemExit(f"No data returned for {symbol} (delisted/blocked?).")
    return tidy_yahoo(raw)


_LABELS = {"KC=F": "ICE Arabica Coffee (DEV STAND-IN — NOT the robusta deliverable)"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="KC=F")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=None, help="output CSV path")
    args = ap.parse_args()

    tidy = fetch_yahoo(args.symbol, args.start, args.end)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    safe = args.symbol.replace("=", "").replace("^", "").replace("/", "")
    out = Path(args.out) if args.out else PROCESSED / f"coffee_{safe}_yahoo.csv"
    tidy.to_csv(out, index=False)

    label = _LABELS.get(args.symbol, f"Yahoo {args.symbol}")
    note = PROCESSED / "PROVENANCE.md"
    note.write_text(
        f"# data/processed provenance\n\n"
        f"- `{out.name}` — {label}. Source: Yahoo Finance `{args.symbol}` via yfinance, "
        f"auto_adjust=True, fetched {dt.date.today()}.\n"
        f"  This is a **development stand-in**, not the robusta case-study series. "
        f"The hand-built robusta roll-adjusted series (`robusta_continuous.csv`) replaces "
        f"it once enough contracts are downloaded.\n"
    )

    print("=" * 72)
    print(f"  {label}")
    print("=" * 72)
    print(f"symbol : {args.symbol}")
    print(f"rows   : {len(tidy)}   {tidy['date'].min()} -> {tidy['date'].max()}")
    print(f"close  : {float(tidy['close'].iloc[0]):.2f} -> {float(tidy['close'].iloc[-1]):.2f}")
    print(f"written: {out}")
    print(f"note   : {note}")
    print("\nLoader-compatible: regime work can read this now; swap in robusta_continuous.csv later.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
