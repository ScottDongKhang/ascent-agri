"""Fetch individual ICE Robusta contracts from Databento and write them into
data/raw/ as RM<code><yy>.csv, so the existing loader/roll/build_series pipeline
runs unchanged.

Compliant API access (not scraping). Needs a Databento API key:
    export DATABENTO_API_KEY=db-xxxxxxxx
New accounts get $125 in promo credits; daily OHLCV for ~20 contracts costs cents
(check first with `--list`, which only hits free metadata + a tiny sample).

Usage:
    python -m ascentagri.databento_fetch --list      # coverage + cost + real symbols, no big pull
    python -m ascentagri.databento_fetch             # fetch the target chain -> data/raw/
    python -m ascentagri.databento_fetch --schema ohlcv-eod   # settlement-based daily

The pure transform functions (parse/split/sanity) are unit-tested without network.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .contracts import Contract, MONTH_CODE_TO_MONTH

DATASET = "IFEU.IMPACT"           # ICE Futures Europe (incl. Robusta Coffee)
PARENT_SYMBOL = "RC.FUT"          # all Robusta Coffee futures contracts
ROOT_OUT = "RM"                   # filenames keep the RM root used across this project
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# Databento ICE contract symbols look like 'RCN23' / 'RC N3' / 'RCX2024'.
_SYM_RE = re.compile(r"^RC\s?([FGHJKMNQUVXZ])\s?(\d{1,4})$")


def _yy_to_year(yy: int) -> int:
    if yy >= 1000:           # already 4-digit
        return yy
    if yy >= 100:            # 3-digit, unexpected -> treat as junk
        return -1
    if yy < 10:              # single digit (e.g. 3 -> 2023), valid for our 2020s window
        return 2020 + yy
    return 2000 + yy if yy < 70 else 1900 + yy


def parse_databento_symbol(symbol: str) -> Optional[Contract]:
    """Parse a Databento RC contract symbol -> Contract, or None if it is not a
    plain robusta delivery-month contract (skips spreads, options, other months)."""
    m = _SYM_RE.match(symbol.strip().upper())
    if not m:
        return None
    code, yy = m.group(1), int(m.group(2))
    if code not in MONTH_CODE_TO_MONTH:        # not in robusta cycle (F H K N U X)
        return None
    year = _yy_to_year(yy)
    if not (2018 <= year <= 2035):             # plausibility guard
        return None
    return Contract(root="RC", month_code=code, year=year, month=MONTH_CODE_TO_MONTH[code])


def to_output_filename(contract: Contract) -> str:
    """Filename our loader expects: RM<code><yy>.csv (root forced to RM)."""
    return f"{ROOT_OUT}{contract.month_code}{contract.year % 100:02d}.csv"


def split_and_format(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split a Databento `to_df()` OHLCV frame (DatetimeIndex + 'symbol' column)
    into {filename: tidy per-contract frame [date, open, high, low, close, volume]}.
    Rows whose symbol is not a plain robusta contract are dropped."""
    keep = ["open", "high", "low", "close", "volume"]
    out: Dict[str, pd.DataFrame] = {}
    for sym, g in df.groupby("symbol"):
        contract = parse_databento_symbol(str(sym))
        if contract is None:
            continue
        fname = to_output_filename(contract)
        tidy = g[[c for c in keep if c in g.columns]].copy()
        tidy.insert(0, "date", pd.to_datetime(g.index).date)
        tidy = tidy.sort_values("date").reset_index(drop=True)
        # if the same contract appears under >1 raw symbol spelling, concat
        out[fname] = pd.concat([out[fname], tidy]).drop_duplicates("date") if fname in out else tidy
    return out


def price_sanity_warnings(frames: Dict[str, pd.DataFrame],
                          lo: float = 200.0, hi: float = 50000.0) -> List[str]:
    """Robusta trades ~1500-5500 USD/tonne. Flag any contract whose median close is
    wildly outside [lo, hi] -- usually a price-scaling problem, not a real value."""
    warns = []
    for fname, f in frames.items():
        if "close" not in f or f["close"].dropna().empty:
            warns.append(f"{fname}: no close prices")
            continue
        med = float(f["close"].median())
        if not (lo <= med <= hi):
            warns.append(f"{fname}: median close {med:g} outside plausible [{lo:g}, {hi:g}] "
                         f"-- check price scaling/units")
    return warns


# --------------------------------------------------------------------------------------
# Network layer (not unit-tested; exercised live with a real key)
# --------------------------------------------------------------------------------------

def _client():
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise SystemExit("Set DATABENTO_API_KEY (export DATABENTO_API_KEY=db-...). "
                         "New accounts get $125 free credits at databento.com.")
    import databento as db
    return db.Historical(key=key)


def _default_window(start_delivery: str, end: Optional[str]):
    # request from ~12 months before the oldest target delivery (captures each
    # contract's liquid front/next-leg window) through `end` (default today).
    sd = dt.date.fromisoformat(start_delivery + "-01")
    start = (sd.replace(day=1) - pd.DateOffset(months=12)).date().isoformat()
    end = end or dt.date.today().isoformat()
    return start, end


def cmd_list(client, schema: str, start: str, end: str) -> None:
    rng = client.metadata.get_dataset_range(DATASET)
    print(f"Dataset {DATASET} coverage: {rng}")
    cost = client.metadata.get_cost(
        dataset=DATASET, start=start, end=end, symbols=[PARENT_SYMBOL],
        schema=schema, stype_in="parent",
    )
    print(f"Estimated cost for {schema} {PARENT_SYMBOL} {start}->{end}: ${cost:.4f}  (you have $125 credit)")
    # tiny recent sample just to reveal the REAL raw-symbol format
    sample = client.timeseries.get_range(
        dataset=DATASET, schema=schema, symbols=[PARENT_SYMBOL], stype_in="parent",
        start=(dt.date.fromisoformat(end) - dt.timedelta(days=12)).isoformat(), end=end,
    ).to_df()
    syms = sorted(map(str, sample["symbol"].unique())) if "symbol" in sample else []
    print(f"Sample of live raw symbols (last ~12d): {syms[:40]}")
    parsed = {s: (to_output_filename(c) if (c := parse_databento_symbol(s)) else "—SKIP—") for s in syms}
    for s, f in parsed.items():
        print(f"    {s:<12} -> {f}")


def cmd_fetch(client, schema: str, start: str, end: str,
              start_delivery: str, end_delivery: str, write: bool) -> int:
    print(f"Fetching {schema} {PARENT_SYMBOL} {start} -> {end} from {DATASET} ...")
    df = client.timeseries.get_range(
        dataset=DATASET, schema=schema, symbols=[PARENT_SYMBOL], stype_in="parent",
        start=start, end=end,
    ).to_df()
    if df.empty:
        print("No data returned. Check coverage with --list.")
        return 1

    frames = split_and_format(df)

    # keep only contracts whose delivery is within the requested chain window
    lo = dt.date.fromisoformat(start_delivery + "-01")
    hi = dt.date.fromisoformat(end_delivery + "-01")
    kept = {}
    for fname, f in frames.items():
        c = parse_databento_symbol("RC" + fname[len(ROOT_OUT):-4])  # RM->RC, strip .csv
        deliv = dt.date(c.year, c.month, 1)
        if lo <= deliv <= hi:
            kept[fname] = f

    for w in price_sanity_warnings(kept):
        print(f"  WARNING: {w}")

    print(f"\nParsed {len(frames)} contracts; {len(kept)} within target window "
          f"{start_delivery}..{end_delivery}:")
    for fname in sorted(kept):
        f = kept[fname]
        print(f"  {fname:<10} {f['date'].min()} -> {f['date'].max()}  ({len(f)} rows)")

    if write:
        RAW.mkdir(parents=True, exist_ok=True)
        for fname, f in kept.items():
            f.to_csv(RAW / fname, index=False)
        print(f"\nWrote {len(kept)} files to {RAW}. Now run:  python -m ascentagri.build_series")
    else:
        print("\n(dry run -- pass without --dry-run to write into data/raw/)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="coverage + cost + real symbols only")
    ap.add_argument("--schema", default="ohlcv-1d", help="ohlcv-1d (default) or ohlcv-eod (settlement)")
    ap.add_argument("--start-delivery", default="2023-07", help="oldest target delivery YYYY-MM")
    ap.add_argument("--end-delivery", default="2026-09", help="newest target delivery YYYY-MM")
    ap.add_argument("--end", default=None, help="fetch end date YYYY-MM-DD (default today)")
    ap.add_argument("--dry-run", action="store_true", help="fetch + report but do not write files")
    args = ap.parse_args()

    client = _client()
    start, end = _default_window(args.start_delivery, args.end)
    if args.list:
        cmd_list(client, args.schema, start, end)
        return 0
    return cmd_fetch(client, args.schema, start, end,
                     args.start_delivery, args.end_delivery, write=not args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
