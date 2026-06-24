"""Read manually-downloaded Barchart per-contract CSVs into tidy OHLCV frames.

Barchart daily historical CSVs are typically:
    Time, Open, High, Low, Last, Change, Volume, Open Int
and end with a non-data footer line ("Downloaded from Barchart.com as of ...").
We map Time->index, Last->close, tolerate a 'Close' column and minor header
variations, and drop rows whose date doesn't parse (the footer).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import pandas as pd

from .contracts import Contract, parse_contract

PathLike = Union[str, Path]

# Map lower-cased Barchart headers -> our canonical names.
_COLMAP = {
    "time": "date",
    "date": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "last": "close",
    "close": "close",
    "settle": "close",
    "volume": "volume",
    "open int": "open_interest",
    "open interest": "open_interest",
}


def load_contract_csv(path: PathLike) -> pd.DataFrame:
    """Load one contract CSV -> DataFrame indexed by date with a 'close' column."""
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    cols = {c: _COLMAP[c] for c in raw.columns if c in _COLMAP}
    df = raw.rename(columns=cols)[list(dict.fromkeys(cols.values()))]

    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(
            f"{path}: could not find a date column and a close/last column. "
            f"Found headers: {list(raw.columns)}"
        )

    # Parse dates; the Barchart footer row becomes NaT and is dropped.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Numeric coercion for any present OHLCV columns.
    for c in ("open", "high", "low", "close", "volume", "open_interest"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def load_raw_dir(directory: PathLike) -> Dict[Contract, pd.DataFrame]:
    """Load every RM*.csv in `directory`, keyed by parsed Contract, delivery-sorted."""
    directory = Path(directory)
    out: Dict[Contract, pd.DataFrame] = {}
    for p in sorted(directory.glob("*.csv")):
        try:
            contract = parse_contract(p.name)
        except ValueError:
            continue  # not a contract CSV (e.g. README sidecar); skip
        out[contract] = load_contract_csv(p)
    return dict(sorted(out.items(), key=lambda kv: (kv[0].year, kv[0].month)))
