"""ICE Robusta Coffee contract codes and roll-date math.

Delivery months (every 2 months): Jan/Mar/May/Jul/Sep/Nov.
First Notice Day (FND) = 4th business day before the 1st business day of the
delivery month. We roll `roll_offset_bd` business days before FND.

Business days are Mon-Fri. Exchange holidays are not modeled in v1; pass a
`holidays` list (np.datetime64-compatible) to tighten this later.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# ICE robusta delivery-month cycle.
MONTH_CODE_TO_MONTH = {"F": 1, "H": 3, "K": 5, "N": 7, "U": 9, "X": 11}
MONTH_TO_CODE = {m: c for c, m in MONTH_CODE_TO_MONTH.items()}

_SYMBOL_RE = re.compile(r"^(?P<root>[A-Z]{1,3})(?P<mcode>[A-Z])(?P<yy>\d{2})$")


@dataclass(frozen=True)
class Contract:
    root: str
    month_code: str
    year: int  # full year, e.g. 2024
    month: int  # delivery month, 1-12

    @property
    def symbol(self) -> str:
        return f"{self.root}{self.month_code}{self.year % 100:02d}"

    @property
    def delivery_first_of_month(self) -> dt.date:
        return dt.date(self.year, self.month, 1)


def _yy_to_year(yy: int) -> int:
    # Pivot for a 2020s-era case study: 00-69 -> 2000s, 70-99 -> 1900s.
    return 2000 + yy if yy < 70 else 1900 + yy


def parse_contract(symbol_or_filename: str) -> Contract:
    """Parse 'RMK24' or 'RMK24.csv' into a Contract."""
    stem = symbol_or_filename.strip()
    if stem.lower().endswith(".csv"):
        stem = stem[:-4]
    stem = stem.upper()
    m = _SYMBOL_RE.match(stem)
    if not m:
        raise ValueError(f"Unrecognized contract symbol: {symbol_or_filename!r}")
    mcode = m.group("mcode")
    if mcode not in MONTH_CODE_TO_MONTH:
        raise ValueError(
            f"{mcode!r} is not a robusta delivery month "
            f"(valid: {''.join(MONTH_CODE_TO_MONTH)})"
        )
    return Contract(
        root=m.group("root"),
        month_code=mcode,
        year=_yy_to_year(int(m.group("yy"))),
        month=MONTH_CODE_TO_MONTH[mcode],
    )


def _busday_holidays(holidays: Optional[Sequence]) -> np.busdaycalendar:
    hols = np.array(holidays, dtype="datetime64[D]") if holidays else np.array([], dtype="datetime64[D]")
    return np.busdaycalendar(holidays=hols)


def _to_date(d64: np.datetime64) -> dt.date:
    return d64.astype("datetime64[D]").astype(dt.date)


def first_notice_day(year: int, month: int, holidays: Optional[Sequence] = None) -> dt.date:
    """4th business day before the 1st business day of the delivery month."""
    cal = _busday_holidays(holidays)
    first = np.datetime64(dt.date(year, month, 1), "D")
    # First business day on/after the 1st of the month.
    first_bd = np.busday_offset(first, 0, roll="forward", busdaycal=cal)
    fnd = np.busday_offset(first_bd, -4, roll="backward", busdaycal=cal)
    return _to_date(fnd)


_CYCLE_CODES = list(MONTH_CODE_TO_MONTH)  # F H K N U X, in delivery order


def next_in_cycle(contract: Contract) -> Contract:
    """The next delivery month in the robusta cycle, wrapping the year at Nov->Jan."""
    i = _CYCLE_CODES.index(contract.month_code)
    if i + 1 < len(_CYCLE_CODES):
        code = _CYCLE_CODES[i + 1]
        year = contract.year
    else:
        code = _CYCLE_CODES[0]
        year = contract.year + 1
    return Contract(root=contract.root, month_code=code, year=year,
                    month=MONTH_CODE_TO_MONTH[code])


def chain_gaps(contracts: Sequence[Contract]) -> list:
    """Symbols missing from an otherwise-consecutive delivery chain.

    A continuous front-month series needs an unbroken chain; any gap breaks the
    splice at that roll. Returns the missing contract symbols (delivery order).
    """
    ordered = sorted(contracts, key=lambda c: (c.year, c.month))
    missing = []
    for cur, nxt in zip(ordered, ordered[1:]):
        expected = next_in_cycle(cur)
        while (expected.year, expected.month) != (nxt.year, nxt.month):
            missing.append(expected.symbol)
            expected = next_in_cycle(expected)
    return missing


def roll_date(
    contract: Contract, roll_offset_bd: int = 5, holidays: Optional[Sequence] = None
) -> dt.date:
    """`roll_offset_bd` business days before First Notice Day."""
    cal = _busday_holidays(holidays)
    fnd = np.datetime64(first_notice_day(contract.year, contract.month, holidays), "D")
    rd = np.busday_offset(fnd, -roll_offset_bd, roll="backward", busdaycal=cal)
    return _to_date(rd)
