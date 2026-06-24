"""Tests for reading Barchart per-contract CSVs."""
from pathlib import Path

import pandas as pd
import pytest

from ascentagri.loader import load_contract_csv

FIX = Path(__file__).parent / "fixtures" / "RMK24_sample.csv"


def test_loads_barchart_csv_with_close_from_last():
    df = load_contract_csv(FIX)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert "close" in df.columns
    # 'Last' maps to close
    assert df.loc["2024-04-16", "close"] == pytest.approx(3900.0)
    assert df.loc["2024-04-18", "close"] == pytest.approx(3925.0)


def test_skips_barchart_footer_row():
    df = load_contract_csv(FIX)
    # 4 real data rows, footer dropped
    assert len(df) == 4


def test_index_is_sorted_ascending():
    df = load_contract_csv(FIX)
    assert df.index.is_monotonic_increasing


def test_keeps_ohlcv_columns():
    df = load_contract_csv(FIX)
    for col in ("open", "high", "low", "close", "volume"):
        assert col in df.columns
