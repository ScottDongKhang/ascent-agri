"""Tests for the Databento -> per-contract-CSV transform (pure, no network)."""
import numpy as np
import pandas as pd
import pytest

from ascentagri.databento_fetch import (
    parse_databento_symbol,
    to_output_filename,
    split_and_format,
    price_sanity_warnings,
)


def test_parse_symbol_plain():
    c = parse_databento_symbol("RCN23")
    assert (c.month, c.year, c.month_code) == (7, 2023, "N")


def test_parse_symbol_with_space_and_single_digit_year():
    assert parse_databento_symbol("RC N3").year == 2023
    assert parse_databento_symbol("RC N3").month == 7


def test_parse_symbol_four_digit_year():
    assert parse_databento_symbol("RCX2024").year == 2024
    assert parse_databento_symbol("RCX2024").month == 11


def test_parse_symbol_rejects_non_robusta_month():
    # M = June is a valid futures code but NOT in the robusta cycle (F H K N U X)
    assert parse_databento_symbol("RCM24") is None


def test_parse_symbol_rejects_spreads_and_junk():
    assert parse_databento_symbol("RCN23-RCF24") is None
    assert parse_databento_symbol("GARBAGE") is None


def test_output_filename_uses_RM_root():
    # source is Databento 'RC' but our pipeline/checklist names files RM<code><yy>
    assert to_output_filename(parse_databento_symbol("RCN23")) == "RMN23.csv"


def _fake_to_df():
    idx = pd.to_datetime(["2024-07-01", "2024-07-02", "2024-07-01", "2024-07-02"])
    idx.name = "ts_event"
    return pd.DataFrame(
        {
            "open": [3900, 3910, 3950, 3960],
            "high": [3950, 3940, 3990, 3990],
            "low": [3880, 3890, 3930, 3940],
            "close": [3920, 3930, 3970, 3980],
            "volume": [1000, 1100, 800, 900],
            "symbol": ["RCN24", "RCN24", "RCU24", "RCU24"],
            "rtype": [34, 34, 34, 34],  # an extra column to be ignored
        },
        index=idx,
    )


def test_split_and_format_keys_and_columns():
    out = split_and_format(_fake_to_df())
    assert set(out) == {"RMN24.csv", "RMU24.csv"}
    f = out["RMN24.csv"]
    assert list(f.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert f["close"].tolist() == [3920, 3930]
    assert f["date"].is_monotonic_increasing


def test_split_and_format_skips_unparseable_symbols():
    df = _fake_to_df()
    df.loc[df.index[0], "symbol"] = "RCN24-RCU24"  # a spread row -> dropped
    out = split_and_format(df)
    # the spread row is dropped; RMN24 keeps only its one good row
    assert "RMN24.csv" in out
    assert len(out["RMN24.csv"]) == 1


def test_price_sanity_warns_on_scaling_error():
    good = {"RMN24.csv": pd.DataFrame({"close": [3900, 3920]})}
    bad = {"RMN24.csv": pd.DataFrame({"close": [3.9e-6, 3.92e-6]})}
    assert price_sanity_warnings(good) == []
    assert price_sanity_warnings(bad)  # non-empty -> a warning was raised
