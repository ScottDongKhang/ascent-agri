"""Tests for the Yahoo (yfinance) -> tidy OHLCV transform (pure, no network)."""
import pandas as pd
import pytest

from ascentagri.vendor_fetch import tidy_yahoo


def _flat_frame():
    idx = pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-04"])
    idx.name = "Date"
    return pd.DataFrame(
        {"Open": [180, 178, 181], "High": [182, 179, 183], "Low": [177, 176, 180],
         "Close": [181, 178, 182], "Volume": [1000, 1200, 900]},
        index=idx,
    )


def _multiindex_frame():
    # yfinance single-ticker download often returns (Price, Ticker) MultiIndex columns
    f = _flat_frame()
    f.columns = pd.MultiIndex.from_product([list(f.columns), ["KC=F"]])
    return f


def test_tidy_flat_columns():
    out = tidy_yahoo(_flat_frame())
    assert list(out.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert out["date"].is_monotonic_increasing          # sorted ascending
    assert out["close"].tolist() == [178, 181, 182]      # reordered by date


def test_tidy_multiindex_columns():
    out = tidy_yahoo(_multiindex_frame())
    assert list(out.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert out["close"].tolist() == [178, 181, 182]


def test_tidy_output_is_loader_compatible(tmp_path):
    # the written CSV must round-trip through the project's own loader
    from ascentagri.loader import load_contract_csv
    out = tidy_yahoo(_flat_frame())
    p = tmp_path / "coffee.csv"
    out.to_csv(p, index=False)
    df = load_contract_csv(p)
    assert "close" in df.columns and len(df) == 3
