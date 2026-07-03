"""Farm-gate economics — the price-transmission lens.

Connects the world futures price to what it means for a Dak Lak grower:
ICE robusta trades in USD per metric tonne; Vietnamese farm-gate prices are
quoted in VND per kilogram. The conversion is mechanical but the framing
matters — it turns "the market moved 3%" into "a farmer's crop is worth
~3,000 đồng/kg less this week", which is the number extension services and
growers actually discuss.

Honesty notes:
  * This is a FUTURES-EQUIVALENT number, not an observed farm-gate quote.
    Real farm-gate prices trade at a differential (basis) to the exchange,
    reflecting local logistics, quality, and dealer margins. We publish the
    conversion, clearly labeled, not a pretend farm-gate survey.
  * USD/VND comes from Yahoo (VND=X). It is an optional input everywhere —
    a failed FX fetch never blocks the daily publish.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
USDVND_PATH = ROOT / "data" / "processed" / "usdvnd.csv"


def fetch_usdvnd(start: str = "2017-01-01") -> pd.DataFrame:
    """Fetch USD/VND from Yahoo (VND=X). Returns tidy [date, vnd_per_usd]."""
    import yfinance as yf
    df = yf.download("VND=X", start=start, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError("yfinance VND=X returned no rows")
    close = df["Close"]
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    out = pd.DataFrame({
        "date": pd.to_datetime(close.index).tz_localize(None),
        "vnd_per_usd": pd.to_numeric(close.values, errors="coerce"),
    }).dropna().reset_index(drop=True)
    return out


def load_usdvnd(fetch_if_missing: bool = True) -> Optional[pd.Series]:
    """Cached USD/VND as a date-indexed Series. Returns None on any failure —
    farm-gate economics is an optional layer, never a publish blocker."""
    try:
        if not USDVND_PATH.exists():
            if not fetch_if_missing:
                return None
            USDVND_PATH.parent.mkdir(parents=True, exist_ok=True)
            df = fetch_usdvnd()
            df.to_csv(USDVND_PATH, index=False)
            print(f"[economics] wrote {USDVND_PATH.name}: {len(df)} rows "
                  f"({df['date'].min().date()} -> {df['date'].max().date()})")
        df = pd.read_csv(USDVND_PATH, parse_dates=["date"])
        return df.set_index("date")["vnd_per_usd"].sort_index()
    except Exception as exc:
        log.warning("economics: USD/VND unavailable (%s) — layer disabled", exc)
        return None


def futures_usd_tonne_to_vnd_kg(usd_per_tonne: float,
                                vnd_per_usd: float) -> float:
    """ICE robusta quote (USD/metric tonne) → futures-equivalent VND/kg."""
    return usd_per_tonne * vnd_per_usd / 1000.0


def transmission_line(robusta_usd_tonne: float, vnd_per_usd: float,
                      chg_1m: Optional[float] = None) -> str:
    """One plain-English sentence for the site/brief.

    Example: 'At today's exchange rate the futures price is worth ~94,600
    đồng/kg of green coffee; a month ago the same kilogram priced ~3,100
    đồng higher.'
    """
    vnd_kg = futures_usd_tonne_to_vnd_kg(robusta_usd_tonne, vnd_per_usd)
    line = (f"At the current exchange rate the robusta futures price is "
            f"worth ~{vnd_kg:,.0f} đồng/kg of green coffee (futures-"
            f"equivalent, before local basis)")
    if chg_1m is not None and abs(chg_1m) > 0.001:
        delta = vnd_kg - vnd_kg / (1 + chg_1m)
        direction = "less" if delta < 0 else "more"
        line += (f"; a month of price movement is worth ~{abs(delta):,.0f} "
                 f"đồng/kg {direction} to a grower")
    return line + "."


def _vn_num(x: float) -> str:
    """Vietnamese thousands formatting (dots): 94808 → '94.808'."""
    return f"{x:,.0f}".replace(",", ".")


def transmission_line_vi(robusta_usd_tonne: float, vnd_per_usd: float,
                         chg_1m: Optional[float] = None) -> str:
    """Vietnamese variant of the farm-gate sentence, written independently."""
    vnd_kg = futures_usd_tonne_to_vnd_kg(robusta_usd_tonne, vnd_per_usd)
    line = (f"Theo tỷ giá hiện tại, giá robusta kỳ hạn tương đương "
            f"~{_vn_num(vnd_kg)} đồng/kg cà phê nhân (quy đổi từ giá kỳ hạn, "
            f"chưa tính chênh lệch giá tại địa phương)")
    if chg_1m is not None and abs(chg_1m) > 0.001:
        delta = vnd_kg - vnd_kg / (1 + chg_1m)
        direction = "thấp hơn" if delta < 0 else "cao hơn"
        line += (f"; biến động giá một tháng qua tương đương "
                 f"~{_vn_num(abs(delta))} đồng/kg {direction} cho người trồng")
    return line + "."
