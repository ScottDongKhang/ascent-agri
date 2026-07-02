"""ascentagri/backtest/costs.py — ported near-verbatim from Ascent Capital.

Transaction cost model for backtesting.

Cost components:
1. Spread cost: half bid-ask spread per side
2. Market impact: proportional to sqrt(participation rate) × volatility
3. Commission: per-contract/share fee
"""
from __future__ import annotations
import math


def estimate_trade_cost(
    price: float,
    shares: int,
    daily_volume: float,
    volatility: float,
    spread_bps: float = 5.0,
    impact_mult: float = 0.1,
    commission_per_share: float = 0.0,
) -> dict:
    """Estimate total cost for a single trade. Returns dict with breakdown."""
    if shares == 0 or price <= 0:
        return {"total_cost": 0.0, "spread_cost": 0.0, "impact_cost": 0.0, "commission": 0.0}

    notional = abs(shares) * price

    # Spread cost
    spread_cost = notional * (spread_bps / 10_000)

    # Market impact (simplified Almgren-Chriss)
    if daily_volume > 0:
        participation = abs(shares) / daily_volume
        impact_cost = notional * impact_mult * volatility * math.sqrt(participation)
    else:
        impact_cost = notional * 0.001  # fallback: 10bps

    # Commission
    commission = abs(shares) * commission_per_share

    total = spread_cost + impact_cost + commission

    return {
        "total_cost": total,
        "total_cost_bps": (total / notional * 10_000) if notional > 0 else 0,
        "spread_cost": spread_cost,
        "impact_cost": impact_cost,
        "commission": commission,
        "notional": notional,
    }


def flat_cost_model(turnover_fraction: float, cost_bps: float = 10.0) -> float:
    """Simple flat cost model: total cost = turnover × cost_per_unit.
    Returns cost as a fraction of portfolio."""
    return turnover_fraction * (cost_bps / 10_000)
