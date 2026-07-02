"""ascent-agri configuration — single dataclass, coffee-tuned defaults.

All knobs for the alpha stack, exposure overlays, backtest engine, and
walk-forward evaluation live here so the demo script, notebook, and tests
share one source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class AgriConfig:
    # ── instrument ─────────────────────────────────────────────────────
    series_name: str = "ROBUSTA"

    # ── alpha stack ────────────────────────────────────────────────────
    # Rolling time-series z-score window (replaces Ascent's cross-sectional
    # z-score, which silently zeroes out at N=1 instruments).
    ts_z_window: int = 252
    ts_z_min_periods: int = 60
    # Base sleeve weights (regime layer adjusts these per date, then renormalizes)
    sleeve_weights: Dict[str, float] = field(
        default_factory=lambda: {"trend": 0.75, "meanrev": 0.25}
    )
    # Long-only score→position mapping: position = clip(score, 0, score_cap)/score_cap
    score_cap: float = 1.5
    max_exposure: float = 1.0

    # ── exposure overlays (coffee-tuned) ───────────────────────────────
    ma_window: int = 200
    ma_multiplier: float = 0.70          # below own 200d MA → scale exposure
    vol_target: float = 0.20             # robusta runs ~30% annualized vol
    vol_lookback: int = 21
    vol_floor: float = 0.25
    vol_cap: float = 1.00

    # ── backtest engine ────────────────────────────────────────────────
    initial_capital: float = 1_000_000.0
    spread_bps: float = 5.0
    impact_bps: float = 5.0
    rebalance_freq_days: int = 5
    execution_delay: int = 1

    # ── walk-forward evaluation ────────────────────────────────────────
    wf_train_days: int = 504
    wf_test_days: int = 63
    wf_step_days: int = 63
    wf_purge_days: int = 5
    wf_min_train_days: int = 252
    wf_regime_k: int = 3                 # fixed K inside WF folds (no per-fold selection)
    wf_hmm_restarts: int = 5

    # ── regime engine overrides (merged over regime.types defaults) ───
    regime_overrides: Dict = field(default_factory=dict)


def get_config(**overrides) -> AgriConfig:
    """Build the default config, optionally overriding any field by name."""
    cfg = AgriConfig()
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"AgriConfig has no field '{key}'")
        setattr(cfg, key, value)
    return cfg
