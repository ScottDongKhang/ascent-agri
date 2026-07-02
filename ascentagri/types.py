"""Shared type definitions for ascent-agri.

`AgentOutput` keeps the same shape as the multi-agent platform this project
was ported from: a standardized record a strategy run emits, so downstream
consumers (reports, notebooks) never reach into pipeline internals.
Here there is a single instrument, so `target_weights` holds one entry —
the target exposure for the continuous robusta series.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class AgentOutput:
    """Standardized output emitted by a strategy run.

    Fields:
        agent_id:       identifier, e.g. "robusta"
        as_of_date:     the trading date this output corresponds to
        target_weights: {series_name: target exposure in [0, 1]}
        regime_signal:  regime label for this date (e.g. "calm_bull"),
                        None if the regime engine produced no signal
        alpha_scores:   optional DataFrame of sleeve scores (diagnostics)
        skill_score:    rolling OOS Sharpe of the strategy, None if
                        insufficient history
        metadata:       freeform diagnostics dict
    """
    agent_id: str
    as_of_date: date
    target_weights: Dict[str, float]
    regime_signal: Optional[str] = None
    alpha_scores: Optional[pd.DataFrame] = None
    skill_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.agent_id:
            raise ValueError("agent_id cannot be empty")
        if not isinstance(self.target_weights, dict):
            raise TypeError("target_weights must be a dict")

    @property
    def n_positions(self) -> int:
        return sum(1 for w in self.target_weights.values() if w > 0)

    @property
    def total_weight(self) -> float:
        return sum(self.target_weights.values())

    def summary(self) -> str:
        skill_str = f"{self.skill_score:.3f}" if self.skill_score is not None else "N/A"
        return (
            f"[AgentOutput] {self.agent_id} | {self.as_of_date} | "
            f"{self.n_positions} positions | regime={self.regime_signal} | "
            f"skill={skill_str}"
        )
