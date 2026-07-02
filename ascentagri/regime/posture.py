"""ascentagri/regime/posture.py — ported near-verbatim from Ascent Capital.
Converts regime engine output -> plain-English posture description.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class RegimeSummary:
    asof: str
    regime_label: str           # e.g. "stressed"
    probs: Dict[str, float]     # state -> probability
    confidence: float           # max_prob - second_max_prob  (0..1)
    days_in_regime: int
    posture: str                # constructive / selective / neutral / defensive / crisis / uncertain
    risk_multiplier: float      # 0.4 .. 1.0
    notes: str

    @property
    def posture_color(self) -> str:
        """Returns a hex color string suitable for HTML display."""
        return {
            "constructive": "22c55e",   # green
            "selective":    "84cc16",   # lime
            "neutral":      "94a3b8",   # slate
            "defensive":    "f59e0b",   # amber
            "crisis":       "ef4444",   # red
            "uncertain":    "64748b",   # muted
        }.get(self.posture, "94a3b8")

    def to_dict(self) -> dict:
        return {
            "asof":            self.asof,
            "regime_label":    self.regime_label,
            "probs":           self.probs,
            "confidence":      round(self.confidence, 4),
            "days_in_regime":  self.days_in_regime,
            "posture":         self.posture,
            "risk_multiplier": self.risk_multiplier,
            "notes":           self.notes,
        }


# ── Deterministic mapping ────────────────────────────────────────────────────

_POSTURE_MAP = {
    "calm_bull": {
        "posture":         "constructive",
        "risk_multiplier": 1.00,
        "notes":           "Trend intact. Full exposure appropriate. Favour momentum.",
    },
    "euphoric": {
        "posture":         "selective",
        "risk_multiplier": 0.85,
        "notes":           "Late-cycle signals. Avoid crowded momentum; trim size.",
    },
    "stressed": {
        "posture":         "defensive",
        "risk_multiplier": 0.65,
        "notes":           "Risk elevated. Reduce exposure; tighten stops.",
    },
    "crisis": {
        "posture":         "crisis",
        "risk_multiplier": 0.40,
        "notes":           "Capital preservation mode. Minimal exposure; await stabilisation.",
    },
    # fallback for any other label
    "neutral": {
        "posture":         "neutral",
        "risk_multiplier": 0.80,
        "notes":           "Mixed signals. Maintain balanced exposure; monitor regime closely.",
    },
}

_UNCERTAIN_OVERRIDE = {
    "posture":         "uncertain",
    "risk_multiplier": 0.75,
    "notes":           "Low regime confidence. Avoid large conviction bets; wait for clarity.",
}


def _confidence(probs: Dict[str, float]) -> float:
    """confidence = max_prob - second_max_prob.  Range 0..1."""
    sorted_p = sorted(probs.values(), reverse=True)
    if len(sorted_p) < 2:
        return sorted_p[0] if sorted_p else 0.0
    return sorted_p[0] - sorted_p[1]


def compute_posture_from_regime(
    asof: str,
    regime_label: str,
    probs: Dict[str, float],
    days_in_regime: Optional[int] = None,
    min_confidence: float = 0.50,
) -> RegimeSummary:
    """Pure function — no I/O.

    asof            : date string YYYY-MM-DD
    regime_label    : label from regime/decision.py  e.g. "stressed"
    probs           : dict of state -> probability
    days_in_regime  : how long we've been in this regime (0 if unknown)
    min_confidence  : if confidence < this threshold, override to 'uncertain'
    """
    conf = _confidence(probs)
    dwell = days_in_regime or 0

    label_lower = regime_label.lower().strip()

    if conf < min_confidence:
        meta = _UNCERTAIN_OVERRIDE
    else:
        meta = _POSTURE_MAP.get(label_lower, _POSTURE_MAP["neutral"])

    return RegimeSummary(
        asof=asof,
        regime_label=regime_label,
        probs=probs,
        confidence=round(conf, 4),
        days_in_regime=dwell,
        posture=meta["posture"],
        risk_multiplier=meta["risk_multiplier"],
        notes=meta["notes"],
    )
