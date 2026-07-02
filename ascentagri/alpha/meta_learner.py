"""ascentagri/alpha/meta_learner.py — ported near-verbatim from Ascent Capital.

Bayesian IC meta-learner for alpha sleeve weights.

Maintains per-(regime, sleeve) Gaussian posteriors. Applies a Gaussian
conjugate update after each rebalance holding period. Derives Kelly-inspired
weights blended toward regime defaults by confidence
alpha_conf = min(1.0, n / 20) where n is the number of rebalance observations.

Only change from the source: state/log paths point at this repo's
data/interim/ and outputs/ directories, and the AI-prior hook is kept but
unused here (no LLM layer in this artifact).
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
POSTERIORS_PATH = _REPO_ROOT / "data" / "interim" / "sleeve_posteriors.json"
IC_LOG_PATH = _REPO_ROOT / "outputs" / "sleeve_ic_log.jsonl"
WEIGHTS_LOG_PATH = _REPO_ROOT / "outputs" / "meta_learner_weights.jsonl"

_PRIOR_VAR = 0.005
_OBS_NOISE = 0.003
_FULL_TRUST_N = 20
_MIN_OBS_TRUST = 3
_AI_PRIOR_MAX_DELTA = 0.010
_MAX_SINGLE_SLEEVE = 0.85
_VALID_LABELS = {"calm_bull", "stressed", "crisis", "euphoric", "uncertain"}


class SleeveMetaLearner:
    """Per-(regime, sleeve) Bayesian meta-learner for alpha sleeve weights.

    State stored as JSON:
    { "calm_bull": { "trend": { "mu": 0.015, "var": 0.003, "n": 29 }, ... }, ... }
    """

    def __init__(self, posteriors_path: Path = POSTERIORS_PATH):
        self._path = Path(posteriors_path)
        self._state: Dict[str, Dict[str, Dict]] = {}
        self._load()

    def get_weights(
        self,
        regime: str,
        regime_defaults: Dict[str, float],
        ai_prior: Optional[Dict[str, float]] = None,
    ) -> Optional[Dict[str, float]]:
        """Derive sleeve weights for a given regime.

        Returns None if no posterior data exists for this regime, or any
        sleeve has fewer than _MIN_OBS_TRUST observations. Caller should
        fall back to the regime-default weights on None.

        ai_prior: {sleeve: delta_ic} — shifts effective mu for this call only.
        Bounded to ±_AI_PRIOR_MAX_DELTA per sleeve. Does NOT write to posterior.
        """
        regime = str(regime).lower()
        if regime not in _VALID_LABELS:
            return None

        regime_state = self._state.get(regime, {})
        if not regime_state:
            return None

        min_n = min((v.get("n", 0) for v in regime_state.values()), default=0)
        if min_n < _MIN_OBS_TRUST:
            return None

        if any(sleeve not in regime_state for sleeve in regime_defaults):
            return None

        raw_weights: Dict[str, float] = {}
        for sleeve in regime_defaults:
            s = regime_state.get(sleeve)
            if s is None:
                raw_weights[sleeve] = 0.0
                continue
            mu = float(s.get("mu", 0.0))
            var = max(float(s.get("var", _PRIOR_VAR)), 1e-9)
            if ai_prior and sleeve in ai_prior:
                delta = max(-_AI_PRIOR_MAX_DELTA, min(_AI_PRIOR_MAX_DELTA, float(ai_prior[sleeve])))
                mu = mu + delta
            raw_weights[sleeve] = max(0.0, mu / math.sqrt(var))

        total_raw = sum(raw_weights.values())
        if total_raw <= 0:
            return None

        kelly_w = {s: w / total_raw for s, w in raw_weights.items()}

        result: Dict[str, float] = {}
        for sleeve, default_w in regime_defaults.items():
            s = regime_state.get(sleeve, {})
            n = int(s.get("n", 0))
            alpha_conf = min(1.0, n / _FULL_TRUST_N)
            result[sleeve] = alpha_conf * kelly_w.get(sleeve, 0.0) + (1 - alpha_conf) * default_w

        total = sum(result.values())
        if total <= 0:
            return None

        result = {s: w / total for s, w in result.items()}
        result = _enforce_cap(result, _MAX_SINGLE_SLEEVE)

        if abs(sum(result.values()) - 1.0) > 1e-4:
            t = sum(result.values()) or 1.0
            result = {s: w / t for s, w in result.items()}

        return result

    def update_rebalance(self, regime: str, sleeve_ic: Dict[str, float]) -> None:
        """Gaussian conjugate update for each sleeve after a rebalance
        holding period. sleeve_ic: {sleeve_name: realized_ic_for_period}"""
        regime = str(regime).lower()
        if regime not in _VALID_LABELS:
            log.warning("[MetaLearner] Unknown regime '%s' — skipping update", regime)
            return

        if regime not in self._state:
            self._state[regime] = {}

        for sleeve, ic_obs in sleeve_ic.items():
            _ic = float(ic_obs)
            if not math.isfinite(_ic):
                log.warning("[MetaLearner] Non-finite IC for sleeve=%s (%.4g) — skipping", sleeve, _ic)
                continue

            if sleeve not in self._state[regime]:
                self._state[regime][sleeve] = {"mu": 0.0, "var": _PRIOR_VAR, "n": 0}

            s = self._state[regime][sleeve]
            mu, var, n = float(s["mu"]), float(s["var"]), int(s["n"])

            precision_prior = 1.0 / var
            precision_obs = 1.0 / _OBS_NOISE
            precision_post = precision_prior + precision_obs
            mu_post = (mu * precision_prior + _ic * precision_obs) / precision_post
            var_post = 1.0 / precision_post

            self._state[regime][sleeve] = {
                "mu": round(mu_post, 6),
                "var": round(var_post, 6),
                "n": n + 1,
            }

        self._save()
        log.info("[MetaLearner] Updated posteriors: regime=%s sleeves=%d", regime, len(sleeve_ic))

    def seed_from_ic_log(self, ic_log_path: Path = IC_LOG_PATH) -> int:
        """Seed posteriors from a sleeve-IC jsonl log using observed mean IC
        per sleeve. Only seeds (regime, sleeve) pairs not already in state.
        Returns count of log entries processed."""
        ic_log_path = Path(ic_log_path)
        if not ic_log_path.exists():
            return 0

        entries = []
        for line in ic_log_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                continue

        by_regime_sleeve: Dict = defaultdict(lambda: defaultdict(list))
        for entry in entries:
            regime = str(entry.get("regime", "")).lower()
            if not regime or regime not in _VALID_LABELS:
                continue
            for sleeve, stats in entry.get("sleeves", {}).items():
                ic = stats.get("mean_ic")
                if ic is not None:
                    by_regime_sleeve[regime][sleeve].append(float(ic))

        seeded = 0
        for regime, sleeves in by_regime_sleeve.items():
            if regime not in self._state:
                self._state[regime] = {}
            for sleeve, ics in sleeves.items():
                if sleeve not in self._state[regime]:
                    mu_seed = sum(ics) / len(ics)
                    self._state[regime][sleeve] = {
                        "mu": round(mu_seed, 6),
                        "var": _PRIOR_VAR,
                        "n": len(ics),
                    }
                    seeded += 1

        if seeded > 0:
            self._save()
            log.info("[MetaLearner] Seeded %d (regime, sleeve) posteriors from ic_log", seeded)

        return len(entries)

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._state = json.loads(self._path.read_text())
            except Exception as e:
                log.warning("[MetaLearner] Failed to load posteriors (%s) — fresh start", e)
                self._state = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.parent / (self._path.name + ".tmp")
        tmp.write_text(json.dumps(self._state, indent=2))
        tmp.replace(self._path)


def _enforce_cap(weights: Dict[str, float], cap: float) -> Dict[str, float]:
    result = dict(weights)
    for _ in range(50):
        capped = {s: w for s, w in result.items() if w > cap}
        if not capped:
            break
        freed = sum(w - cap for w in capped.values())
        uncapped = {s: w for s, w in result.items() if w <= cap}
        if not uncapped:
            break
        for s in capped:
            result[s] = cap
        total_uncapped = sum(uncapped.values()) or 1.0
        for s, w in uncapped.items():
            result[s] = w + freed * (w / total_uncapped)
    return result


def log_weight_proposal(regime: str, weights: Dict[str, float], source: str) -> None:
    """Append to meta_learner_weights.jsonl for audit trail."""
    from datetime import date
    WEIGHTS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "date": date.today().isoformat(),
        "regime": regime,
        "source": source,
        "weights": {s: round(w, 4) for s, w in weights.items()},
    }
    try:
        with open(WEIGHTS_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        log.warning("[MetaLearner] log_weight_proposal write failed: %s", e)
