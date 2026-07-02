"""Hysteresis decision layer: dwell, asymmetric thresholds, entropy filter."""
import numpy as np
import pandas as pd

from ascentagri.regime.decision import (
    RegimeDecisionEngine,
    _HysteresisStateMachine,
    _entropy,
)
from ascentagri.regime.types import RegimeLabel

LABELS = {0: RegimeLabel.CALM_BULL, 1: RegimeLabel.CRISIS}


def _machine(**kw):
    defaults = dict(
        initial_regime=0,
        enter_threshold=0.55,
        min_dwell_days=3,
        entropy_uncertain_threshold=0.99,
        severity={0: 0, 1: 3},
        downgrade_threshold=0.40,
        upgrade_threshold=0.70,
    )
    defaults.update(kw)
    return _HysteresisStateMachine(**defaults)


def test_entropy_bounds():
    assert _entropy(np.array([1.0, 0.0])) < 0.01
    assert abs(_entropy(np.array([0.5, 0.5])) - 1.0) < 1e-9


def test_no_switch_before_min_dwell():
    m = _machine()
    # crisis dominant for 2 days only — no switch yet (min_dwell_days=3)
    for _ in range(2):
        state, flag, _, _ = m.step(np.array([0.3, 0.7]))
        assert state == 0 and not flag
    # third consecutive day commits the transition
    state, flag, _, _ = m.step(np.array([0.3, 0.7]))
    assert state == 1 and flag


def test_sub_threshold_day_resets_streak():
    """A dominant-but-below-threshold day resets the candidate streak.
    Only reachable in the upgrade direction (threshold 0.70 > 0.5); in the
    downgrade direction any K=2 dominant prob (≥0.5) clears the 0.40 bar.
    (NB: a day where the CURRENT regime is dominant again does NOT reset the
    candidate — ported source semantics.)"""
    m = _machine(initial_regime=1)        # start in crisis
    m.step(np.array([0.75, 0.25]))        # calm candidate, streak 1
    m.step(np.array([0.75, 0.25]))        # streak 2
    m.step(np.array([0.60, 0.40]))        # calm dominant but < 0.70 → reset
    # two more strong calm days rebuild the streak to 2 — still no switch
    m.step(np.array([0.75, 0.25]))
    state, flag, _, _ = m.step(np.array([0.75, 0.25]))
    assert state == 1 and not flag        # reset worked: needs a 3rd day
    state, flag, _, _ = m.step(np.array([0.75, 0.25]))
    assert state == 0 and flag            # third consecutive day switches


def test_asymmetric_thresholds_downgrade_faster_than_upgrade():
    # downgrade (calm→crisis): prob 0.60 ≥ downgrade_threshold 0.40 → counts
    # ([0.45, 0.55] would trip the entropy filter at threshold 0.99)
    m = _machine()
    for _ in range(3):
        state, _, _, _ = m.step(np.array([0.40, 0.60]))
    assert state == 1, "downgrade should trigger at 0.60 > 0.40 threshold"

    # upgrade (crisis→calm): prob 0.60 < upgrade_threshold 0.70 → never counts
    m = _machine(initial_regime=1)
    for _ in range(10):
        state, _, _, _ = m.step(np.array([0.60, 0.40]))
    assert state == 1, "upgrade should NOT trigger below the 0.70 threshold"


def test_high_entropy_flags_uncertain_without_state_change():
    m = _machine(entropy_uncertain_threshold=0.90)
    _, _, _, is_uncertain = m.step(np.array([0.51, 0.49]))
    assert is_uncertain
    state, _, _, _ = m.step(np.array([0.95, 0.05]))
    assert state == 0


def test_decision_engine_emits_signals_with_expected_fields():
    dates = pd.bdate_range("2024-01-01", periods=30)
    probs = np.zeros((30, 2))
    probs[:15, 0], probs[:15, 1] = 0.9, 0.1
    probs[15:, 0], probs[15:, 1] = 0.1, 0.9
    prob_df = pd.DataFrame(probs, index=dates, columns=[0, 1])

    eng = RegimeDecisionEngine(state_labels=LABELS, min_dwell_days=3,
                               entropy_uncertain_threshold=0.99)
    signals = eng.process(prob_df)
    assert len(signals) == 30
    assert signals[0].label == RegimeLabel.CALM_BULL
    assert signals[-1].label == RegimeLabel.CRISIS
    assert sum(s.transition_flag for s in signals) == 1
    frame = eng.process_to_frame(prob_df)
    for col in ["label", "risk_multiplier", "entropy", "dwell_days",
                "prob_0", "prob_1", "sleeve_trend", "sleeve_meanrev"]:
        assert col in frame.columns
    # crisis rows carry the defensive risk multiplier
    assert (frame.loc[frame["label"] == "crisis", "risk_multiplier"] == 0.40).all()
