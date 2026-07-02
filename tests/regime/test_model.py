"""Regime model: fit on synthetic two-state data, causal prediction, labeling."""
import numpy as np
import pandas as pd
import pytest

from ascentagri.regime.model import RegimeModel, label_states
from ascentagri.regime.types import RegimeLabel


@pytest.fixture(scope="module")
def two_state_panel():
    """Synthetic panel with an obvious calm→wild regime flip halfway."""
    rng = np.random.default_rng(42)
    n = 600
    dates = pd.bdate_range("2020-01-01", periods=n)
    calm = rng.normal(0.0008, 0.008, n // 2)
    wild = rng.normal(-0.001, 0.030, n - n // 2)
    rets = np.concatenate([calm, wild])
    vol21 = pd.Series(rets).rolling(21, min_periods=5).std().values * np.sqrt(252)
    panel = pd.DataFrame({
        "ret_21d": pd.Series(rets).rolling(21, min_periods=5).mean().values,
        "rvol_21d": vol21,
        "ret_1d": rets,
    }, index=dates)
    return panel


def test_fit_and_predict_shapes(two_state_panel):
    model = RegimeModel(k_regimes=2, hmm_restarts=3)
    assert model.fit(two_state_panel)
    probs = model.predict_probs(two_state_panel)
    assert probs.shape == (len(two_state_panel), 2)
    np.testing.assert_allclose(probs.sum(axis=1).values, 1.0, atol=1e-6)


def test_predict_on_new_data_matches_index(two_state_panel):
    """The walk-forward path: fit on train slice, score a longer span.
    (The ported Markov backend originally returned train-length output
    regardless of input — this locks in the fix.)"""
    train = two_state_panel.iloc[:400]
    model = RegimeModel(k_regimes=2, hmm_restarts=3)
    assert model.fit(train)
    span = two_state_panel  # 600 rows > train's 400
    probs = model.predict_probs(span)
    assert len(probs) == len(span)
    assert probs.index.equals(span.index)


def test_states_separate_calm_from_wild(two_state_panel):
    model = RegimeModel(k_regimes=2, hmm_restarts=3)
    assert model.fit(two_state_panel)
    probs = model.predict_probs(two_state_panel)
    dominant = probs.values.argmax(axis=1)
    first_half_state = np.bincount(dominant[50:280]).argmax()
    second_half_state = np.bincount(dominant[320:]).argmax()
    assert first_half_state != second_half_state


def test_insufficient_data_fails_gracefully():
    dates = pd.bdate_range("2024-01-01", periods=100)
    panel = pd.DataFrame({"x": np.random.default_rng(0).normal(size=100)},
                         index=dates)
    model = RegimeModel(k_regimes=2)
    assert model.fit(panel) is False
    probs = model.predict_probs(panel)     # uniform fallback
    np.testing.assert_allclose(probs.values, 0.5, atol=1e-9)


def test_label_states_k2_orders_by_mean():
    labels = label_states(np.array([0.01, -0.02]), np.array([0.1, 0.3]), 2)
    assert labels[0] == RegimeLabel.CALM_BULL
    assert labels[1] == RegimeLabel.CRISIS


def test_label_states_k3_assigns_crisis_to_worst():
    labels = label_states(np.array([0.02, 0.0, -0.03]),
                          np.array([0.1, 0.15, 0.4]), 3)
    assert labels[2] == RegimeLabel.CRISIS
    assert RegimeLabel.CALM_BULL in labels.values()


def test_label_states_none_means_uncertain():
    labels = label_states(None, None, 3)
    assert all(v == RegimeLabel.UNCERTAIN for v in labels.values())
