"""Bayesian sleeve meta-learner: posterior updates, trust gating, cap."""
import numpy as np

from ascentagri.alpha.meta_learner import SleeveMetaLearner, _enforce_cap

DEFAULTS = {"trend": 0.75, "meanrev": 0.25}


def _learner(tmp_path):
    return SleeveMetaLearner(posteriors_path=tmp_path / "posteriors.json")


def test_no_data_returns_none(tmp_path):
    ml = _learner(tmp_path)
    assert ml.get_weights("calm_bull", DEFAULTS) is None


def test_below_min_obs_returns_none(tmp_path):
    ml = _learner(tmp_path)
    ml.update_rebalance("calm_bull", {"trend": 0.02, "meanrev": 0.01})
    assert ml.get_weights("calm_bull", DEFAULTS) is None   # n=1 < 3


def test_posterior_update_moves_mu_toward_observation(tmp_path):
    ml = _learner(tmp_path)
    for _ in range(5):
        ml.update_rebalance("calm_bull", {"trend": 0.05, "meanrev": -0.02})
    state = ml._state["calm_bull"]
    assert state["trend"]["mu"] > 0.02
    assert state["meanrev"]["mu"] < 0.0
    assert state["trend"]["n"] == 5


def test_weights_sum_to_one_and_favor_high_ic_sleeve(tmp_path):
    ml = _learner(tmp_path)
    for _ in range(10):
        ml.update_rebalance("calm_bull", {"trend": 0.06, "meanrev": 0.001})
    w = ml.get_weights("calm_bull", DEFAULTS)
    assert w is not None
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert w["trend"] > w["meanrev"]


def test_invalid_regime_ignored(tmp_path):
    ml = _learner(tmp_path)
    ml.update_rebalance("weird_regime", {"trend": 0.05})
    assert ml._state == {}
    assert ml.get_weights("weird_regime", DEFAULTS) is None


def test_state_persists_across_instances(tmp_path):
    ml = _learner(tmp_path)
    for _ in range(4):
        ml.update_rebalance("stressed", {"trend": -0.01, "meanrev": 0.03})
    ml2 = _learner(tmp_path)
    w = ml2.get_weights("stressed", DEFAULTS)
    assert w is not None
    assert w["meanrev"] > 0.25   # meanrev earned weight in stressed


def test_enforce_cap_redistributes():
    w = _enforce_cap({"a": 0.95, "b": 0.05}, cap=0.85)
    assert w["a"] <= 0.85 + 1e-9
    assert abs(sum(w.values()) - 1.0) < 1e-9
