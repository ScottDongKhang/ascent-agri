"""Regime engine end-to-end on synthetic data + crisis override rule."""
import numpy as np
import pandas as pd
import pytest

from ascentagri.regime.engine import RegimeEngine, apply_crisis_override


@pytest.fixture(scope="module")
def prices():
    rng = np.random.default_rng(21)
    n = 700
    dates = pd.bdate_range("2020-01-01", periods=n)
    calm = rng.normal(0.0006, 0.010, n // 2)
    wild = rng.normal(-0.0012, 0.032, n - n // 2)
    return pd.Series(2200 * np.exp(np.cumsum(np.concatenate([calm, wild]))),
                     index=dates)


def test_fit_produces_signal_cache(prices):
    eng = RegimeEngine()
    eng.fit(prices, run_model_selection=False, k_override=2, hmm_restarts=3)
    frame = eng.get_signal_series()
    assert not frame.empty
    assert frame.index.equals(prices.index)
    for col in ["label", "risk_multiplier", "crisis_override"]:
        assert col in frame.columns


def test_get_signal_is_causal(prices):
    eng = RegimeEngine()
    eng.fit(prices, run_model_selection=False, k_override=2, hmm_restarts=3)
    mid_date = prices.index[350]
    sig = eng.get_signal(mid_date)
    assert sig is not None
    assert sig.date <= mid_date
    # a date before the series has no signal
    assert eng.get_signal(prices.index[0] - pd.Timedelta(days=10)) is None


def test_predict_signal_frame_on_longer_span(prices):
    train = prices.iloc[:500]
    eng = RegimeEngine()
    eng.fit(train, run_model_selection=False, k_override=2, hmm_restarts=3)
    from ascentagri.regime.features import RegimeFeatureBuilder
    panel = RegimeFeatureBuilder(prices).build()
    frame = eng.predict_signal_frame(panel, prices=prices)
    assert len(frame) == len(prices)


def test_crisis_override_fires_on_crash():
    dates = pd.bdate_range("2024-01-01", periods=300)
    rng = np.random.default_rng(1)
    rets = rng.normal(0.0005, 0.012, 300)
    rets[250:256] = -0.08           # 6-day cluster crash, 5d ret ≪ -10%, vol spike
    prices = pd.Series(3000 * np.exp(np.cumsum(rets)), index=dates)
    signal_df = pd.DataFrame({
        "label": "calm_bull", "risk_multiplier": 1.0,
    }, index=dates)
    out = apply_crisis_override(signal_df, prices,
                                ret_5d_threshold=-0.10, vol_threshold=0.45)
    assert out["crisis_override"].any()
    overridden = out[out["crisis_override"]]
    assert (overridden["label"] == "crisis").all()
    assert (overridden["risk_multiplier"] == 0.40).all()
    # calm period untouched
    assert not out["crisis_override"].iloc[:200].any()


def test_crisis_override_ignores_calm_series():
    dates = pd.bdate_range("2024-01-01", periods=200)
    prices = pd.Series(np.linspace(100, 120, 200), index=dates)
    signal_df = pd.DataFrame({"label": "calm_bull", "risk_multiplier": 1.0},
                             index=dates)
    out = apply_crisis_override(signal_df, prices)
    assert not out["crisis_override"].any()


def test_should_refit_after_interval(prices):
    eng = RegimeEngine(config={"regime_refit_every_days": 63})
    eng.fit(prices, run_model_selection=False, k_override=2, hmm_restarts=3)
    assert not eng.should_refit(prices.index[-1])
    assert eng.should_refit(prices.index[-1] + pd.Timedelta(days=90))
