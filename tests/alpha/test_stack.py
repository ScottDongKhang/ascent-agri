"""Long-only score→position mapping and the minimal combiner."""
import numpy as np
import pandas as pd
import pytest

from ascentagri.alpha.stack import (
    build_features,
    build_positions,
    score_to_position,
    sleeve_weight_frame,
)
from ascentagri.config import get_config


@pytest.fixture
def close():
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2020-01-01", periods=700)
    rets = rng.normal(0.0004, 0.018, len(dates))
    return pd.Series(2500 * np.exp(np.cumsum(rets)), index=dates)


def test_score_to_position_long_only_and_bounded():
    dates = pd.bdate_range("2023-01-02", periods=5)
    score = pd.Series([-2.0, -0.1, 0.0, 0.75, 5.0], index=dates)
    pos = score_to_position(score, score_cap=1.5, max_exposure=1.0)
    assert (pos >= 0).all(), "long-only: no short coffee positions, ever"
    assert (pos <= 1.0).all()
    assert pos.iloc[0] == 0.0 and pos.iloc[1] == 0.0
    assert pos.iloc[4] == 1.0                    # above cap → fully invested
    assert 0 < pos.iloc[3] < 1                   # monotonic interior


def test_score_to_position_monotonic():
    dates = pd.bdate_range("2023-01-02", periods=100)
    score = pd.Series(np.linspace(-3, 3, 100), index=dates)
    pos = score_to_position(score)
    assert (pos.diff().dropna() >= -1e-12).all()


def test_build_features_keys_and_causality(close):
    feats = build_features(close)
    for key in ["mom_5d", "mom_21d", "mom_63d", "mom_skip1m", "zscore_20d",
                "rsi_14", "bb_pct_20d", "sma_cross_10_50", "macd_hist"]:
        assert key in feats
    # causality: truncating the input must not change earlier feature values
    cut = len(close) - 30
    feats_trunc = build_features(close.iloc[:cut])
    for key in ["mom_63d", "macd_hist", "rsi_14"]:
        pd.testing.assert_series_equal(
            feats[key].iloc[:cut], feats_trunc[key], check_names=False)


def test_build_positions_long_only_bounded(close):
    cfg = get_config()
    pos, diag = build_positions(close, regime_signal_df=None, config=cfg)
    assert (pos >= 0).all()
    assert (pos <= cfg.max_exposure + 1e-12).all()
    assert set(diag["sleeves"]) == {"trend", "meanrev"}
    # positions must not be degenerate all-zero on a live series
    assert (pos.iloc[300:] > 0).any()


def test_build_positions_respects_regime_risk_multiplier(close):
    cfg = get_config()
    # pure risk-multiplier signal (no sleeve tilts): positions must be an
    # exact 0.40x scaling of the base run (overlays are multiplicative)
    sig = pd.DataFrame({
        "label": "crisis",
        "risk_multiplier": 0.40,
    }, index=close.index)
    pos_regime, _ = build_positions(close, regime_signal_df=sig, config=cfg)
    pos_base, _ = build_positions(close, regime_signal_df=None, config=cfg)
    np.testing.assert_allclose(pos_regime.values, (0.40 * pos_base).values,
                               atol=1e-12)


def test_sleeve_weight_frame_renormalizes(close):
    sig = pd.DataFrame({
        "label": "stressed",
        "sleeve_trend": -0.10,
        "sleeve_meanrev": +0.10,
    }, index=close.index[:10])
    w = sleeve_weight_frame(close.index[:10], {"trend": 0.75, "meanrev": 0.25}, sig)
    np.testing.assert_allclose(w.sum(axis=1).values, 1.0, atol=1e-9)
    assert (w["trend"] < 0.75).all()
    assert (w["meanrev"] > 0.25).all()
