"""Ledger: append-only idempotency and delayed-execution scoring math."""
import json

import numpy as np
import pandas as pd
import pytest

from ascentagri.ledger import (
    LedgerScore,
    append_entry,
    read_ledger,
    score_ledger,
)


def _entry(date, close, exposure, label="calm_bull"):
    return {"schema": 1, "date": date, "close": close, "exposure": exposure,
            "label": label, "risk_multiplier": 1.0, "series": "TEST"}


def test_append_is_idempotent_per_date(tmp_path):
    p = tmp_path / "ledger.jsonl"
    assert append_entry(_entry("2026-07-01", 100.0, 0.5), p) is True
    assert append_entry(_entry("2026-07-01", 999.0, 0.9), p) is False
    entries = read_ledger(p)
    assert len(entries) == 1
    assert entries[0]["close"] == 100.0     # history never rewritten


def test_read_skips_malformed_lines(tmp_path):
    p = tmp_path / "ledger.jsonl"
    p.write_text(json.dumps(_entry("2026-07-01", 100, 0.5)) + "\n"
                 + "{not json}\n"
                 + json.dumps(_entry("2026-07-02", 101, 0.5)) + "\n")
    assert len(read_ledger(p)) == 2


def test_score_too_young():
    s = score_ledger([_entry("2026-07-01", 100, 0.5),
                      _entry("2026-07-02", 101, 0.5)])
    assert s.n_scored_days == 0
    assert "too young" in s.summary_line()


def test_score_delayed_execution_math():
    """exposure(t0) must earn the t1→t2 return — never the t0→t1 return."""
    entries = [
        _entry("2026-07-01", 100.0, 1.0),   # full exposure decided at t0
        _entry("2026-07-02", 100.0, 0.0),   # flat decided at t1
        _entry("2026-07-03", 110.0, 0.0),   # +10% happens t1→t2
        _entry("2026-07-06", 121.0, 0.0),   # +10% happens t2→t3
    ]
    s = score_ledger(entries)
    # day t2: exposure(t0)=1.0 × +10% = +10%
    assert s.strategy_daily.iloc[0] == pytest.approx(0.10)
    # day t3: exposure(t1)=0.0 × +10% = 0
    assert s.strategy_daily.iloc[1] == pytest.approx(0.0)
    assert s.strategy_return == pytest.approx(0.10)
    assert s.bh_return == pytest.approx(1.21 / 1.0 - 1, rel=1e-9)


def test_score_full_exposure_tracks_buy_and_hold():
    rng = np.random.default_rng(2)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 50)))
    dates = pd.bdate_range("2026-01-01", periods=50)
    entries = [_entry(str(d.date()), float(c), 1.0)
               for d, c in zip(dates, closes)]
    s = score_ledger(entries)
    # always fully exposed → strategy return equals B&H over the scored window
    scored_bh = float((1 + s.bh_daily).prod() - 1)
    assert s.strategy_return == pytest.approx(scored_bh)
    assert s.mean_exposure == 1.0


def test_label_counts():
    entries = [_entry("2026-07-01", 100, 0.5, "crisis"),
               _entry("2026-07-02", 101, 0.5, "crisis"),
               _entry("2026-07-03", 102, 0.5, "calm_bull")]
    s = score_ledger(entries)
    assert s.label_counts == {"crisis": 2, "calm_bull": 1}
