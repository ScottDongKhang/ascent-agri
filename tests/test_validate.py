"""Tests for the validation diagnostics."""
import numpy as np
import pandas as pd
import pytest

from ascentagri.roll import build_continuous
from ascentagri.validate import roll_return_report, offset_sensitivity
from tests.test_roll import make_fixture, STEP


def test_roll_return_report_shows_jump_removed():
    clist, P, contracts = make_fixture()
    res = build_continuous(contracts, roll_offset_bd=5, window=4)
    rep = roll_return_report(res)
    assert len(rep) == 3
    for col in ("roll_date", "naive_return", "adjusted_return", "artificial_jump_removed"):
        assert col in rep.columns
    # artificial jump removed at each roll ~ contango step (3%)
    assert np.allclose(rep["artificial_jump_removed"].values, STEP, atol=2e-3)
    # adjusted boundary returns are small (just underlying noise), naive are not
    assert rep["adjusted_return"].abs().max() < 0.02
    assert rep["naive_return"].abs().min() > 0.01


def test_offset_sensitivity_sweeps_all_offsets():
    clist, P, contracts = make_fixture()
    sens = offset_sensitivity(contracts, offsets=(3, 5, 10), window=4)
    assert set(sens["roll_offset_bd"].unique()) == {3, 5, 10}
    for col in ("roll_offset_bd", "expiring", "next", "ratio", "cumulative_factor"):
        assert col in sens.columns
    # for the constant-contango fixture, the ratio is offset-invariant (==1+STEP)
    assert np.allclose(sens["ratio"].values, 1 + STEP, atol=1e-9)
