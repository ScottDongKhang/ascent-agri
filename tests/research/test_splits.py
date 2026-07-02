"""Walk-forward splits: ordering, purge gap, minimum train length."""
import pandas as pd

from ascentagri.research.splits import expanding_splits, walk_forward_splits


def _dates(n=500):
    return pd.bdate_range("2020-01-01", periods=n)


def test_windows_ordered_and_purged():
    dates = _dates()
    splits = walk_forward_splits(dates, train_days=252, test_days=63,
                                 step_days=63, purge_days=5)
    assert len(splits) > 0
    for s in splits:
        assert s.train_start < s.train_end < s.test_start <= s.test_end
        # purge gap: at least purge_days trading days between train end and test start
        gap = dates.get_loc(s.test_start) - dates.get_loc(s.train_end)
        assert gap == 6  # purge_days + 1


def test_train_window_fixed_length():
    dates = _dates()
    splits = walk_forward_splits(dates, train_days=252, test_days=63, step_days=63)
    for s in splits:
        n_train = dates.get_loc(s.train_end) - dates.get_loc(s.train_start) + 1
        assert n_train == 252


def test_test_windows_tile_without_overlap():
    dates = _dates(700)
    splits = walk_forward_splits(dates, train_days=252, test_days=63,
                                 step_days=63, purge_days=5)
    for a, b in zip(splits, splits[1:]):
        assert b.test_start > a.test_end


def test_too_short_series_yields_no_splits():
    dates = _dates(100)
    splits = walk_forward_splits(dates, train_days=252, test_days=63)
    assert splits == []


def test_expanding_splits_grow_train():
    dates = _dates()
    splits = expanding_splits(dates, initial_train_days=252, test_days=63,
                              step_days=63)
    assert len(splits) >= 2
    assert all(s.train_start == dates[0] for s in splits)
    ends = [s.train_end for s in splits]
    assert ends == sorted(ends)
