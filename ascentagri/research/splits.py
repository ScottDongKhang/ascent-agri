"""ascentagri/research/splits.py — ported verbatim from Ascent Capital.

Time-aware train/test splits for leakage-free evaluation.

The split generator yields (train_start, train_end, test_start, test_end)
date tuples. A purge gap between train and test prevents label leakage when
targets are multi-day.
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class SplitWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    fold_id: int

    def __repr__(self):
        return (f"Fold {self.fold_id}: "
                f"train [{self.train_start.date()} → {self.train_end.date()}] | "
                f"test [{self.test_start.date()} → {self.test_end.date()}]")


def walk_forward_splits(
    dates: pd.DatetimeIndex,
    train_days: int = 252,
    test_days: int = 63,
    step_days: int = 21,
    purge_days: int = 5,
    min_train_days: int = 126,
) -> "list[SplitWindow]":
    """Generate walk-forward train/test splits.

    Args:
        dates: Sorted DatetimeIndex of available dates
        train_days: Number of trading days in training window
        test_days: Number of trading days in test window
        step_days: How many days to advance between folds
        purge_days: Gap between train end and test start (prevents label leakage)
        min_train_days: Minimum required training days

    Returns:
        List of SplitWindow objects
    """
    dates = dates.sort_values()
    n = len(dates)
    splits = []
    fold_id = 0

    start_idx = 0
    while True:
        train_start_idx = start_idx
        train_end_idx = train_start_idx + train_days - 1

        if train_end_idx >= n:
            break

        test_start_idx = train_end_idx + purge_days + 1
        test_end_idx = test_start_idx + test_days - 1

        if test_end_idx >= n:
            # Last fold: use remaining data as test
            test_end_idx = n - 1
            if test_start_idx >= n or (test_end_idx - test_start_idx) < 5:
                break

        # Validate
        actual_train = train_end_idx - train_start_idx + 1
        if actual_train < min_train_days:
            start_idx += step_days
            continue

        splits.append(SplitWindow(
            train_start=dates[train_start_idx],
            train_end=dates[train_end_idx],
            test_start=dates[test_start_idx],
            test_end=dates[test_end_idx],
            fold_id=fold_id,
        ))
        fold_id += 1
        start_idx += step_days

        if test_end_idx >= n - 1:
            break

    return splits


def expanding_splits(
    dates: pd.DatetimeIndex,
    initial_train_days: int = 252,
    test_days: int = 63,
    step_days: int = 63,
    purge_days: int = 5,
) -> "list[SplitWindow]":
    """Expanding window: training always starts from the beginning."""
    dates = dates.sort_values()
    n = len(dates)
    splits = []
    fold_id = 0

    train_end_idx = initial_train_days - 1
    while True:
        test_start_idx = train_end_idx + purge_days + 1
        test_end_idx = test_start_idx + test_days - 1

        if test_end_idx >= n:
            test_end_idx = n - 1
            if test_start_idx >= n:
                break

        splits.append(SplitWindow(
            train_start=dates[0],
            train_end=dates[train_end_idx],
            test_start=dates[test_start_idx],
            test_end=dates[test_end_idx],
            fold_id=fold_id,
        ))
        fold_id += 1
        train_end_idx += step_days

        if test_end_idx >= n - 1:
            break

    return splits
