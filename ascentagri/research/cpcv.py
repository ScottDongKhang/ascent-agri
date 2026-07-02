"""ascentagri/research/cpcv.py — ported verbatim from Ascent Capital.

Combinatorial Purged Cross-Validation (CPCV) splitter.

From de Prado, "Advances in Financial Machine Learning", Chapter 12.

Default parameters:
    n_splits = 6, n_test_splits = 2
    → C(6, 2) = 15 train/test splits
    → each date appears in C(5, 1) = 5 test folds
    → prediction for each date = mean across those 5 models

    purge_days  = 5  (covers mom_5d feature lookback — no leakage)
    embargo_days = 5  (additional buffer post test fold)
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Iterator

import numpy as np
import pandas as pd


class CPCVSplitter:
    """Combinatorial Purged Cross-Validation splitter.

    Produces C(n_splits, n_test_splits) train/test date index pairs.
    Purge gap removes training observations whose forward returns overlap
    with the test period. Embargo gap prevents leakage from slow features.
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        purge_days: int = 5,
        embargo_days: int = 5,
    ) -> None:
        if n_test_splits >= n_splits:
            raise ValueError(
                f"n_test_splits ({n_test_splits}) must be < n_splits ({n_splits})"
            )
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self._last_splits: "list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]" = []

    # ── public API ────────────────────────────────────────────────────────────

    def split(
        self, dates: pd.DatetimeIndex
    ) -> "Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]":
        """Yield (train_dates, test_dates) for each of C(N, k) combinations."""
        dates = pd.DatetimeIndex(sorted(set(dates)))
        n = len(dates)
        fold_indices = self._make_fold_indices(n)
        self._last_splits = []

        for test_fold_combo in combinations(range(self.n_splits), self.n_test_splits):
            # Test: union of selected fold index arrays
            test_idx = np.concatenate([fold_indices[f] for f in test_fold_combo])
            test_idx = np.sort(test_idx)
            test_dates = dates[test_idx]

            # Build excluded set: test indices + purge zone before each test fold
            # + embargo zone after each test fold
            excluded: "set[int]" = set(test_idx)

            for fold_id in test_fold_combo:
                fold = fold_indices[fold_id]
                fold_start = fold[0]
                fold_end = fold[-1]

                # Purge: remove train dates immediately before this test fold
                for offset in range(1, self.purge_days + 1):
                    idx = fold_start - offset
                    if idx >= 0:
                        excluded.add(idx)

                # Embargo: remove train dates immediately after this test fold
                for offset in range(1, self.embargo_days + 1):
                    idx = fold_end + offset
                    if idx < n:
                        excluded.add(idx)

            train_idx = np.array(
                [i for i in range(n) if i not in excluded], dtype=int
            )
            train_dates = dates[train_idx] if len(train_idx) > 0 else pd.DatetimeIndex([])

            self._last_splits.append((train_dates, test_dates))
            yield train_dates, test_dates

    def n_splits_total(self) -> int:
        """Total number of train/test splits = C(n_splits, n_test_splits)."""
        return math.comb(self.n_splits, self.n_test_splits)

    def expected_test_appearances(self) -> int:
        """Each date appears in C(N-1, k-1) test folds."""
        return math.comb(self.n_splits - 1, self.n_test_splits - 1)

    # ── private helpers ───────────────────────────────────────────────────────

    def _make_fold_indices(self, n: int) -> "list[np.ndarray]":
        """Divide [0, n) into n_splits approximately equal contiguous folds."""
        indices = np.arange(n)
        return [arr for arr in np.array_split(indices, self.n_splits) if len(arr) > 0]
