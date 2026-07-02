"""CPCV splitter: combination count, purge/embargo, no train/test overlap."""
import pandas as pd
import pytest

from ascentagri.research.cpcv import CPCVSplitter


def _dates(n=300):
    return pd.bdate_range("2021-01-01", periods=n)


def test_split_count_is_c_n_k():
    sp = CPCVSplitter(n_splits=6, n_test_splits=2)
    assert sp.n_splits_total() == 15
    splits = list(sp.split(_dates()))
    assert len(splits) == 15


def test_each_date_appears_in_expected_test_folds():
    sp = CPCVSplitter(n_splits=6, n_test_splits=2, purge_days=0, embargo_days=0)
    dates = _dates()
    counts = pd.Series(0, index=dates)
    for _, test in sp.split(dates):
        counts.loc[test] += 1
    assert (counts == sp.expected_test_appearances()).all()   # C(5,1)=5


def test_no_train_test_overlap_and_purge_respected():
    sp = CPCVSplitter(n_splits=6, n_test_splits=2, purge_days=5, embargo_days=5)
    dates = _dates()
    pos = {d: i for i, d in enumerate(dates)}
    for train, test in sp.split(dates):
        train_set = set(train)
        assert not train_set & set(test)
        test_idx = sorted(pos[d] for d in test)
        # purge: the 5 dates immediately before each test block are excluded
        for i in test_idx:
            for off in range(1, 6):
                j = i - off
                if j >= 0 and (j + 1) not in test_idx and j not in test_idx:
                    pass  # only block boundaries matter; checked below
        # boundary check on the first test index of each contiguous block
        blocks = []
        start = test_idx[0]
        for a, b in zip(test_idx, test_idx[1:]):
            if b != a + 1:
                blocks.append(start); start = b
        blocks.append(start)
        for b0 in blocks:
            for off in range(1, 6):
                j = b0 - off
                if j >= 0:
                    assert dates[j] not in train_set, "purge violated"


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        CPCVSplitter(n_splits=4, n_test_splits=4)
