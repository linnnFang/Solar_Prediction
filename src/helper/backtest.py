"""
Walk-forward backtesting for time-series forecasting.

`WalkForward` splits the data into ordered folds where each fold trains on the
past and tests on the following window, never using the future to predict the
past. Window sizes are given as pandas offsets, so lookback and horizon can be
tuned freely (e.g. "30D", "7D", or pd.DateOffset(months=1)).
"""

from dataclasses import dataclass

import pandas as pd


def _as_offset(x):
    """Return a pandas offset from an alias string or DateOffset (None stays None)."""
    if x is None or isinstance(x, pd.DateOffset):
        return x
    return pd.tseries.frequencies.to_offset(x)


@dataclass
class Fold:
    """One train/test split, described by its time boundaries."""
    id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp   # exclusive; equals test_start - gap
    test_start: pd.Timestamp
    test_end: pd.Timestamp    # exclusive

    def split(self, df, time_col="ts"):
        """Return (train_df, test_df) sliced from df by this fold's boundaries."""
        t = df[time_col]
        train = df[(t >= self.train_start) & (t < self.train_end)]
        test = df[(t >= self.test_start) & (t < self.test_end)]
        return train, test


class WalkForward:
    """
    Generate expanding- or rolling-window walk-forward folds.

    horizon  : length of each test window.
    lookback : length of each train window; None means expanding (use all past).
    step     : gap between consecutive test starts; defaults to horizon.
    gap      : spacing left between train end and test start (leakage guard).
    warmup   : history to reserve before the first test window; sets the first
               test start to first_ts + warmup when anchor is not given.
    anchor   : explicit first test-window start; overrides warmup.
    """

    def __init__(self, horizon, lookback=None, step=None, gap=0, warmup=None, anchor=None):
        self.horizon = _as_offset(horizon)
        self.lookback = _as_offset(lookback)
        self.step = _as_offset(step) or self.horizon
        self.gap = _as_offset(gap) if gap else None
        self.warmup = _as_offset(warmup)
        self.anchor = pd.Timestamp(anchor) if anchor is not None else None

    def _first_test_start(self, first_ts):
        """Choose the first test-window start from anchor, warmup, or horizon."""
        if self.anchor is not None:
            return self.anchor
        return first_ts + (self.warmup or self.horizon)

    def folds(self, df, time_col="ts"):
        """Return the list of Fold boundaries for df, in chronological order."""
        first, last = df[time_col].min(), df[time_col].max()
        test_start = self._first_test_start(first)

        folds, i = [], 0
        while test_start < last:
            test_end = test_start + self.horizon
            train_end = test_start - self.gap if self.gap else test_start
            train_start = first if self.lookback is None else train_end - self.lookback
            if first <= train_start < train_end:
                folds.append(Fold(i, train_start, train_end, test_start, test_end))
                i += 1
            test_start = test_start + self.step
        return folds

    def split(self, df, time_col="ts"):
        """Yield (train_df, test_df, fold) for each fold."""
        for fold in self.folds(df, time_col):
            train, test = fold.split(df, time_col)
            yield train, test, fold

    def summary(self, df, time_col="ts"):
        """Return a DataFrame of fold boundaries and row counts, without keeping the slices."""
        rows = [{
            "fold": fold.id,
            "train_start": fold.train_start, "train_end": fold.train_end,
            "test_start": fold.test_start, "test_end": fold.test_end,
            "n_train": len(train), "n_test": len(test),
        } for train, test, fold in self.split(df, time_col)]
        return pd.DataFrame(rows)

    @classmethod
    def gefcom_monthly(cls, warmup_months=13):
        """
        Preset mirroring the GEFCom rolling tasks: expanding window, one calendar
        month per test fold, starting after `warmup_months` of history.
        """
        month = pd.DateOffset(months=1)
        return cls(horizon=month, step=month, warmup=pd.DateOffset(months=warmup_months))
