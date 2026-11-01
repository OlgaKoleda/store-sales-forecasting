"""Tests for src/lags.py, above all: lag features must never see the forecast period.

The leakage test works like an experiment. Compute the features, then change the sales of the
last 15 days before some date to absurd values, and compute them again. If any feature for that
date changes, it was using sales it isn't allowed to know, and the test fails.
"""

import numpy as np
import pandas as pd

from src.lags import full_calendar, lag_features


def make_wide(n_days=100, seed=0):
    """A small fake sales table: 100 days x 2 series, in the same layout as the real one."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2017-01-01", periods=n_days, name="date")
    columns = pd.MultiIndex.from_tuples([(1, "BREAD"), (2, "MILK")], names=["store_nbr", "family"])
    return pd.DataFrame(rng.poisson(20, size=(n_days, 2)).astype(float), index=dates, columns=columns)


def test_features_ignore_the_last_15_days():
    wide = make_wide()
    target_date = wide.index[-1]
    before = lag_features(wide).loc[target_date]

    changed = wide.copy()
    changed.loc[target_date - pd.Timedelta(days=15):] = 1_000_000  # the 15 days before, and the day itself
    after = lag_features(changed).loc[target_date]

    pd.testing.assert_frame_equal(before, after)


def test_lag_16_is_the_value_16_days_before():
    wide = make_wide()
    features = lag_features(wide)
    date = wide.index[50]
    expected = np.log1p(wide.loc[date - pd.Timedelta(days=16), (1, "BREAD")])
    assert np.isclose(features.loc[(date, 1, "BREAD"), "lag_16"], expected)


def test_missing_day_becomes_nan_not_zero():
    wide = make_wide().drop(pd.Timestamp("2017-01-10"))  # like Christmas: the row is missing
    filled = full_calendar(wide)
    assert pd.Timestamp("2017-01-10") in filled.index
    assert filled.loc["2017-01-10"].isna().all()


def test_features_exist_for_16_future_days():
    wide = make_wide()
    end = wide.index[-1] + pd.Timedelta(days=16)
    features = lag_features(wide, end=end)
    future = features.loc[wide.index[-1] + pd.Timedelta(days=1):]
    assert len(future) == 16 * 2
    assert future[["lag_16", "mean_14", "same_weekday_mean_4"]].notna().all().all()
