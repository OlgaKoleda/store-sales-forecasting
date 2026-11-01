"""Tests for src/extra.py: each extra column may only use what is known at forecast time."""

import numpy as np
import pandas as pd

from src.extra import oil_features, promotion_features


def make_rows(n_days=60, seed=0):
    """Fake train rows in the Kaggle layout: 2 series, `n_days` days, random promotions."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2017-01-01", periods=n_days)
    rows = [
        {"date": d, "store_nbr": store, "family": family, "onpromotion": int(rng.integers(0, 5))}
        for d in dates for store, family in [(1, "BREAD"), (2, "MILK")]
    ]
    return pd.DataFrame(rows)


COLUMNS = pd.MultiIndex.from_tuples([(1, "BREAD"), (2, "MILK")], names=["store_nbr", "family"])


def test_promo_uses_the_forecast_days_own_promotion():
    rows = make_rows()
    date = rows["date"].max()
    changed = rows.copy()
    changed.loc[changed["date"] == date, "onpromotion"] = 100  # promotions are planned in advance: allowed to matter
    before = promotion_features(rows, rows.iloc[:0], COLUMNS).loc[date]
    after = promotion_features(changed, rows.iloc[:0], COLUMNS).loc[date]
    assert (after["promo"] == np.float32(np.log1p(100))).all()
    assert not before.equals(after)


def test_promo_ignores_the_15_days_before():
    rows = make_rows()
    date = rows["date"].max()
    changed = rows.copy()
    between = (changed["date"] >= date - pd.Timedelta(days=15)) & (changed["date"] < date)
    changed.loc[between, "onpromotion"] = 100
    before = promotion_features(rows, rows.iloc[:0], COLUMNS).loc[date]
    after = promotion_features(changed, rows.iloc[:0], COLUMNS).loc[date]
    pd.testing.assert_frame_equal(before, after)


def test_oil_uses_only_prices_at_least_16_days_old():
    dates = pd.date_range("2017-01-01", periods=60)
    oil = pd.Series(50.0, index=dates)
    changed = oil.copy()
    changed.iloc[-16:] = 1000.0  # the last 16 days: the forecast day and the 15 before
    pd.testing.assert_frame_equal(oil_features(oil, dates[-1:]), oil_features(changed, dates[-1:]))
