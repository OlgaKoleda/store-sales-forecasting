"""Tests for src/features.py.

Run them from the project root with:  pytest

How tests work: pytest finds every function whose name starts with test_ and runs it.
Each one checks a few cases with `assert`; if an assert is false, pytest reports which one.

Why tiny hand-made examples instead of the real files? They run in a fraction of a second,
they don't need the Kaggle data, and each one checks exactly one rule we decided on.
If someone changes the feature code later and breaks a rule, a test fails immediately.
"""

import pandas as pd

from src.features import calendar_features, days_since_payday, holiday_features


def test_days_since_payday_resets_on_paydays():
    dates = pd.to_datetime(["2017-08-14", "2017-08-15", "2017-08-16", "2017-08-30", "2017-08-31", "2017-09-01"])
    assert days_since_payday(dates).tolist() == [14, 0, 1, 15, 0, 1]


def test_days_since_payday_short_month():
    # February 2017 has 28 days: the 28th is a payday, and 1 March is "1 day after".
    dates = pd.to_datetime(["2017-02-27", "2017-02-28", "2017-03-01"])
    assert days_since_payday(dates).tolist() == [12, 0, 1]


def test_calendar_features_weekday():
    features = calendar_features(pd.to_datetime(["2017-08-14", "2017-08-20"]))  # a Monday and a Sunday
    assert features["day_of_week"].tolist() == [0, 6]


# A tiny stores table and holidays table, in the same layout as the Kaggle files.
STORES = pd.DataFrame({
    "store_nbr": [1, 2],
    "city": ["Quito", "Ambato"],
    "state": ["Pichincha", "Tungurahua"],
})

HOLIDAYS = pd.DataFrame({
    "date": pd.to_datetime(["2017-08-10", "2017-08-11", "2017-08-24"]),
    "type": ["Holiday", "Transfer", "Holiday"],
    "locale": ["National", "National", "Local"],
    "locale_name": ["Ecuador", "Ecuador", "Ambato"],
    "description": ["Primer Grito de Independencia", "Traslado Primer Grito de Independencia", "Fundacion de Ambato"],
    "transferred": [True, False, False],
})


def test_transferred_holiday_counts_on_its_new_date_only():
    features = holiday_features(pd.to_datetime(["2017-08-10", "2017-08-11"]), STORES, HOLIDAYS)
    # 10 August was moved away: a normal working day. 11 August is the day off instead.
    assert features.loc[("2017-08-10", 1), "national_day_off"] == 0
    assert features.loc[("2017-08-11", 1), "national_day_off"] == 1


def test_local_holiday_applies_only_to_its_city():
    features = holiday_features(pd.to_datetime(["2017-08-24"]), STORES, HOLIDAYS)
    assert features.loc[("2017-08-24", 2), "local_holiday"] == 1  # store 2 is in Ambato
    assert features.loc[("2017-08-24", 1), "local_holiday"] == 0  # store 1 is in Quito


def test_one_row_per_date_and_store():
    dates = pd.date_range("2017-08-01", "2017-08-31")
    features = holiday_features(dates, STORES, HOLIDAYS)
    assert len(features) == len(dates) * len(STORES)
    assert not features.index.duplicated().any()
