"""Lag features: what past sales tell us about a future day.

THE RULE: we forecast 16 days ahead. On the day we make the forecast, the most recent sales we
know are 16 days before the LAST forecast day. To keep things simple and safe, every feature here
uses only sales from at least 16 days before the date it describes. That way, the same features can
be computed for every forecast day, and no feature can ever peek into the forecast period.

The price of this simplicity: for the FIRST forecast days, fresher data existed (1, 2, ... days old)
and we don't use it. Notebook 05 measures that cost.

All features are computed on log1p(sales), the scale of the RMSLE metric (see src/metrics.py).
"""

import numpy as np
import pandas as pd

from src.validation import HORIZON


def full_calendar(wide, end=None):
    """Add rows for missing dates, so that "16 rows back" always means "16 days back".

    - Christmas Day is missing from train every year (stores closed). Its row is added as NaN
      ("unknown"), not 0: a closed day says nothing about demand, so averages should skip it,
      and pandas' rolling means do skip NaN.
    - `end` extends the table into the future (e.g. to the last test date) with NaN rows,
      so lag features can be computed for days whose sales we don't know yet.
    """
    end = wide.index.max() if end is None else pd.Timestamp(end)
    return wide.reindex(pd.date_range(wide.index.min(), end, name="date"))


def closed_store_days(wide):
    """True for each (date, series) whose store sold nothing at all that day, in any family.

    A whole store at zero means the store was closed (New Year's Day, a renovation, not open yet),
    not that nobody wanted to buy anything. Like Christmas, such a day says nothing about demand.

    Why this matters (found after submission v2, see notebook 07): stores 18 and 25 were closed for
    weeks from mid-August 2016. Kept as training rows with sales 0, those days taught the model
    "store 25 sells nothing from 22 August" (through the store and calendar columns), and a year
    later it forecast those stores 3-5 times too low. src/dataset.py therefore drops closed days
    as targets; here they also become NaN in the features.
    """
    store_totals = wide.T.groupby(level="store_nbr", observed=True).sum(min_count=1).T  # dates x stores
    closed = store_totals == 0
    # Repeat each store's column once per family, in the same order as wide's columns.
    return pd.DataFrame(
        closed[wide.columns.get_level_values("store_nbr")].to_numpy(),
        index=wide.index,
        columns=wide.columns,
    )


def lag_features(wide, end=None, horizon=HORIZON, mask_closed=True):
    """Lag features for every (date, series), as a long table indexed by (date, store_nbr, family).

    `wide`: sales, rows = dates, columns = (store_nbr, family). See notebook 02.

    Every feature is built from `known` or from shifts of at least `horizon` days:
    known = log sales shifted by 16 days, i.e. on each date, the value from 16 days before.
    A rolling mean of `known` over 14 rows is then the average of days 16..29 before the date.

    mask_closed: days on which a store was closed become NaN ("unknown") before anything is
    computed, so averages skip them instead of counting them as zero demand. False reproduces
    the behaviour before the fix (used in notebook 07 to show what went wrong).
    """
    if mask_closed:
        wide = wide.mask(closed_store_days(wide))
    log_sales = np.log1p(full_calendar(wide, end))
    known = log_sales.shift(horizon)

    def same_weekday_mean(weeks):
        # Same weekday as the date, 3, 4, 5, ... weeks before. 3 weeks = 21 days is the first
        # same-weekday lag that is at least 16 days old. NaNs (e.g. Christmas) are skipped by mean().
        lags = [log_sales.shift(7 * week) for week in range(3, 3 + weeks)]
        return pd.concat(lags).groupby(level=0).mean()

    features = {
        # Single days
        "lag_16": known,                       # the most recent day we're allowed to use
        "lag_21": log_sales.shift(21),         # same weekday, 3 weeks before
        "lag_28": log_sales.shift(28),         # same weekday, 4 weeks before
        "lag_364": log_sales.shift(364),       # same weekday, 52 weeks before ("same time last year")
        # Averages: less noise than single days (notebook 02)
        "mean_7": known.rolling(7, min_periods=1).mean(),
        "mean_14": known.rolling(14, min_periods=1).mean(),
        "mean_28": known.rolling(28, min_periods=1).mean(),
        "mean_56": known.rolling(56, min_periods=1).mean(),
        # Weekday pattern (notebook 03): same weekday, averaged over several weeks
        "same_weekday_mean_4": same_weekday_mean(4),
        "same_weekday_mean_8": same_weekday_mean(8),
        # How many of those 28 days had zero sales: dead series, closed stores, not yet open
        "zero_days_28": (known == 0).astype(float).where(known.notna()).rolling(28, min_periods=1).sum(),
    }
    return to_long(features)


def to_long(features):
    """Turn {name: wide table} into one long table: one row per (date, series), one column per feature.

    Each wide table is flattened row by row (all series of the first date, then the second date, ...),
    and the index is built in exactly the same order. This avoids pandas' stack(), whose behaviour
    changed between pandas 2 and 3.
    """
    first = next(iter(features.values()))
    dates, columns = first.index, first.columns
    index = pd.MultiIndex.from_arrays(
        [
            np.repeat(dates, len(columns)),
            np.tile(columns.get_level_values("store_nbr"), len(dates)),
            np.tile(columns.get_level_values("family"), len(dates)),
        ],
        names=["date", "store_nbr", "family"],
    )
    return pd.DataFrame({name: table.to_numpy(dtype="float32").ravel() for name, table in features.items()}, index=index)
