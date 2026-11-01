"""Put all features together into one model-ready table.

One row per (date, store, family). Columns: lag features (src/lags.py), calendar features and
holiday features (src/features.py), and the target, log1p(sales), when it is known.
"""

import numpy as np
import pandas as pd

from src.features import calendar_features, holiday_features
from src.lags import closed_store_days, full_calendar, lag_features


def build_table(train, stores, holidays, end=None, mask_closed_features=True, drop_closed_targets=True):
    """The full feature table, from the first date of train up to `end`.

    `end` defaults to the last date of train. Pass the last test date to also get rows for the
    test period: their features are complete (all lags are >= 16 days old, and the test period is
    16 days long), but their target is NaN, because those sales are what we must forecast.

    The two switches control how days a store was closed are handled (see notebook 07).
    Both True is the fix; setting them to False reproduces the behaviour before it:
    - mask_closed_features: closed days are "unknown" (NaN) in the lag features;
    - drop_closed_targets: closed days are not used as training targets.
    """
    wide = train.pivot_table(index="date", columns=["store_nbr", "family"], values="sales", observed=True)
    table = lag_features(wide, end=end, mask_closed=mask_closed_features)
    dates = table.index.get_level_values("date")

    # Calendar: the same for every series on a date -> look it up by date.
    calendar = calendar_features(dates.unique())
    for column in calendar.columns:
        table[column] = calendar[column].reindex(dates).to_numpy()

    # Holidays: depend on the date and the store -> look them up by (date, store).
    flags = holiday_features(dates.unique(), stores, holidays)
    keys = pd.MultiIndex.from_arrays([dates, table.index.get_level_values("store_nbr")])
    for column in flags.columns:
        table[column] = flags[column].reindex(keys).to_numpy()

    # Target: log1p(sales), the scale of the metric. NaN where unknown (Christmas, test period),
    # and also on days the store was closed: a closure can't be forecast from sales history.
    # Kept as zeros, those rows taught the model "this store sells nothing on these dates",
    # and it repeated that a year later: the cause of the bad v2 submission (notebook 07).
    # full_calendar gives the same dates and columns, in the same order, as the rows of `table`.
    if drop_closed_targets:
        wide = wide.mask(closed_store_days(wide))
    table["target"] = np.log1p(full_calendar(wide, end).to_numpy().ravel())
    return table
