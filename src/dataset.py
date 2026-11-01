"""Put all features together into one model-ready table.

One row per (date, store, family). Columns: lag features (src/lags.py), calendar features and
holiday features (src/features.py), and the target, log1p(sales), when it is known.
"""

import numpy as np
import pandas as pd

from src.extra import oil_features, promotion_features, store_features
from src.features import calendar_features, holiday_features
from src.lags import closed_store_days, full_calendar, lag_features


def build_table(train, stores, holidays, end=None, mask_closed_features=True, drop_closed_targets=True,
                extras=(), test=None, oil=None):
    """The full feature table, from the first date of train up to `end`.

    `end` defaults to the last date of train. Pass the last test date to also get rows for the
    test period: their features are complete (all lags are >= 16 days old, and the test period is
    16 days long), but their target is NaN, because those sales are what we must forecast.

    The two switches control how days a store was closed are handled (see notebook 07).
    Both True is the fix; setting them to False reproduces the behaviour before it:
    - mask_closed_features: closed days are "unknown" (NaN) in the lag features;
    - drop_closed_targets: closed days are not used as training targets.

    `extras`: which extra data to add (src/extra.py, notebook 08), any of
    - "promo": promotion features; needs `test` (the test file holds the test days' promotions);
    - "store": store type and cluster;
    - "oil":   oil price features; needs `oil` (see load_oil).
    """
    wide = train.pivot_table(index="date", columns=["store_nbr", "family"], values="sales", observed=True)
    table = lag_features(wide, end=end, mask_closed=mask_closed_features)
    dates = table.index.get_level_values("date")
    stores_of_rows = table.index.get_level_values("store_nbr")

    # Extra data. Promotions come in the same row order as the lag features (same dates, same columns);
    # store and oil features are looked up by the key they depend on.
    if "promo" in extras:
        promo = promotion_features(train, test, wide.columns, end=end)
        for column in promo.columns:
            table[column] = promo[column].to_numpy()
    if "store" in extras:
        info = store_features(stores)
        for column in info.columns:
            table[column] = info[column].reindex(stores_of_rows).to_numpy()
    if "oil" in extras:
        oil_table = oil_features(oil, dates.unique())
        for column in oil_table.columns:
            table[column] = oil_table[column].reindex(dates).to_numpy()

    # Calendar: the same for every series on a date -> look it up by date.
    calendar = calendar_features(dates.unique())
    for column in calendar.columns:
        table[column] = calendar[column].reindex(dates).to_numpy()

    # Holidays: depend on the date and the store -> look them up by (date, store).
    flags = holiday_features(dates.unique(), stores, holidays)
    keys = pd.MultiIndex.from_arrays([dates, stores_of_rows])
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
