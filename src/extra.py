"""Extra data sources: promotions, store information, oil prices.

The question for every extra column is the same as for lags: WHEN would we know it?
- Promotions: planned in advance, and the test file contains them for the 16 test days. We may
  use the promotion of the forecast day itself.
- Store information (type, cluster): fixed facts about a store, known at any time.
- Oil prices: the file has prices for the test days too, but in real life nobody knows the
  future oil price. So, like sales, we only use prices at least 16 days old.
- Transactions (receipts per store and day): not used. There are none for the test period, so they
  could only be used 16 days old, and they mostly repeat what total sales already say.
"""

import numpy as np
import pandas as pd

from src.data import DATA_DIR
from src.lags import full_calendar, to_long
from src.validation import HORIZON


def promotion_features(train, test, columns, end=None, horizon=HORIZON):
    """Promotion features for every (date, series), as a long table like lag_features.

    `columns`: the series, in the order of the sales table (wide.columns), so that the rows come out
    in the same order as the lag features' rows.

    - promo: log1p(number of items in the family on promotion that day). The forecast day's own
      value, which is allowed because promotions are planned in advance (see the module docstring).
    - promo_vs_recent: promo minus its average over the same 14 days that mean_14 covers
      (16 to 29 days before). The lag features describe sales under the RECENT promotion level;
      this column tells the model whether the forecast day has more or fewer promotions than that.
    """
    rows = pd.concat([train, test])[["date", "store_nbr", "family", "onpromotion"]]
    # Plain numbers and text: train and test have separate category lists, which pandas can't combine.
    rows = rows.astype({"store_nbr": int, "family": str})
    promo = rows.pivot_table(index="date", columns=["store_nbr", "family"], values="onpromotion")
    promo = promo.reindex(columns=columns)
    promo = np.log1p(full_calendar(promo, end))
    recent = promo.shift(horizon).rolling(14, min_periods=1).mean()
    return to_long({"promo": promo, "promo_vs_recent": promo - recent})


def store_features(stores):
    """Store type (A-E) and cluster (groups of similar stores), one row per store."""
    return stores.set_index("store_nbr")[["type", "cluster"]].rename(columns={"type": "store_type"})


def load_oil(data_dir=DATA_DIR):
    """Daily oil price. Weekends and holidays are missing in the file: we carry the last known price forward."""
    oil = pd.read_csv(f"{data_dir}/oil.csv", parse_dates=["date"]).set_index("date")["dcoilwtico"]
    return oil.reindex(pd.date_range(oil.index.min(), oil.index.max(), name="date")).ffill().bfill()


def oil_features(oil, dates, horizon=HORIZON):
    """One row per date: the average oil price of the 14 days ending 16 days before, and its 8-week change.

    Only prices at least 16 days old, like the sales lags: in reality we don't know future oil prices.
    """
    known = oil.shift(horizon)
    mean_14 = known.rolling(14, min_periods=1).mean()
    mean_56 = known.rolling(56, min_periods=1).mean()
    features = pd.DataFrame({"oil_mean_14": mean_14, "oil_change": mean_14 / mean_56 - 1})
    return features.reindex(pd.DatetimeIndex(dates))
