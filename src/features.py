"""Calendar and holiday features: what the model knows about each date in advance.

A model only sees columns. It can't look at a date and "know" that it's a Saturday, the day
after payday, or Christmas Eve. Features turn that knowledge into numbers.

Calendar features are safe to use for forecasting: we know today what weekday it will be
on 31 August, and the holiday file lists future holidays. So no leakage risk here
(unlike features built from past sales, which come in step 6).
"""

import numpy as np
import pandas as pd

from src.data import DATA_DIR


# ---------------------------------------------------------------------------
# Calendar: the same for every store
# ---------------------------------------------------------------------------

def days_since_payday(dates):
    """Days since the last public-sector payday (the 15th or the last day of the month).

    0 on a payday, 1 the day after, and so on. The count restarts at every payday, so it
    runs 0..15 (or 0..14 after the 15th, depending on the month's length).

    Why not just "day of month"? Spending follows the money: notebook 04 shows sales are
    highest right after a payday and lowest just before the next one. This feature measures
    that directly, and it treats the 16th like the 1st: both are "1 day after payday".
    """
    dates = pd.DatetimeIndex(dates)
    day = dates.day.to_numpy()
    # From the 15th on: days since the 15th.
    # On the 1st..14th: the last payday was the last day of the previous month, `day` days ago.
    since = np.where(day >= 15, day - 15, day)
    since = np.where(dates.is_month_end, 0, since)  # the last day of the month is itself a payday
    return pd.Series(since, index=dates)


def calendar_features(dates):
    """One row per date: weekday, day of month, month, and the payday cycle."""
    dates = pd.DatetimeIndex(dates, name="date")
    return pd.DataFrame(
        {
            "day_of_week": dates.day_of_week,     # 0 = Monday ... 6 = Sunday
            "day_of_month": dates.day,
            "month": dates.month,
            "days_since_payday": days_since_payday(dates).to_numpy(),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Holidays: depend on WHERE the store is
# ---------------------------------------------------------------------------

def load_holidays(data_dir=DATA_DIR):
    return pd.read_csv(f"{data_dir}/holidays_events.csv", parse_dates=["date"])


def load_stores(data_dir=DATA_DIR):
    return pd.read_csv(f"{data_dir}/stores.csv")


def holiday_features(dates, stores, holidays):
    """One row per (date, store): which kind of special day it is for that store.

    The holidays file has three traps, all handled here:

    1. transferred=True means the holiday was MOVED to another date. On its original date,
       people work normally. Those rows are skipped; the new date appears as type "Transfer".
    2. locale: a "National" holiday applies to every store, a "Regional" one only to stores in
       that state, a "Local" one only to stores in that city. Matching on locale_name does this.
    3. Several rows can share a date (e.g. a holiday and an event). We only ask "is there at
       least one", so duplicates don't matter.

    Columns (1 = yes, 0 = no):
    - national_day_off:    national holiday, transferred holiday, or bridge day (an extra day off)
    - national_additional: days added around a holiday, mostly around Christmas and New Year
    - national_work_day:   a weekend day declared a working day (to make up for a bridge day)
    - national_event:      events like Mother's Day, Black Friday, football matches (not a day off)
    - earthquake:          the 31 days after the April 2016 earthquake (an "event" of its own kind)
    - regional_holiday:    a holiday in the store's state
    - local_holiday:       a holiday in the store's city
    """
    holidays = holidays[~holidays["transferred"]]  # trap 1

    national = holidays[holidays["locale"] == "National"]
    is_quake = national["description"].str.startswith("Terremoto")
    national_dates = {
        "national_day_off": national.loc[national["type"].isin(["Holiday", "Transfer", "Bridge"]), "date"],
        "national_additional": national.loc[national["type"] == "Additional", "date"],
        "national_work_day": national.loc[national["type"] == "Work Day", "date"],
        "national_event": national.loc[(national["type"] == "Event") & ~is_quake, "date"],
        "earthquake": national.loc[is_quake, "date"],
    }

    # Start from every (date, store) pair, with the store's city and state attached.
    grid = pd.MultiIndex.from_product([pd.DatetimeIndex(dates), stores["store_nbr"]], names=["date", "store_nbr"])
    out = grid.to_frame(index=False).merge(stores[["store_nbr", "city", "state"]], on="store_nbr", how="left")

    for column, these_dates in national_dates.items():
        out[column] = out["date"].isin(these_dates).astype(int)

    # Trap 2: regional and local holidays only count where the store is.
    for column, locale, store_column in [("regional_holiday", "Regional", "state"), ("local_holiday", "Local", "city")]:
        days = holidays[(holidays["locale"] == locale) & (holidays["type"] != "Work Day")]
        pairs = set(zip(days["date"], days["locale_name"]))
        out[column] = [int((d, place) in pairs) for d, place in zip(out["date"], out[store_column])]

    return out.drop(columns=["city", "state"]).set_index(["date", "store_nbr"])
