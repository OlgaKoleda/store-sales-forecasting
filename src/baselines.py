"""Simple forecasting rules (baselines), as reusable functions.

Every function has the same shape, so they can be swapped freely:

    forecast = rule(history, dates)

- history: wide table (rows = dates, columns = series) with ONLY the data known when forecasting.
- dates:   the future dates to forecast (a DatetimeIndex).
- returns: a table with one row per date in `dates` and the same columns as `history`.

The same function serves backtesting (history = the past up to a cutoff) and the real Kaggle
forecast (history = all of train, dates = the test days). Same code for both means no gap
between what we validated and what we submit.
"""

import numpy as np
import pandas as pd


def _repeat(values, dates, columns):
    """One value per series, repeated on every forecast date."""
    return pd.DataFrame(np.tile(np.asarray(values, dtype=float), (len(dates), 1)), index=dates, columns=columns)


def last_day(history, dates):
    """Repeat the last known day."""
    return _repeat(history.iloc[-1], dates, history.columns)


def mean_last(history, dates, days=14):
    """Average of the last `days` days, repeated."""
    return _repeat(history.iloc[-days:].mean(), dates, history.columns)


def weekday_mean(history, dates, weeks=4):
    """Each forecast day gets the average of the same weekday over the last `weeks` weeks.

    weeks=1 is "same weekday last week".
    """
    recent = history.iloc[-7 * weeks:]
    by_weekday = recent.groupby(recent.index.day_of_week).mean()  # 7 rows: Monday ... Sunday
    forecast = by_weekday.loc[dates.day_of_week]                  # pick the matching weekday for each date
    forecast.index = dates
    return forecast


def level_and_weekday(history, dates, level_days=14, profile_weeks=8):
    """A smooth level from recent days, adjusted by each series' weekday pattern.

    The idea, combining the two winners of notebook 02:
    - level: how much the series sells on a typical day lately (average of the last `level_days`).
      Many days, so little noise.
    - weekday profile: how much more or less it sells on each weekday than on its average day,
      estimated from a longer period (`profile_weeks`), again to reduce noise.
    - forecast = level, moved up or down by the profile of the forecast day's weekday.

    Everything is computed on log1p(sales), the same scale the RMSLE metric uses:
    - On the log scale, "Saturday sells 50% more" becomes "add a fixed amount", so the
      adjustment is a simple + instead of a ×.
    - The average of log values is the single number with the smallest RMSLE, so averaging on
      this scale optimises exactly what we are scored on.
    At the end, expm1 (the inverse of log1p) converts back to sales.

    One guard: for a series that is almost always zero, level + a negative weekday adjustment
    can fall slightly below 0, and expm1 of a negative number is negative. Sales can't be
    negative, so those forecasts are clipped to 0.
    """
    log_history = np.log1p(history)

    level = log_history.iloc[-level_days:].mean()  # one number per series

    long = log_history.iloc[-7 * profile_weeks:]
    profile = long.groupby(long.index.day_of_week).mean() - long.mean()  # 7 rows, each "above/below average"

    log_forecast = profile.loc[dates.day_of_week].to_numpy() + level.to_numpy()
    return pd.DataFrame(np.expm1(log_forecast), index=dates, columns=history.columns).clip(lower=0)
