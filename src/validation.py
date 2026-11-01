"""Backtesting: scoring a forecasting rule on several past periods instead of one.

The idea: pretend we are at an earlier date (the "cutoff"), forecast the next 16 days using only
data up to the cutoff, and score against what really happened. Repeat for several cutoffs.
It is cross-validation for time series, with one strict rule: train on the past, test on the future.
"""

import pandas as pd

from src.metrics import rmsle

HORIZON = 16  # the Kaggle test period is 16 days


def make_cutoffs(last_date, n_windows, horizon=HORIZON):
    """Cutoff dates for `n_windows` back-to-back test windows that end on `last_date`.

    The windows do not overlap and leave no gaps, so together they cover the last
    n_windows * horizon days. Returned oldest first.

    Example with horizon=16: the newest window is the last 16 days, its cutoff is
    the day before them; the previous window ends on that cutoff, and so on.
    """
    last_date = pd.Timestamp(last_date)
    return [last_date - pd.Timedelta(days=horizon * k) for k in range(n_windows, 0, -1)]


def split(wide, cutoff, horizon=HORIZON):
    """History = everything up to and including the cutoff; actual = the next `horizon` days."""
    history = wide.loc[:cutoff]
    actual = wide.loc[cutoff + pd.Timedelta(days=1): cutoff + pd.Timedelta(days=horizon)]
    return history, actual


def backtest(wide, rules, cutoffs, horizon=HORIZON):
    """Score every rule on every window.

    rules: {name: function(history, dates)}, see src/baselines.py.
    Returns a table: one row per window (labelled by its first forecast day), one column per rule.
    """
    scores = {}
    for cutoff in cutoffs:
        history, actual = split(wide, cutoff, horizon)
        first_day = actual.index[0].strftime("%Y-%m-%d")
        scores[first_day] = {name: rmsle(actual, rule(history, actual.index)) for name, rule in rules.items()}
    return pd.DataFrame(scores).T.rename_axis("window starts")
