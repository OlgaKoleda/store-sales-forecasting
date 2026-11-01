"""The LightGBM model: training, forecasting and backtesting on the feature table (src/dataset.py).

One model learns from all 1,782 series at once. Each row of the table is one (date, store, family);
the model learns how the features of a row relate to its target, log1p(sales).
Training on the log scale with the ordinary squared error is exactly what RMSLE rewards.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.metrics import rmsle
from src.validation import HORIZON

# Columns the model may use. "target" is what it predicts; "date" is only used to split by time.
CATEGORICAL = ["store_nbr", "family"]

PARAMS = {
    "objective": "regression",  # squared error, on log1p(sales) -> the RMSLE
    "learning_rate": 0.05,      # how big each tree's correction is: smaller = more trees, usually better
    "num_leaves": 63,           # how complex each tree can be
    "min_data_in_leaf": 100,    # every leaf must describe at least 100 rows: protects against noise
    "feature_fraction": 0.8,    # each tree sees a random 80% of the features...
    "bagging_fraction": 0.8,    # ...and a random 80% of the rows: makes trees more diverse
    "bagging_freq": 1,
    "seed": 0,                  # same randomness every run, so results are reproducible
    "verbose": -1,
}

TRAIN_START = "2015-01-01"  # earlier rows have incomplete features (no lag_364, few months of history)
MAX_TREES = 3000
EARLY_STOPPING = 100


def prepare(table):
    """Move date/store/family out of the index into columns; mark store and family as categories."""
    data = table.reset_index()
    for column in CATEGORICAL:
        data[column] = data[column].astype("category")
    return data


def feature_columns(data):
    return [c for c in data.columns if c not in ("date", "target")]


def rows_between(data, start, end):
    """Rows with start <= date <= end and a known target."""
    return data[(data["date"] >= start) & (data["date"] <= end) & data["target"].notna()]


def fit(data, cutoff, exclude=None):
    """Train a model on everything known at `cutoff`, choosing the number of trees safely.

    How many trees? Too few underfits, too many overfits. We can't pick the number by looking at the
    forecast period (that would be tuning on the test). Instead:
      1. hold out the last 16 days BEFORE the cutoff, as an inner validation period;
      2. train on the rest, adding trees until the inner validation stops improving (early stopping);
      3. retrain on ALL data up to the cutoff with that number of trees, so the newest
         16 days (often the most informative) are used too.

    `exclude`: optional (start, end) dates to leave out of training, e.g. the earthquake weeks.
    Returns the trained model and the number of trees chosen.
    """
    cutoff = pd.Timestamp(cutoff)
    features = feature_columns(data)
    train = rows_between(data, TRAIN_START, cutoff)
    if exclude is not None:
        train = train[~train["date"].between(*exclude)]

    inner_cutoff = cutoff - pd.Timedelta(days=HORIZON)
    inner_train = train[train["date"] <= inner_cutoff]
    inner_valid = train[train["date"] > inner_cutoff]
    probe = lgb.train(
        PARAMS,
        lgb.Dataset(inner_train[features], inner_train["target"]),
        num_boost_round=MAX_TREES,
        valid_sets=[lgb.Dataset(inner_valid[features], inner_valid["target"])],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )
    n_trees = probe.best_iteration

    model = lgb.train(PARAMS, lgb.Dataset(train[features], train["target"]), num_boost_round=n_trees)
    return model, n_trees


def forecast(model, data, dates):
    """Forecast sales (not logs) for the given dates, as a wide table: dates x (store_nbr, family)."""
    rows = data[data["date"].isin(dates)]
    predicted = np.expm1(model.predict(rows[feature_columns(data)])).clip(min=0)
    long = pd.Series(predicted, index=pd.MultiIndex.from_frame(rows[["date", "store_nbr", "family"]]))
    return long.unstack(["store_nbr", "family"])


def backtest_model(data, wide, cutoffs, exclude=None):
    """Like src.validation.backtest, for the model: train at each cutoff, score the next 16 days.

    Returns one row per window (labelled by its first forecast day) with the RMSLE and the trees used,
    plus a dict of the forecasts, for further analysis.
    """
    rows, forecasts = {}, {}
    for cutoff in cutoffs:
        dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=HORIZON, name="date")
        model, n_trees = fit(data, cutoff, exclude=exclude)
        predicted = forecast(model, data, dates)
        actual = wide.loc[dates, predicted.columns]
        label = dates[0].strftime("%Y-%m-%d")
        rows[label] = {"RMSLE": rmsle(actual, predicted), "trees": n_trees}
        forecasts[label] = predicted
    return pd.DataFrame(rows).T.rename_axis("window starts"), forecasts
