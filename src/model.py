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
# Categories, not numbers: store 44 isn't "bigger" than store 2, cluster 13 isn't "more" than cluster 1.
CATEGORICAL = ["store_nbr", "family", "store_type", "cluster"]

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
    """Move date/store/family out of the index into columns; mark the categorical columns as categories."""
    data = table.reset_index()
    for column in CATEGORICAL:
        if column in data.columns:  # store_type and cluster only exist with the "store" extras
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


def guard_unseen_promotions(data, cutoff):
    """Neutralise promotion features where the promotion situation was never seen in training.

    Tree models can't extrapolate: faced with a feature combination outside everything they were
    trained on, they guess. Submission v4 showed how badly (notebook 09): on 16 August 2017 EVERY
    store had a promotion in a few families, more than ever before, and the model forecast 3-30
    times too much.

    The rule, decided from the training data only:
    - breadth = share of stores with a promotion, per (date, family);
    - for each family, the highest breadth in the training period (TRAIN_START up to the cutoff);
    - after the cutoff, a (date, family) above that maximum is "unseen". On those rows the promotion
      features are set to "promotions as usual": promo = its recent level, promo_vs_recent = 0.

    Returns the guarded copy of `data` and the flagged (date, family) breadths.
    """
    cutoff = pd.Timestamp(cutoff)
    promoted = data["promo"] > 0  # promo is log1p(onpromotion): > 0 means at least one item on promotion
    breadth = promoted.groupby([data["date"], data["family"]], observed=True).mean()
    dates = breadth.index.get_level_values("date")

    seen_max = breadth[(dates >= TRAIN_START) & (dates <= cutoff)].groupby(level="family", observed=True).max()
    future = breadth[dates > cutoff]
    limit = seen_max.reindex(future.index.get_level_values("family")).to_numpy()
    unseen = future[future.to_numpy() > limit]

    flagged = pd.MultiIndex.from_frame(data[["date", "family"]]).isin(unseen.index)
    guarded = data.copy()
    guarded.loc[flagged, "promo"] = data.loc[flagged, "promo"] - data.loc[flagged, "promo_vs_recent"]
    guarded.loc[flagged, "promo_vs_recent"] = 0.0
    return guarded, unseen


def backtest_model(data, wide, cutoffs, exclude=None, guard=False):
    """Like src.validation.backtest, for the model: train at each cutoff, score the next 16 days.

    guard=True applies guard_unseen_promotions to the forecast rows (training is unchanged).
    Returns one row per window (labelled by its first forecast day) with the RMSLE, the trees used
    and, with the guard, how many (date, family) pairs it flagged; plus a dict of the forecasts.
    """
    rows, forecasts = {}, {}
    for cutoff in cutoffs:
        dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=HORIZON, name="date")
        model, n_trees = fit(data, cutoff, exclude=exclude)
        forecast_data, unseen = guard_unseen_promotions(data, cutoff) if guard else (data, [])
        predicted = forecast(model, forecast_data, dates)
        actual = wide.loc[dates, predicted.columns]
        label = dates[0].strftime("%Y-%m-%d")
        rows[label] = {"RMSLE": rmsle(actual, predicted), "trees": n_trees}
        if guard:  # count only the flagged (date, family) pairs inside this 16-day window
            rows[label]["flagged"] = int(unseen.index.get_level_values("date").isin(dates).sum())
        forecasts[label] = predicted
    return pd.DataFrame(rows).T.rename_axis("window starts"), forecasts
