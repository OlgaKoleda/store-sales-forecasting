"""The competition metric, written once and imported everywhere.

Why not just call scikit-learn? We could (notebook 02 checks that both agree). Writing it
ourselves makes the two details below visible, and keeps them in one place we control.
"""

import numpy as np


def rmsle(actual, predicted):
    """Root Mean Squared Logarithmic Error: the Kaggle metric for this competition.

    In words: take the log of every actual and predicted value, measure the usual RMSE between them.

    Why the log? It turns errors into *relative* errors. Predicting 15 when the truth is 10 is
    about as bad as predicting 15,000 when the truth is 10,000 (both 50% too high). Without the log,
    the few huge series (like GROCERY I in big stores) would dominate the score completely.

    Two details:
    - log1p(x) = log(1 + x). The "+1" is needed because log(0) is minus infinity, and 31% of
      our sales are zero.
    - Negative predictions are clipped to 0. Sales can't be negative, and log1p of a value
      below -1 is undefined, so the metric would crash. Later models can output small
      negative numbers by accident, so we guard against it here once.

    Works with numbers, NumPy arrays, pandas Series or DataFrames (all values are flattened).
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.clip(np.asarray(predicted, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(predicted) - np.log1p(actual)) ** 2)))
