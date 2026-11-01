"""Loading the Kaggle Store Sales files.

Every notebook loads data through these functions, so the loading rules live in one place.
"""

from pathlib import Path

import pandas as pd

# Path(__file__) is this file (src/data.py). .parent is src/, .parent again is the project root.
# Building the path this way works no matter which folder you run Python from
# (the project root, notebooks/, ...). A plain "data/train.csv" would only work from the root.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_train(data_dir=DATA_DIR):
    """Read train.csv: one row per (date, store, product family) with the sales of that day.

    Three choices worth knowing:

    - parse_dates: without it, "2013-01-01" is just text. As a real date, pandas can sort it,
      take the weekday, shift by 7 days, and so on. This is the most important line for
      time series work.
    - dtype: store_nbr and family repeat a lot (54 stores, 33 families over 3 million rows).
      "category" stores each distinct value once, which cuts memory use a lot.
      Memory matters here: the file has about 3 million rows.
    - sort_values: time series code (shift, rolling) assumes rows are in time order.
      Sorting once at load time means no later step has to remember to do it.
    """
    df = pd.read_csv(
        Path(data_dir) / "train.csv",
        parse_dates=["date"],
        dtype={"store_nbr": "category", "family": "category"},
    )
    return df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)


def load_test(data_dir=DATA_DIR):
    """Read test.csv: the 16 days we must forecast. Same columns as train, but no sales."""
    df = pd.read_csv(
        Path(data_dir) / "test.csv",
        parse_dates=["date"],
        dtype={"store_nbr": "category", "family": "category"},
    )
    return df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)


# This block runs only when the file is executed directly (`python -m src.data`),
# not when a notebook imports it. It is a quick check that the data is in place.
if __name__ == "__main__":
    train = load_train()
    test = load_test()

    print(f"train: {len(train):,} rows, {train['date'].min():%Y-%m-%d} to {train['date'].max():%Y-%m-%d}")
    print(f"test:  {len(test):,} rows, {test['date'].min():%Y-%m-%d} to {test['date'].max():%Y-%m-%d}")
    print(f"stores: {train['store_nbr'].nunique()}, product families: {train['family'].nunique()}")
    print(f"memory used by train: {train.memory_usage(deep=True).sum() / 1e6:.0f} MB")
