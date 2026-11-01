"""Loading the Kaggle Store Sales files.

Every notebook loads data through these functions, so the loading rules live in one place.
"""

from pathlib import Path

import pandas as pd

# Path(__file__) is this file (src/data.py). .parent is src/, .parent again is the project root.
# Building the path this way works no matter which folder you run Python from
# (the project root, notebooks/, ...). A plain "data/train.csv" would only work from the root.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _read(filename, data_dir):
    """Shared reading rules for train.csv and test.csv (they have the same layout).

    Four choices worth knowing:

    - parse_dates: without it, "2013-01-01" is just text. As a real date, pandas can sort it,
      take the weekday, shift by 7 days, and so on. This is the most important line for
      time series work.
    - category: store_nbr and family repeat a lot (54 stores, 33 families over 3 million rows).
      "category" stores each distinct value once, which cuts memory use a lot.
    - store_nbr is read as a whole number FIRST, and only then made a category. This is
      deliberate: if we ask read_csv for a category directly, pandas 2 makes the categories
      text ("44") while pandas 3 makes them numbers (44), so `store_nbr == 44` works in one
      version and silently matches nothing in the other. Being explicit removes the guesswork.
      Lesson: don't rely on a library's default types; state the type you need.
    - sort_values: time series code (shift, rolling) assumes rows are in time order.
      Sorting once at load time means no later step has to remember to do it.
    """
    df = pd.read_csv(
        Path(data_dir) / filename,
        parse_dates=["date"],
        dtype={"store_nbr": "int64", "family": "category"},
    )
    df["store_nbr"] = df["store_nbr"].astype("category")
    return df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)


def load_train(data_dir=DATA_DIR):
    """Read train.csv: one row per (date, store, product family) with the sales of that day."""
    return _read("train.csv", data_dir)


def load_test(data_dir=DATA_DIR):
    """Read test.csv: the 16 days we must forecast. Same columns as train, but no sales."""
    return _read("test.csv", data_dir)


# This block runs only when the file is executed directly (`python -m src.data`),
# not when a notebook imports it. It is a quick check that the data is in place.
if __name__ == "__main__":
    train = load_train()
    test = load_test()

    print(f"train: {len(train):,} rows, {train['date'].min():%Y-%m-%d} to {train['date'].max():%Y-%m-%d}")
    print(f"test:  {len(test):,} rows, {test['date'].min():%Y-%m-%d} to {test['date'].max():%Y-%m-%d}")
    print(f"stores: {train['store_nbr'].nunique()}, product families: {train['family'].nunique()}")
    print(f"memory used by train: {train.memory_usage(deep=True).sum() / 1e6:.0f} MB")
