"""Tests for the promotion guard in src/model.py."""

import numpy as np
import pandas as pd

from src.model import guard_unseen_promotions


def make_data():
    """4 stores, 1 family, 4 days. Day 4 (after the cutoff) has a promotion in every store:
    in training (days 1-3) at most half the stores ever had one."""
    dates = pd.to_datetime(["2016-01-01", "2016-01-02", "2016-01-03", "2016-01-04"])
    promoted = {  # which stores have a promotion on each day
        dates[0]: [1, 1, 0, 0],
        dates[1]: [1, 0, 0, 0],
        dates[2]: [0, 0, 1, 1],
        dates[3]: [1, 1, 1, 1],
    }
    rows = [
        {"date": d, "store_nbr": store, "family": "BREAD",
         "promo": np.log1p(5.0) if flags[store - 1] else 0.0, "promo_vs_recent": 1.0 if flags[store - 1] else -0.5}
        for d, flags in promoted.items() for store in [1, 2, 3, 4]
    ]
    data = pd.DataFrame(rows)
    data["family"] = data["family"].astype("category")
    return data, dates


def test_unseen_breadth_is_flagged_and_neutralised():
    data, dates = make_data()
    guarded, unseen = guard_unseen_promotions(data, cutoff=dates[2])

    assert list(unseen.index) == [(dates[3], "BREAD")]          # 100% of stores > the training maximum of 50%
    day_4 = guarded["date"] == dates[3]
    assert (guarded.loc[day_4, "promo_vs_recent"] == 0).all()
    # promo becomes its recent level: promo - promo_vs_recent
    expected = data.loc[day_4, "promo"] - data.loc[day_4, "promo_vs_recent"]
    assert np.allclose(guarded.loc[day_4, "promo"], expected)


def test_training_rows_and_seen_situations_are_unchanged():
    data, dates = make_data()
    guarded, unseen = guard_unseen_promotions(data, cutoff=dates[3])  # day 4 is now training data
    assert len(unseen) == 0
    pd.testing.assert_frame_equal(guarded, data)
