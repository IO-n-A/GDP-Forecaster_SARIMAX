import pandas as pd
import numpy as np
from pathlib import Path

import pytest


def test_exog_alignment_no_lookahead():
    # Import lazily to keep import path stable under pytest
    from helpers.temporal import monthly_to_quarterly_avg

    # Monthly series Jan..Jun 2020; Q1 mean=(10+20+30)/3, Q2 mean=5
    idx = pd.date_range("2020-01-01", periods=6, freq="MS")
    s = pd.Series([10.0, 20.0, 30.0, 5.0, 5.0, 5.0], index=idx, name="PMI")

    exog_q = monthly_to_quarterly_avg(s, name="EXOG")

    # GDP quarterly timestamps at quarter end
    gdp_idx = pd.to_datetime(["2020-03-31", "2020-06-30"])
    ex_aligned = exog_q.reindex(gdp_idx)

    assert ex_aligned.index.equals(pd.DatetimeIndex(gdp_idx))
    assert ex_aligned.loc[pd.Timestamp("2020-03-31"), "EXOG"] == pytest.approx((10 + 20 + 30) / 3.0, rel=1e-9)
    assert ex_aligned.loc[pd.Timestamp("2020-06-30"), "EXOG"] == pytest.approx(5.0, rel=1e-9)


def test_quarter_end_index_from_monthly():
    from helpers.temporal import monthly_to_quarterly_avg

    # A single monthly point in Feb 2021 belongs to 2021Q1 -> quarter end 2021-03-31
    idx = pd.date_range("2021-02-01", periods=1, freq="MS")
    s = pd.Series([42.0], index=idx, name="X")

    q = monthly_to_quarterly_avg(s, name="X")
    assert q.index[0] == pd.Timestamp("2021-03-31")