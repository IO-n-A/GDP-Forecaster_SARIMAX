import pandas as pd
import numpy as np

from helpers.temporal import monthly_to_quarterly_avg


def test_monthly_to_quarterly_avg_basic():
    # Jan–Mar 2020, month-start index
    idx = pd.date_range("2020-01-01", periods=3, freq="MS")
    s = pd.Series([1.0, 2.0, 3.0], index=idx, name="PMI")
    out = monthly_to_quarterly_avg(s, name="EXOG")

    # Expect single quarter at 2020-03-31 with mean = 2.0
    assert list(out.index) == [pd.Timestamp("2020-03-31")]
    assert out.shape == (1, 1)
    assert out.columns.tolist() == ["EXOG"]
    assert np.isclose(out.iloc[0, 0], 2.0)


def test_monthly_to_quarterly_avg_two_quarters():
    # Jan–Jun 2020, alternating values for clear quarter means
    idx = pd.date_range("2020-01-01", periods=6, freq="MS")
    vals = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]  # Q1 mean=2.0, Q2 mean=(4+6+8)/3=6.0
    s = pd.Series(vals, index=idx, name="PMI")
    out = monthly_to_quarterly_avg(s, name="EXOG")

    # Expect two quarters: 2020-03-31 and 2020-06-30
    expected_idx = [pd.Timestamp("2020-03-31"), pd.Timestamp("2020-06-30")]
    assert list(out.index) == expected_idx
    assert out.shape == (2, 1)
    assert np.isclose(out.loc[pd.Timestamp("2020-03-31"), "EXOG"], 2.0)
    assert np.isclose(out.loc[pd.Timestamp("2020-06-30"), "EXOG"], 6.0)