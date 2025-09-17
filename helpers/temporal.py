# -*- coding: utf-8 -*-
"""
Temporal utilities for frequency alignment and aggregation.

Functions
---------
- monthly_to_quarterly_avg(series, name): Aggregate monthly series to quarterly
  by within-quarter arithmetic mean, indexed at the quarter end. This preserves
  time-causality (uses only months within each quarter).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def _ensure_datetime_index(s: pd.Series) -> pd.Series:
    """
    Ensure a DatetimeIndex for the input series.

    - If PeriodIndex, convert to Timestamp index. For monthly data, align at month end.
    - Leaves DatetimeIndex unchanged.
    """
    if isinstance(s.index, pd.PeriodIndex):
        # If monthly, align to end-of-month; otherwise generic to_timestamp.
        try:
            is_monthly = (s.index.freqstr or "").upper().startswith("M")
        except Exception:
            is_monthly = False
        if is_monthly:
            s = s.copy()
            s.index = s.index.to_timestamp(how="end")
        else:
            s = s.copy()
            s.index = s.index.to_timestamp()
    elif not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("monthly_to_quarterly_avg expects a Series with DatetimeIndex or PeriodIndex.")
    return s


def monthly_to_quarterly_avg(series: pd.Series, name: Optional[str] = None) -> pd.DataFrame:
    """
    Aggregate a monthly time series to quarterly by within-quarter mean.

    Parameters
    ----------
    series : pd.Series
        Monthly series with DatetimeIndex or PeriodIndex.
    name : Optional[str]
        Column name for the returned DataFrame. Defaults to series.name or 'value'.

    Returns
    -------
    pd.DataFrame
        Single-column DataFrame at quarterly frequency ('QE' quarter-end index)
        with values equal to the arithmetic mean of the constituent months.

    Notes
    -----
    - Time-causality: For quarter Q, only months within Q are used (no look-ahead).
    - The output index is at quarter end (e.g., 2001-03-31 for 2001Q1).
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")

    s = _ensure_datetime_index(series.dropna())

    # Ensure monthly granularity; if the index is irregular within months, resample to month-end mean
    # (typical PMI/Govt series are already one observation per month).
    monthly = s.resample("ME").mean()

    # Aggregate to quarter-end by arithmetic mean
    quarterly = monthly.resample("QE").mean()

    col_name = name if name is not None else (series.name if series.name is not None else "value")
    return quarterly.to_frame(col_name)