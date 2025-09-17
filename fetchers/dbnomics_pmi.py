# -*- coding: utf-8 -*-
"""
DB.NOMICS fetchers for PMI-like indicators.

Functions
---------
- fetch_ism_pmi_csv(url, timeout): Fetch ISM Manufacturing PMI from DB.NOMICS CSV endpoint.
- normalize_ism_pmi_df(df): Normalize a raw DB.NOMICS dataframe into a monthly pandas Series.
- ism_pmi_monthly(url, timeout): Convenience wrapper returning a monthly Series named 'PMI'.

Notes
-----
- Endpoint example for ISM Manufacturing PMI (headline):
  https://api.db.nomics.world/series/ISM/pmi/pm.csv
- The schema may vary slightly; this module attempts to normalize common variants.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

DBNOMICS_ISM_PMI_URL: str = "https://api.db.nomics.world/series/ISM/pmi/pm.csv"


def fetch_ism_pmi_csv(url: Optional[str] = None, timeout: int = 30) -> pd.DataFrame:
    """
    Fetch ISM Manufacturing PMI CSV from DB.NOMICS.

    Parameters
    ----------
    url : Optional[str]
        CSV endpoint URL. Defaults to DBNOMICS_ISM_PMI_URL.
    timeout : int
        Network timeout in seconds for the HTTP request.

    Returns
    -------
    pd.DataFrame
        Raw dataframe as returned by pandas.read_csv.

    Raises
    ------
    RuntimeError
        If the HTTP request or CSV parsing fails.
    """
    u = url or DBNOMICS_ISM_PMI_URL
    try:
        # pandas handles HTTP(S) directly; pass timeout via storage_options
        df = pd.read_csv(u, storage_options={"timeout": timeout})
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to fetch ISM PMI from DB.NOMICS at {u}: {e}") from e


def normalize_ism_pmi_df(df: pd.DataFrame) -> pd.Series:
    """
    Normalize a raw DB.NOMICS PMI dataframe into a monthly Series.

    Expected flexible schemas
    -------------------------
    - Columns may include 'period' or 'TIME_PERIOD' for dates.
    - Value columns may be named 'pm', 'value', 'OBS_VALUE', or similar.

    Returns
    -------
    pd.Series
        Monthly series with DatetimeIndex (month start), name 'PMI', sorted ascending and
        set to monthly frequency.

    Raises
    ------
    ValueError
        If date or value columns cannot be identified.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input dataframe is empty or invalid.")

    # Identify date column
    date_candidates = [c for c in df.columns if str(c).lower() in ("period", "time_period", "date")]
    if not date_candidates:
        # try common index names if 'period' provided as unnamed
        date_candidates = [c for c in df.columns if "period" in str(c).lower()]
    if not date_candidates:
        raise ValueError(f"Could not locate a date/period column in columns={list(df.columns)}")
    date_col = date_candidates[0]

    # Identify value column
    val_candidates = []
    for c in df.columns:
        lc = str(c).lower()
        if lc in ("pm", "value", "obs_value", "pmi", "obsvalue"):
            val_candidates.append(c)
    if not val_candidates:
        # Fallback: pick the last non-date column heuristically
        non_date_cols = [c for c in df.columns if c != date_col]
        if not non_date_cols:
            raise ValueError("No value column found.")
        value_col = non_date_cols[-1]
    else:
        value_col = val_candidates[0]

    out = (
        df[[date_col, value_col]]
        .rename(columns={date_col: "date", value_col: "PMI"})
        .assign(date=lambda d: pd.to_datetime(d["date"], errors="coerce"))
        .dropna(subset=["date"])
        .sort_values("date")
        .set_index("date")["PMI"]
    )

    # Ensure numeric, coerce errors
    out = pd.to_numeric(out, errors="coerce").dropna().sort_index()

    # Normalize to monthly start frequency; missing months remain NaN unless we asfreq
    out = out.asfreq("MS")
    out.name = "PMI"
    return out


def ism_pmi_monthly(url: Optional[str] = None, timeout: int = 30) -> pd.Series:
    """
    Convenience wrapper: fetch and normalize ISM PMI to a monthly Series.

    Parameters
    ----------
    url : Optional[str]
        DB.NOMICS CSV endpoint (defaults to DBNOMICS_ISM_PMI_URL).
    timeout : int
        Network timeout in seconds.

    Returns
    -------
    pd.Series
        Monthly PMI series with frequency 'MS'.
    """
    raw = fetch_ism_pmi_csv(url=url, timeout=timeout)
    return normalize_ism_pmi_df(raw)


__all__ = [
    "DBNOMICS_ISM_PMI_URL",
    "fetch_ism_pmi_csv",
    "normalize_ism_pmi_df",
    "ism_pmi_monthly",
]