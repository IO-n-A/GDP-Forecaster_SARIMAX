# -*- coding: utf-8 -*-
"""
China NBS PMI fetchers.

Functions
---------
- load_nbs_pmi_csv(path): Load an exported NBS PMI CSV and normalize to a monthly Series.
- fetch_dbnomics_nbs(url, timeout): Optionally fetch a DB.NOMICS CSV mirror and normalize to a monthly Series.
- normalize_nbs_pmi_df(df): Normalize common NBS/DB.NOMICS dataframe schemas to a monthly Series.

Notes
-----
- Official NBS portal: https://data.stats.gov.cn/english/
- CSV exports may vary in schema; this module attempts to robustly detect date/value columns.
- The output Series is monthly at 'MS' (month start) and named 'PMI'.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def normalize_nbs_pmi_df(df: pd.DataFrame) -> pd.Series:
    """
    Normalize an NBS/DB.NOMICS-like PMI dataframe to a monthly Series.

    The function attempts to identify:
    - date column: one of {'date','period','TIME_PERIOD'}
    - value column: one of {'PMI','value','OBS_VALUE','obs_value'}

    Returns
    -------
    pd.Series
        Monthly PMI series with DatetimeIndex at 'MS' and name 'PMI'.

    Raises
    ------
    ValueError if necessary columns cannot be identified or the frame is empty.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input dataframe is empty or invalid.")

    # Identify date column
    date_candidates = [c for c in df.columns if str(c).lower() in ("date", "period", "time_period")]
    if not date_candidates:
        # fallback: any column containing the token 'period' or 'time'
        date_candidates = [c for c in df.columns if ("period" in str(c).lower()) or ("time" in str(c).lower())]
    if not date_candidates:
        raise ValueError(f"Could not locate a date/period column in columns={list(df.columns)}")
    date_col = date_candidates[0]

    # Identify value column
    value_candidates = [c for c in df.columns if str(c).lower() in ("pmi", "value", "obs_value")]
    if not value_candidates:
        # pick the last non-date column as a fallback
        non_date_cols = [c for c in df.columns if c != date_col]
        if not non_date_cols:
            raise ValueError("No numeric/value column found.")
        value_col = non_date_cols[-1]
    else:
        value_col = value_candidates[0]

    s = (
        df[[date_col, value_col]]
        .rename(columns={date_col: "date", value_col: "PMI"})
        .assign(date=lambda d: pd.to_datetime(d["date"], errors="coerce"))
        .dropna(subset=["date"])
        .set_index("date")["PMI"]
        .sort_index()
    )

    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s.asfreq("MS")
    s.name = "PMI"
    return s


def load_nbs_pmi_csv(path: str | Path, encoding: Optional[str] = None) -> pd.Series:
    """
    Load an NBS PMI CSV export and return a normalized monthly Series.

    Parameters
    ----------
    path : str | Path
        Filesystem path to a CSV exported from the NBS portal (or a curated CSV).
    encoding : Optional[str]
        Override file encoding if needed.

    Returns
    -------
    pd.Series
        Monthly PMI series (name='PMI', freq='MS').
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"NBS PMI CSV not found: {p}")
    df = pd.read_csv(p, encoding=encoding) if encoding else pd.read_csv(p)
    return normalize_nbs_pmi_df(df)


def fetch_dbnomics_nbs(url: str, timeout: int = 30) -> pd.Series:
    """
    Optionally fetch a DB.NOMICS CSV mirror of NBS PMI and return a normalized monthly Series.

    Parameters
    ----------
    url : str
        CSV endpoint on DB.NOMICS for the NBS PMI series.
    timeout : int
        HTTP timeout in seconds.

    Returns
    -------
    pd.Series
        Monthly PMI series (name='PMI', freq='MS').

    Notes
    -----
    Accepts arbitrary CSV schema and attempts to normalize common patterns.
    """
    try:
        df = pd.read_csv(url, storage_options={"timeout": timeout})
    except Exception as e:
        raise RuntimeError(f"Failed to fetch NBS PMI CSV from DB.NOMICS at {url}: {e}") from e
    return normalize_nbs_pmi_df(df)


__all__ = [
    "normalize_nbs_pmi_df",
    "load_nbs_pmi_csv",
    "fetch_dbnomics_nbs",
]