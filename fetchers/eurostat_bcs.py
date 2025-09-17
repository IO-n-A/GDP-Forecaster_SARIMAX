# -*- coding: utf-8 -*-
"""
Eurostat/ESTAT Business & Consumer Surveys (BCS) fetchers via pandasdmx.

Functions
---------
- fetch_esi_sdmx(geo='EA19', s_adj='SA'): Fetch Economic Sentiment Indicator (ESI), monthly.
- fetch_industry_conf_sdmx(geo='EA19', s_adj='SA'): Fetch Industry Confidence, monthly.
- to_monthly_series(obj, value_name='VALUE', out_name='ESI'): Normalize pandasdmx output to a monthly Series.

Notes
-----
Dataset: EI_BSSI_M_R2 (monthly, revised)
Common codes:
- FREQ = 'M'
- GEO = 'EA19' (Euro Area) or 'EU27_2020' (European Union)
- S_ADJ = 'SA' (seasonally adjusted)
- INDIC:
    * 'BS-ESI-I'  (Economic Sentiment Indicator)
    * 'BS-ICI-I'  (Industry Confidence)
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


DATASET_ID: str = "EI_BSSI_M_R2"
INDIC_ESI: str = "BS-ESI-I"
INDIC_INDUSTRY: str = "BS-ICI-I"


def _ensure_pandasdmx():
    try:
        import pandasdmx as sdmx  # noqa: F401
    except Exception as e:
        raise ImportError("pandasdmx is required for Eurostat BCS fetchers. Install with 'pip install pandasdmx'.") from e


def _estat_request():
    import pandasdmx as sdmx
    return sdmx.Request("ESTAT")


def fetch_esi_sdmx(geo: str = "EA19", s_adj: str = "SA") -> pd.DataFrame:
    """
    Fetch monthly ESI from Eurostat via pandasdmx.

    Parameters
    ----------
    geo : str
        Geographic code, e.g., 'EA19' or 'EU27_2020'.
    s_adj : str
        Seasonal adjustment, typically 'SA'.

    Returns
    -------
    pd.DataFrame
        A normalized dataframe with columns ['TIME_PERIOD','VALUE'].
    """
    _ensure_pandasdmx()
    req = _estat_request()
    params = {"GEO": geo, "INDIC": INDIC_ESI, "S_ADJ": s_adj, "FREQ": "M"}
    resp = req.data(DATASET_ID, params=params)
    return _normalize_estat_to_df(resp)


def fetch_industry_conf_sdmx(geo: str = "EA19", s_adj: str = "SA") -> pd.DataFrame:
    """
    Fetch monthly Industry Confidence from Eurostat via pandasdmx.

    Parameters
    ----------
    geo : str
        Geographic code, e.g., 'EA19' or 'EU27_2020'.
    s_adj : str
        Seasonal adjustment, typically 'SA'.

    Returns
    -------
    pd.DataFrame
        A normalized dataframe with columns ['TIME_PERIOD','VALUE'].
    """
    _ensure_pandasdmx()
    req = _estat_request()
    params = {"GEO": geo, "INDIC": INDIC_INDUSTRY, "S_ADJ": s_adj, "FREQ": "M"}
    resp = req.data(DATASET_ID, params=params)
    return _normalize_estat_to_df(resp)


def _normalize_estat_to_df(resp) -> pd.DataFrame:
    """
    Normalize a pandasdmx Response to a tidy DataFrame with ['TIME_PERIOD','VALUE'].
    """
    obj = resp.to_pandas()
    # pandasdmx may return a Series with MultiIndex or a DataFrame
    if isinstance(obj, pd.Series):
        df = obj.reset_index()
        # The last column is typically the value
        if df.columns[-1] == 0:
            df = df.rename(columns={0: "VALUE"})
        if "TIME_PERIOD" not in df.columns:
            # TIME_PERIOD can be a column or embedded in the index
            time_cols = [c for c in df.columns if str(c).upper() in ("TIME_PERIOD", "TIME", "PERIOD")]
            if time_cols:
                df = df.rename(columns={time_cols[0]: "TIME_PERIOD"})
        cols = ["TIME_PERIOD", "VALUE"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            # try to infer
            raise ValueError(f"Could not locate required columns in Eurostat response: missing={missing}, cols={list(df.columns)}")
        out = df[["TIME_PERIOD", "VALUE"]]
    elif isinstance(obj, pd.DataFrame):
        # Try common naming patterns
        cols = list(obj.columns)
        if "TIME_PERIOD" in cols and ("OBS_VALUE" in cols or "value" in cols):
            val_col = "OBS_VALUE" if "OBS_VALUE" in cols else "value"
            out = obj[["TIME_PERIOD", val_col]].rename(columns={val_col: "VALUE"})
        else:
            # As a fallback, look for a numeric column
            num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(obj[c])]
            if not num_cols:
                raise ValueError(f"Eurostat DataFrame has no obvious numeric value column. columns={cols}")
            val_col = num_cols[0]
            if "TIME_PERIOD" not in cols:
                # Try to reset index / recover time
                try:
                    tmp = obj.reset_index()
                    if "TIME_PERIOD" in tmp.columns:
                        out = tmp[["TIME_PERIOD", val_col]].rename(columns={val_col: "VALUE"})
                    else:
                        raise ValueError("TIME_PERIOD column not found after reset_index.")
                except Exception as e:
                    raise ValueError("Failed to normalize Eurostat DataFrame; TIME_PERIOD missing.") from e
            else:
                out = obj[["TIME_PERIOD", val_col]].rename(columns={val_col: "VALUE"})
    else:
        raise TypeError("Unsupported pandasdmx to_pandas() output type.")

    # Clean types
    out = out.dropna(subset=["TIME_PERIOD", "VALUE"]).copy()
    out["TIME_PERIOD"] = pd.to_datetime(out["TIME_PERIOD"], errors="coerce")
    out["VALUE"] = pd.to_numeric(out["VALUE"], errors="coerce")
    out = out.dropna(subset=["TIME_PERIOD", "VALUE"]).sort_values("TIME_PERIOD").reset_index(drop=True)
    return out


def to_monthly_series(obj, value_name: str = "VALUE", out_name: str = "ESI") -> pd.Series:
    """
    Convert pandasdmx output (Response or DataFrame) to a monthly Series.

    Parameters
    ----------
    obj : pandasdmx Response or pd.DataFrame
        The response from pandasdmx or a normalized DataFrame as produced by _normalize_estat_to_df.
    value_name : str
        Column name containing numeric values (ignored for Response inputs).
    out_name : str
        Name for the output Series.

    Returns
    -------
    pd.Series
        Monthly Series indexed at month start ('MS').
    """
    if hasattr(obj, "to_pandas"):
        df = _normalize_estat_to_df(obj)
    else:
        if not isinstance(obj, pd.DataFrame):
            raise TypeError("to_monthly_series expects a pandasdmx Response or a normalized DataFrame.")
        df = obj
        if value_name not in df.columns:
            # try common alternatives
            value_name = "VALUE" if "VALUE" in df.columns else (df.columns[-1])

    s = (
        df.rename(columns={"TIME_PERIOD": "date", value_name: out_name})
        .assign(date=lambda d: pd.to_datetime(d["date"], errors="coerce"))
        .dropna(subset=["date"])
        .set_index("date")[out_name]
        .sort_index()
    )

    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s.asfreq("MS")
    s.name = out_name
    return s


__all__ = [
    "DATASET_ID",
    "INDIC_ESI",
    "INDIC_INDUSTRY",
    "fetch_esi_sdmx",
    "fetch_industry_conf_sdmx",
    "to_monthly_series",
]