"""GDP data fetchers for multiple providers.

Purpose
-------
Library utilities used by the GDP fetch CLI to interact with external data providers
and to normalize results for downstream processing.

Providers
---------
- FRED (St. Louis Fed) search and observations endpoints (JSON over HTTPS).
  Authentication: environment variable FRED_API_KEY or an explicit api_key argument.
- OECD QNA access is implemented in a sibling module (oecd_fetchers.py) and used by the CLI.

Key Inputs/Outputs
------------------
- fred_search(): query FRED metadata; returns a DataFrame of series candidates.
- fred_observations(): retrieve observations for a given FRED series_id; returns DataFrame with
  columns ['date', 'value'] (datetime64[ns], float).
- save_csv(): write a DataFrame to disk; creates parent directories safely (mkdir parents).

Assumptions
-----------
- Network requests may fail; callers should handle exceptions where appropriate.
- Quarterly frequency is typical for GDP; fred_search(filter_frequency='Quarterly') is the default.
- This module performs no resampling; values are returned as provided by the API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests


# Base URL for FRED API endpoints (versioned path is handled by route names).
FRED_BASE = "https://api.stlouisfed.org"


@dataclass(frozen=True)
class FredSeries:
    """Descriptor for a FRED series."""

    series_id: str
    title: Optional[str] = None
    notes: Optional[str] = None


# Recommended quarterly real GDP series (subject to change; verify via fred_search).
RECOMMENDED_FRED_SERIES: Dict[str, FredSeries] = {
    # United States: Real Gross Domestic Product, Billions of Chained 2017 Dollars, Quarterly, SAAR
    # https://fred.stlouisfed.org/series/GDPC1
    "US": FredSeries(series_id="GDPC1", title="US Real GDP (Chained 2017 $)"),
    # Euro Area (19): Real GDP (chain-linked volume). Alternative aggregates exist (EA20/EA21, EU27_2020)
    # https://fred.stlouisfed.org/series/CLVMNACSCAB1GQEA19
    "EU_EA19": FredSeries(series_id="CLVMNACSCAB1GQEA19", title="Euro Area 19 Real GDP (Chain-Linked Volume)"),
    # European Union 27 (2020 composition). If unavailable, fall back to EA19 above.
    # https://fred.stlouisfed.org/series/CLVMNACSCAB1GQEU27_2020
    "EU_EU27_2020": FredSeries(
        series_id="CLVMNACSCAB1GQEU27_2020", title="EU27 (2020) Real GDP (Chain-Linked Volume)"
    ),
    # China: Real GDP quarterly series. Multiple candidates exist; search to confirm availability:
    # Example candidates:
    # - CHNGDPNQDSMEI (Gross Domestic Product in constant prices for China, MEI)
    # - NAEXKP02CNQ657S (Constant Price GDP for China, OECD Derived)
    "CN_MEI": FredSeries(series_id="CHNGDPNQDSMEI", title="China GDP (Constant Prices, Quarterly, MEI)"),
}


def _resolve_api_key(explicit: Optional[str]) -> str:
    key = explicit or os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError(
            "FRED API key not found. Set FRED_API_KEY environment variable or pass an explicit api_key."
        )
    return key


def fred_search(
    text: str,
    api_key: Optional[str] = None,
    limit: int = 25,
    offset: int = 0,
    filter_frequency: Optional[str] = "Quarterly",
) -> pd.DataFrame:
    """Search FRED series by free text and optional frequency filter.

    Parameters
    ----------
    text : str
        Free-text query used by FRED's series/search endpoint (matches title/notes/etc.).
    api_key : Optional[str], optional
        FRED API key. If None, reads the FRED_API_KEY env var (resolved in _resolve_api_key).
    limit : int, optional
        Maximum number of results to return (default 25).
    offset : int, optional
        Result offset for pagination (default 0).
    filter_frequency : Optional[str], optional
        If provided, results are filtered to rows whose 'frequency' equals this value
        (case-insensitive), e.g., 'Quarterly'. Set to None to keep all results.

    Returns
    -------
    pd.DataFrame
        DataFrame of candidate series metadata with typical columns:
        id, title, frequency, units, seasonal_adjustment, observation_start, observation_end, popularity.
        Empty DataFrame if no results are found.

    Notes
    -----
    - The FRED response key is 'seriess' (sic). This is normalized into a DataFrame.
    - Filtering by frequency occurs client-side after response parsing.
    """
    key = _resolve_api_key(api_key)
    url = f"{FRED_BASE}/fred/series/search"
    params = {
        "api_key": key,
        "search_text": text,
        "file_type": "json",
        "limit": limit,
        "offset": offset,
        # sort_order can be "desc" (default) and "search_rank" sorts by relevance
        "sort_order": "desc",
        "search_type": "full_text",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()  # Explicitly raise HTTP errors for caller visibility
    payload = r.json()
    items = payload.get("seriess") or []  # FRED returns key 'seriess'
    if not items:
        return pd.DataFrame()

    df = pd.DataFrame(items)
    keep_cols = [
        "id",
        "title",
        "frequency",
        "units",
        "seasonal_adjustment",
        "observation_start",
        "observation_end",
        "popularity",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]
    if filter_frequency and "frequency" in df.columns:
        df = df[df["frequency"].str.lower() == filter_frequency.lower()].reset_index(drop=True)
    return df


def fred_observations(
    series_id: str,
    api_key: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    frequency: Optional[str] = None,
    units: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch observations for a FRED series.

    Parameters
    ----------
    series_id : str
        FRED series identifier (e.g., 'GDPC1').
    api_key : Optional[str], optional
        FRED API key. If None, reads the FRED_API_KEY env var.
    start : Optional[str], optional
        Observation start date (YYYY-MM-DD).
    end : Optional[str], optional
        Observation end date (YYYY-MM-DD).
    frequency : Optional[str], optional
        Optional FRED frequency parameter (e.g., 'q' to aggregate to quarterly on the server).
    units : Optional[str], optional
        Optional FRED units conversion parameter (see FRED API docs).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - date: pandas.Timestamp (datetime64[ns])
        - value: float (NaNs removed when date is missing; non-numeric 'value' coerced to NaN)

    Notes
    -----
    - Requests are performed with a 60s timeout and r.raise_for_status() to surface HTTP errors.
    - '.' values are coerced to NaN per FRED's convention for missing values.
    - No local resampling is applied; values are as returned by the API after coercion.
    """
    key = _resolve_api_key(api_key)
    url = f"{FRED_BASE}/fred/series/observations"
    params = {
        "api_key": key,
        "series_id": series_id,
        "file_type": "json",
    }
    if start:
        params["observation_start"] = start
    if end:
        params["observation_end"] = end
    if frequency:
        params["frequency"] = frequency
    if units:
        params["units"] = units

    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()  # Fail fast on HTTP errors
    payload = r.json()
    obs = payload.get("observations", [])
    df = pd.DataFrame(obs)
    if df.empty:
        return df

    # Convert fields to typed columns
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
    df = df[["date", "value"]].dropna(subset=["date"]).reset_index(drop=True)
    return df


def save_csv(df: pd.DataFrame, out_path: Path) -> None:
    """Write DataFrame to CSV, creating parent directory if needed.

    Parameters
    ----------
    df : pd.DataFrame
        Data to write. Index is not persisted.
    out_path : Path
        Destination CSV path. Parent directories are created if absent.

    Returns
    -------
    None

    Notes
    -----
    - This is a simple wrapper around df.to_csv(out_path, index=False) with mkdir(parents=True).
    - Safe to call repeatedly; directories are not re-created if they already exist.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure folder exists before write
    df.to_csv(out_path, index=False)


def recommended_series(region_key: str) -> FredSeries:
    """Return a recommended FRED series descriptor for a region key.

    Parameters
    ----------
    region_key : str
        One of the known keys (e.g., 'US', 'EU_EA19', 'EU_EU27_2020', 'CN_MEI').

    Returns
    -------
    FredSeries
        Descriptor with series_id and optional metadata.

    Raises
    ------
    KeyError
        If the region_key is unknown to RECOMMENDED_FRED_SERIES.
    """
    if region_key not in RECOMMENDED_FRED_SERIES:
        raise KeyError(f"Unknown region key '{region_key}'. Known keys: {list(RECOMMENDED_FRED_SERIES.keys())}")
    return RECOMMENDED_FRED_SERIES[region_key]
