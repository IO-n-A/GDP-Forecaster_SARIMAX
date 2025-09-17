"""OECD QNA fetchers for quarterly GDP series.

Purpose
-------
Provide helpers to download quarterly GDP (chain-linked volume) from the OECD
Quarterly National Accounts (QNA) SDMX-JSON CSV endpoint, parse the results, and
standardize them to a tidy two-column DataFrame with quarter-end timestamps.

Key I/O
-------
- fetch_oecd_qna(): returns DataFrame with columns ['date', 'gdp'].
  The DataFrame includes df.attrs['oecd_location_used'] indicating the actual
  OECD location code employed (useful when falling back from EU27_2020 to EA19).
- save_csv(): writes DataFrame to disk, ensuring parent directories exist.

Assumptions
-----------
- Quarterly frequency (freq='Q') with period strings like '2000-Q1' returned by OECD.
- We convert 'YYYY-Qn' to pandas PeriodIndex(freq='Q') then to quarter-end Timestamps.
- Robust parsing: handle both byte and text CSV payloads; tolerate variant column names.
- For EU aggregates, we attempt EU27_2020 first, then gracefully fall back to EA19.

Notes
-----
- Network requests use modest timeouts and raise HTTP errors; callers may want to
  control logging levels or catch exceptions at a higher layer.
- This module is used by the CLI in fetchers/fetch_gdp.py; outputs are commonly saved
  under a data/ directory, but save_csv() is generic and writes wherever requested.
"""
from __future__ import annotations

import logging
from io import BytesIO, StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

OECD_QNA_BASE = "https://stats.oecd.org/sdmx-json/data/QNA"


def _build_qna_csv_url(location: str, subject: str, measure: str, freq: str, start: str) -> str:
    """Build the OECD QNA CSV endpoint URL.

    Parameters
    ----------
    location : str
        OECD location code (e.g., 'USA', 'EU27_2020', 'EA19', 'CHN').
    subject : str
        QNA subject code (e.g., 'B1_GA' for GDP).
    measure : str
        QNA measure code (e.g., 'CLV_I10' for chain-linked volume).
    freq : str
        Frequency code ('Q' for quarterly).
    start : str
        Time filter string passed to OECD as 'time=', e.g., '2000-' meaning 2000 onward.

    Returns
    -------
    str
        Fully formatted CSV URL for the OECD QNA API.
    """
    return (
        f"{OECD_QNA_BASE}/{location}.{subject}.{measure}.{freq}"
        f"?contentType=csv&time={start}"
    )


def _read_oecd_csv_from_response(resp: requests.Response) -> pd.DataFrame:
    """Parse OECD CSV content into a DataFrame with robust fallbacks.

    Parameters
    ----------
    resp : requests.Response
        Response from requests.get(...) for the OECD endpoint.

    Returns
    -------
    pd.DataFrame
        Parsed DataFrame (may be empty if parsing fails).

    Notes
    -----
    - Try parsing bytes with pandas' delimiter sniffing; on failure fall back to text.
    - Returns an empty frame if content is empty or if both parsers fail.
    """
    content = resp.content or b""
    try:
        # Let pandas sniff the delimiter (commas/semicolons vary by endpoint/locale).
        return pd.read_csv(BytesIO(content), sep=None, engine="python")
    except Exception:
        # Fallback to text-based parser if byte parsing fails (e.g., due to codec issues).
        text = resp.text or ""
        if not text:
            return pd.DataFrame()
        try:
            return pd.read_csv(StringIO(text), sep=None, engine="python")
        except Exception:
            return pd.DataFrame()


def _to_quarter_end(time_series: pd.Series) -> pd.Series:
    """Convert strings like 'YYYY-Qn' to quarter-end pandas Timestamps.

    Parameters
    ----------
    time_series : pd.Series
        Series of time labels (e.g., '2000-Q1').

    Returns
    -------
    pd.Series
        Datetime series at the end of each quarter.

    Notes
    -----
    - We normalize 'YYYY-Qn' to 'YYYYQn' for proper PeriodIndex(freq='Q') parsing.
    - pandas PeriodIndex(..., freq='Q').to_timestamp(how='end') yields quarter-end dates.
    """
    s = time_series.astype(str).str.strip()
    # Convert 'YYYY-Qn' -> 'YYYYQn' so PeriodIndex(freq='Q') parses correctly.
    s = s.str.replace("-Q", "Q", regex=False)
    periods = pd.PeriodIndex(s, freq="Q")
    return periods.to_timestamp(how="end")


def fetch_oecd_qna(
    location: str,
    subject: str = "B1_GA",
    measure: str = "CLV_I10",
    freq: str = "Q",
    start: str = "2000-",
) -> pd.DataFrame:
    """Fetch quarterly real GDP (chain-linked volume) from OECD QNA CSV endpoint.

    Parameters
    ----------
    location : str
        OECD location code (e.g., 'USA', 'EU27_2020', 'EA19', 'CHN').
    subject : str, optional
        QNA subject (default 'B1_GA' for GDP).
    measure : str, optional
        QNA measure (default 'CLV_I10' for chain-linked volume index).
    freq : str, optional
        Frequency (default 'Q' for quarterly).
    start : str, optional
        Start filter passed as 'time={start}' (default '2000-' for all dates from 2000 on).

    Returns
    -------
    pd.DataFrame
        Columns: ['date', 'gdp'] with date as quarter-end Timestamp and gdp as float.
        DataFrame attribute 'oecd_location_used' indicates the actual location code used.

    Notes
    -----
    - If 'EU27_2020' is requested but no data is returned, we fall back to 'EA19'.
    - Column names in OECD CSV vary; we detect time/value columns from a candidate set.
    - Deduplication by 'date' keeps the last non-null per time period if duplicates occur.
    - HTTP 204 No Content or empty payloads are treated as 'no data', and we proceed to fallbacks.
    """
    headers = {
        "User-Agent": "GDP-ForecasterSARIMAX/0.1 (+https://example.com)",
        "Accept": "text/csv, */*;q=0.1",
    }

    # EU fallback preference: try EU27_2020 first, then EA19 if needed.
    candidates = [location]
    if location in {"EU27_2020", "EU", "EU27"}:
        candidates = ["EU27_2020", "EA19"]

    last_used = candidates[-1]
    for loc in candidates:
        try:
            url = _build_qna_csv_url(loc, subject, measure, freq, start)
            resp = requests.get(url, headers=headers, timeout=60)
            # Some OECD endpoints return 204 or an empty CSV when no data is available.
            if resp.status_code == 204:
                logger.debug("OECD QNA %s returned 204 No Content", loc)
                continue
            resp.raise_for_status()
            raw_df = _read_oecd_csv_from_response(resp)
            if raw_df is None or raw_df.empty:
                logger.debug("OECD QNA %s returned empty CSV", loc)
                continue

            # Identify time and value columns robustly across variants.
            time_col_candidates = ["TIME", "Time", "time", "TIME_PERIOD", "Period", "PERIOD"]
            val_col_candidates = ["Value", "value", "OBS_VALUE", "Value: double", "OBS_VALUE: double"]

            time_col: Optional[str] = next((c for c in time_col_candidates if c in raw_df.columns), None)
            val_col: Optional[str] = next((c for c in val_col_candidates if c in raw_df.columns), None)

            if not time_col or not val_col:
                logger.debug("Unexpected OECD CSV columns: %s", list(raw_df.columns))
                continue

            # Dimension filters when available (limit to the exact requested series).
            loc_col_candidates = ["LOCATION", "Location", "location"]
            subj_col_candidates = ["SUBJECT", "Subject", "subject"]
            meas_col_candidates = ["MEASURE", "Measure", "measure"]
            freq_col_candidates = ["FREQUENCY", "FREQ", "Frequency", "frequency"]

            def pick(col_candidates):
                return next((c for c in col_candidates if c in raw_df.columns), None)

            loc_col = pick(loc_col_candidates)
            subj_col = pick(subj_col_candidates)
            meas_col = pick(meas_col_candidates)
            freq_col = pick(freq_col_candidates)

            # Compose filter mask gradually, only when columns are present.
            mask = pd.Series(True, index=raw_df.index)
            if loc_col:
                mask &= raw_df[loc_col].astype(str).str.upper().eq(str(loc).upper())
            if subj_col:
                mask &= raw_df[subj_col].astype(str).str.upper().eq(str(subject).upper())
            if meas_col:
                mask &= raw_df[meas_col].astype(str).str.upper().eq(str(measure).upper())
            if freq_col:
                mask &= raw_df[freq_col].astype(str).str.upper().eq(str(freq).upper())

            filtered = raw_df[mask].copy() if mask.any() else raw_df.copy()

            # Normalize to tidy frame with typed columns.
            df = pd.DataFrame(
                {
                    "date": _to_quarter_end(filtered[time_col]),
                    "gdp": pd.to_numeric(filtered[val_col], errors="coerce"),
                }
            )
            df = df.dropna(subset=["date", "gdp"]).sort_values("date").reset_index(drop=True)
            # Deduplicate by date if multiple rows exist for the same period; keep last non-null.
            if not df.empty and df["date"].duplicated().any():
                df = df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

            if df.empty:
                logger.debug("OECD QNA %s produced no valid rows after parsing.", loc)
                continue

            df.attrs["oecd_location_used"] = loc
            return df
        except Exception as e:
            # Keep trying other candidates; emit debug details for diagnostics.
            logger.debug("Error fetching OECD QNA for %s: %s", loc, e)
        last_used = loc

    # If no data found across candidates, return an empty frame with the attribute set for transparency.
    empty = pd.DataFrame(columns=["date", "gdp"])
    empty.attrs["oecd_location_used"] = last_used
    return empty


def save_csv(df: pd.DataFrame, out_path: Path) -> None:
    """Write DataFrame to CSV, ensuring parent directory exists.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save. The index is not written.
    out_path : Path
        Target CSV path. Parents are created if missing.

    Returns
    -------
    None

    Notes
    -----
    - Uses out_path.parent.mkdir(parents=True, exist_ok=True) for safe repeated calls.
    - File naming is determined by the caller (e.g., 'data/gdp_US.csv').
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)