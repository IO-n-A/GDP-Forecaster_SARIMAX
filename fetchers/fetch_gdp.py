"""Fetch up-to-date quarterly GDP series for US, EU, and China.

Purpose
-------
Pull quarterly real GDP series from either:
- OECD QNA SDMX-JSON CSV endpoint (default; no API key required), or
- FRED (St. Louis Fed) API (requires API key), with simple fallbacks when the primary
  series has no observations.

Outputs
-------
- One CSV per region under --out-dir (default: data/), named gdp_{TAG}.csv
  where TAG ∈ {US, EU27_2020, EA19, CN}.
- Optional stacked CSV gdp_stacked.csv containing columns ['date','region','gdp'].

CLI overview
------------
--regions           Select regions (supports aliases and ALL).
--source            Choose data provider: OECD (default) or FRED.
--fred-api-key      API key for FRED; also read from env FRED_API_KEY or config/api_keys.env.
--start/--end       Date bounds (FRED only).
--out-dir           Output directory for CSVs (resolved relative to this script if not absolute).
--stacked           Also write a stacked CSV combining selected regions.
--log-level         Logging verbosity.

Assumptions
-----------
- Quarterly frequency series with dates standardized to period ends.
- Output helpers ensure parents are created (safe to run repeatedly).
- File naming is stable to ease downstream automation.

Examples
--------
  # Fetch all regions and write individual CSVs + stacked CSV (OECD default)
  ./time-s/bin/python GDP-ForecasterSARIMAX/fetch_gdp.py --regions ALL --stacked

  # Fetch specific set and place outputs in a custom directory
  ./time-s/bin/python GDP-ForecasterSARIMAX/fetch_gdp.py --regions US EU CN --out-dir GDP-ForecasterSARIMAX/data
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Allow imports when running as a script from a hyphenated parent folder:
# ensure the parent directory (containing the 'fetchers' package) is on sys.path.
_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent
for p in (str(_PARENT_DIR), str(_THIS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)  # Make 'fetchers.*' importable when executed as a script.


def _load_env_from_config() -> None:
    """Load KEY=VALUE pairs from config/api_keys.env into os.environ if present."""
    base_root = _THIS_DIR.parent
    cfg = base_root / "config" / "api_keys.env"
    if cfg.is_file():
        for line in cfg.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


from fetchers.gdp_fetchers import (  # type: ignore
    RECOMMENDED_FRED_SERIES,
    FredSeries,
    fred_observations,
    recommended_series,
    save_csv,
)
from fetchers.oecd_fetchers import fetch_oecd_qna, save_csv as save_csv_oecd  # type: ignore


logger = logging.getLogger(__name__)

# Fallback series ids by region key if the primary series has no observations
FALLBACK_FRED_SERIES_ID: Dict[str, str] = {
    "EU_EU27_2020": "CLVMNACSCAB1GQEA19",  # fallback to EA19
    "CN_MEI": "NAEXKP02CNQ657S",  # OECD Constant Price GDP for China, quarterly (if available)
}


# Region aliases for user-friendly CLI input
ALIASES: Dict[str, str] = {
    "EU": "EU_EU27_2020",
    "CN": "CN_MEI",
}


def normalize_regions(regions: List[str]) -> List[str]:
    """Normalize free-form region tokens to internal FRED region keys.

    Parameters
    ----------
    regions : List[str]
        Tokens like ['US', 'EU', 'CN'] or ['ALL'].

    Returns
    -------
    List[str]
        Normalized keys such as ['US', 'EU_EU27_2020', 'CN_MEI'].
        If 'ALL' is present, returns the full default set: ['US','EU_EU27_2020','CN_MEI'].
    """
    if "ALL" in regions:
        return ["US", "EU_EU27_2020", "CN_MEI"]
    norm = []
    for r in regions:
        r = r.strip().upper()
        if r in ALIASES:
            norm.append(ALIASES[r])
        else:
            norm.append(r)
    return norm


def normalize_oecd_regions(regions: List[str]) -> List[str]:
    """Normalize region tokens for OECD QNA into OECD location codes.

    Parameters
    ----------
    regions : List[str]
        Tokens like ['US', 'EU', 'CN'] or ['ALL'].

    Returns
    -------
    List[str]
        OECD location codes such as ['USA', 'EU27_2020', 'CHN'].
        If 'ALL' is present, returns ['USA', 'EU27_2020', 'CHN'].
    """
    if "ALL" in [r.strip().upper() for r in regions]:
        return ["USA", "EU27_2020", "CHN"]
    mapping = {
        "US": "USA",
        "USA": "USA",
        "EU": "EU27_2020",
        "EU27_2020": "EU27_2020",
        "EA19": "EA19",
        "CN": "CHN",
        "CHN": "CHN",
    }
    norm: List[str] = []
    for r in regions:
        key = r.strip().upper()
        norm.append(mapping.get(key, key))
    return norm


# Added missing helper for OECD filename tags (restored from archived implementation)
def file_tag_for_oecd(loc_code: str) -> str:
    """Map OECD location code to output filename tag."""
    mapping = {
        "USA": "US",
        "CHN": "CN",
        "EU27_2020": "EU27_2020",
        "EA19": "EA19",
    }
    return mapping.get(loc_code, loc_code)


def file_tag_for_region(region_key: str) -> str:
    """Map internal region key to an output filename tag.

    Parameters
    ----------
    region_key : str
        Internal key like 'US', 'EU_EU27_2020', 'EU_EA19', 'CN_MEI'.

    Returns
    -------
    str
        Filename tag such as 'US', 'EU27_2020', 'EA19', or 'CN'.
    """
    mapping = {
        "US": "US",
        "EU_EU27_2020": "EU27_2020",
        "EU_EA19": "EA19",
        "CN_MEI": "CN",
    }
    return mapping.get(region_key, region_key)


def label_for_region(region_key: str) -> str:
    """Human-readable label for a region key.

    Parameters
    ----------
    region_key : str
        Internal key like 'US', 'EU_EU27_2020', 'EU_EA19', 'CN_MEI'.

    Returns
    -------
    str
        Descriptive label, e.g., 'United States' or 'European Union (EU27_2020)'.
    """
    mapping = {
        "US": "United States",
        "EU_EU27_2020": "European Union (EU27_2020)",
        "EU_EA19": "Euro Area (EA19)",
        "CN_MEI": "China",
    }
    return mapping.get(region_key, region_key)


def fetch_region_series(
    region_key: str,
    start: str | None,
    end: str | None,
    api_key: str | None,
) -> Tuple[pd.DataFrame, str]:
    """Fetch observations from FRED for the recommended series with a simple fallback.

    Parameters
    ----------
    region_key : str
        Internal region key (e.g., 'US', 'EU_EU27_2020', 'CN_MEI').
    start : Optional[str]
        Observation start date (YYYY-MM-DD). Passed through to FRED.
    end : Optional[str]
        Observation end date (YYYY-MM-DD). Passed through to FRED.
    api_key : Optional[str]
        FRED API key. If None, callers typically resolve from env.

    Returns
    -------
    Tuple[pd.DataFrame, str]
        (DataFrame, series_id) where DataFrame has columns ['date','value'] and
        series_id is the FRED series actually used.

    Raises
    ------
    RuntimeError
        If no observations are returned after attempting primary and fallback series.
    Exception
        Propagates the last underlying request/HTTP error if appropriate.

    Notes
    -----
    - EU_EU27_2020 falls back to EA19 if no observations are available.
    - China uses MEI/OECD-derived quarterly GDP series by default.
    """
    desc: FredSeries = recommended_series(region_key)
    try_ids = [desc.series_id]
    if region_key in FALLBACK_FRED_SERIES_ID:
        try_ids.append(FALLBACK_FRED_SERIES_ID[region_key])

    last_error = None
    for sid in try_ids:
        try:
            df = fred_observations(series_id=sid, api_key=api_key, start=start, end=end)
            if not df.empty:
                return df, sid
        except Exception as e:
            last_error = e
            logger.debug("Error fetching %s (%s): %s", region_key, sid, e)

    if last_error:
        raise last_error
    # If we get here without error but also no data
    raise RuntimeError(f"No observations returned for region {region_key} using series {try_ids}.")


def main() -> None:
    """CLI entry to fetch quarterly GDP via OECD (default) or FRED.

    OECD branch
    -----------
    - Regions normalized to OECD codes: USA, EU27_2020 (fallback to EA19), CHN.
    - Writes gdp_{TAG}.csv per region; optionally writes gdp_stacked.csv.

    FRED branch
    -----------
    - Regions normalized to internal keys: US, EU_EU27_2020 (fallback to EA19), CN_MEI.
    - Requires --fred-api-key or env FRED_API_KEY (also loaded from config/api_keys.env).
    - Writes gdp_{TAG}.csv per region; optionally writes gdp_stacked.csv.

    Common
    ------
    - --out-dir is resolved relative to this script if not absolute and created with parents.
    - Logging verbosity controlled via --log-level.
    """
    parser = argparse.ArgumentParser(description="Fetch quarterly GDP series for US, EU, and China from OECD (default) or FRED.")
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["ALL"],
        help="Regions to fetch. OECD: ALL, US/USA, EU/EU27_2020/EA19, CN/CHN. FRED: ALL, US, EU, EU_EA19, EU_EU27_2020, CN, CN_MEI",
    )
    parser.add_argument("--start", type=str, default=None, help="Observation start date (YYYY-MM-DD) [FRED only].")
    parser.add_argument("--end", type=str, default=None, help="Observation end date (YYYY-MM-DD) [FRED only].")
    parser.add_argument("--fred-api-key", type=str, default=None, help="FRED API key (or set FRED_API_KEY env var).")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data",
        help="Output directory for CSV files (relative to this script if not absolute).",
    )
    parser.add_argument("--stacked", action="store_true", help="Also write a stacked CSV across regions.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="OECD",
        choices=["OECD", "FRED"],
        help="Data source provider: OECD (default, no API key) or FRED (requires API key).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load env vars from config/api_keys.env if available (non-fatal if missing).
    _load_env_from_config()

    base_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        # Resolve output relative to the script directory for predictable behavior.
        out_dir = base_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)  # Safe creation with parents.

    stacked_frames: List[pd.DataFrame] = []

    # Branch by source: OECD (default) vs FRED
    if args.source.upper() == "OECD":
        # Normalize to OECD location codes
        locations = normalize_oecd_regions(args.regions)
        # Basic validation
        allowed = {"USA", "EU27_2020", "EA19", "CHN"}
        for loc in locations:
            if loc not in allowed:
                raise SystemExit(
                    f"Unknown OECD location '{loc}'. Use one of: ALL, US/USA, EU/EU27_2020/EA19, CN/CHN"
                )

        logger.info("Using OECD QNA CSV endpoint (SDMX) with start='2000-'.")
        for loc in locations:
            logger.info("Fetching OECD QNA for location=%s", loc)
            df = fetch_oecd_qna(location=loc, start="2000-")
            used_loc = df.attrs.get("oecd_location_used", loc)
            file_tag = file_tag_for_oecd(used_loc)

            if df.empty:
                logger.warning("No OECD data for %s (resolved=%s); skipping save.", loc, used_loc)
                continue

            save_path = out_dir / f"gdp_{file_tag}.csv"
            save_csv_oecd(df[["date", "gdp"]], save_path)
            logger.info("Saved %s (%d rows) [oecd_location=%s]", save_path, len(df), used_loc)

            region_label = file_tag  # Use filename tag as region code in stacked CSV
            stacked_frames.append(df.assign(region=region_label)[["date", "region", "gdp"]])

        if args.stacked and stacked_frames:
            stacked = pd.concat(stacked_frames, ignore_index=True)
            stacked = stacked.sort_values(["date", "region"]).reset_index(drop=True)
            save_path = out_dir / "gdp_stacked.csv"
            save_csv_oecd(stacked, save_path)
            logger.info("Saved stacked CSV: %s (%d rows)", save_path, len(stacked))
        return

    # FRED branch (original behavior)
    regions = normalize_regions(args.regions)

    # Validate region keys
    for rk in regions:
        if rk not in RECOMMENDED_FRED_SERIES:
            # Allow fallback key EU_EA19 even if not in main list (we keep it in RECOMMENDED_FRED_SERIES though)
            if rk not in ("EU_EA19",):
                raise SystemExit(
                    f"Unknown region key '{rk}'. Use one of: ALL, US, EU, EU_EA19, EU_EU27_2020, CN, CN_MEI"
                )

    api_key = args.fred_api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        raise SystemExit("Missing FRED API key. Set FRED_API_KEY env var or pass --fred-api-key.")

    for rk in regions:
        label = label_for_region(rk)
        file_tag = file_tag_for_region(rk)
        logger.info("Fetching region=%s (key=%s) from FRED", label, rk)
        df, sid = fetch_region_series(rk, args.start, args.end, api_key)
        df = df.rename(columns={"value": "gdp"})
        df["region"] = label
        save_path = out_dir / f"gdp_{file_tag}.csv"
        save_csv(df[["date", "gdp"]], save_path)
        logger.info("Saved %s (%d rows) [series_id=%s]", save_path, len(df), sid)
        stacked_frames.append(df[["date", "region", "gdp"]])

    if args.stacked and stacked_frames:
        stacked = pd.concat(stacked_frames, ignore_index=True)
        stacked = stacked.sort_values(["date", "region"]).reset_index(drop=True)
        save_path = out_dir / "gdp_stacked.csv"
        save_csv(stacked, save_path)
        logger.info("Saved stacked CSV: %s (%d rows)", save_path, len(stacked))


if __name__ == "__main__":
    main()
