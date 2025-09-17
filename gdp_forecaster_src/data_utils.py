# gdp_forecaster_src/data_utils.py

import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def load_macro_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the US macro quarterly dataset (1959–2009) from CSV or statsmodels.
    
    This function provides a standardized way to load macroeconomic data, either from
    a specified CSV file or from the statsmodels macrodata dataset. If the CSV doesn't
    exist but a path is provided, it will create the CSV from statsmodels data.

    Parameters
    ----------
    data_path : Optional[Path]
        If provided and exists, load from this CSV. If provided and does not exist,
        the statsmodels macrodata dataset is loaded and written to this CSV path (parents created).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns such as ['year', 'quarter', 'realgdp', 'realcons', 'realinv',
        'realgovt', 'realdpi', 'cpi', ...].

    Notes
    -----
    - When persisting, the CSV is written without an index.
    - The dataset covers 1959–2009 (statsmodels.macrodata).
    """
    if data_path and data_path.is_file():
        return pd.read_csv(data_path)
    df = sm.datasets.macrodata.load_pandas().data.copy()
    if data_path:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
    return df


def load_gdp_series_csv(series_path: Path) -> pd.Series:
    """
    Load a GDP series from a CSV file with 'date' and 'gdp' columns.
    
    This function loads and validates a GDP time series CSV, ensuring proper data types
    and handling missing values appropriately.

    Parameters
    ----------
    series_path : Path
        Path to CSV file containing GDP data with 'date' and 'gdp' columns.

    Returns
    -------
    pd.Series
        GDP series with DatetimeIndex, ready for time series analysis.

    Raises
    ------
    SystemExit
        If the file doesn't exist, lacks required columns, or contains no valid data.
    """
    if not series_path.exists():
        raise SystemExit(f"Series CSV not found: {series_path}")

    logger.info("Loading GDP series from: %s", series_path)
    df_series = pd.read_csv(series_path)
    
    if "date" not in df_series.columns or "gdp" not in df_series.columns:
        raise SystemExit("Series CSV must contain 'date' and 'gdp' columns.")
    
    # Parse and validate data
    df_series["date"] = pd.to_datetime(df_series["date"], errors="coerce")
    df_series["gdp"] = pd.to_numeric(df_series["gdp"], errors="coerce")
    df_series = df_series.dropna(subset=["date", "gdp"]).sort_values("date").reset_index(drop=True)
    
    if df_series.empty:
        raise SystemExit("No valid rows found in series CSV after parsing.")

    # Return as Series with DatetimeIndex
    return pd.Series(df_series["gdp"].values, index=df_series["date"], name="gdp")


def infer_country_from_series_path(series_path: Path) -> str:
    """
    Infer a country code from a GDP CSV filename.
    
    This function extracts country information from standardized GDP CSV filenames
    that follow the pattern 'gdp_{COUNTRY}.csv'.

    Parameters
    ----------
    series_path : Path
        Path to the GDP CSV file.

    Returns
    -------
    str
        Inferred country code or the filename stem if pattern doesn't match.

    Examples
    --------
    >>> infer_country_from_series_path(Path("gdp_US.csv"))
    'US'
    >>> infer_country_from_series_path(Path("my_data.csv"))
    'my_data'
    """
    stem = series_path.stem
    return stem[4:] if stem.lower().startswith("gdp_") else (stem or "series")


def ensure_series_csvs(base_dir: Path, countries: list[str], data_dir: Path) -> list[str]:
    """
    Ensure GDP series CSVs exist for specified countries, fetching if necessary.
    
    This function checks for the existence of GDP CSV files for each country and
    attempts to fetch missing ones using the project's GDP fetcher module.

    Parameters
    ----------
    base_dir : Path
        Base project directory.
    countries : list[str]
        List of country codes to check for.
    data_dir : Path
        Directory where GDP CSV files should be located.

    Returns
    -------
    list[str]
        List of countries for which GDP CSV files are available.

    Notes
    -----
    This function attempts to call the fetchers.fetch_gdp module if CSV files
    are missing. It handles import and execution errors gracefully.
    """
    import sys
    import subprocess
    
    data_dir.mkdir(parents=True, exist_ok=True)
    present = {c: (data_dir / f"gdp_{c}.csv").is_file() for c in countries}
    missing = [c for c, ok in present.items() if not ok]

    if missing:
        try:
            from fetchers import fetch_gdp as _fetch_gdp  # type: ignore
            old_argv = sys.argv[:]
            try:
                sys.argv = [old_argv[0], "--regions", "ALL", "--out-dir", str(data_dir), "--source", "OECD"]
                _fetch_gdp.main()
            except SystemExit:
                pass
            finally:
                sys.argv = old_argv
        except Exception as e:
            logger.warning("Failed to fetch GDP CSVs via fetchers.fetch_gdp: %s", e)

    available: list[str] = []
    for c in countries:
        p = data_dir / f"gdp_{c}.csv"
        if p.is_file():
            available.append(c)
        else:
            logger.warning("GDP CSV still missing for %s at %s", c, p)
    return available