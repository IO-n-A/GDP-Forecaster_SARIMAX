"""SARIMAX modeling and evaluation on US macroeconomic quarterly data (1959–2009).

Purpose
-------
- Load macro dataset (from CSV if provided, else statsmodels.macrodata).
- Visualize core series and run stationarity checks (ADF on levels and first differences).
- Grid-search SARIMAX orders by AIC over (p, q, P, Q) with fixed d=1, D=0, s=4 (quarterly seasonality).
- Fit selected configuration, save diagnostics and comparison figures to the figures/ directory.
- Rolling-origin one-step forecasts vs a naive last-value baseline; export evaluation metrics via econ_eval.

Data Sources & Attribution
---------------------------
GDP data is sourced from official statistical agencies:
- US: Bureau of Economic Analysis via FRED (Public Domain)
- EU: Eurostat via OECD SDMX (CC BY 4.0)
- China: National Bureau of Statistics via OECD (CC BY 4.0)

Exogenous variables from region-specific sources:
- US: ISM Manufacturing PMI from DB.nomics
- EU: Economic Sentiment Indicator from Eurostat
- China: NBS Manufacturing PMI

Configuration-Driven Workflow
-----------------------------
Model parameters, data sources, and evaluation settings are managed via
YAML configuration files in the config/ directory. CLI arguments override
configuration values where applicable.

Inputs / Outputs
----------------
- Input CSV (--data) is resolved relative to this file when not absolute; it is created on first run if missing.
- Figures are written under --figures-dir (default: figures/):
  RealGDP.png, MacroPanel.png, Diagnostics.png, ForecastOverlay_*.png,
  APE_Curves_*.png, and MAPEComparison*.png.
- Optional metrics CSV (--metrics-csv) is appended with rows for SARIMA and naive baselines.

CLI Overview
------------
--data, --figures-dir, --log-level, --series-csv (endog-only), --metrics-csv,
--batch-run, --batch-countries, --default-run.
"""
import argparse
import logging
logger = logging.getLogger(__name__)
import warnings
import csv
from itertools import product
from pathlib import Path
from typing import List, Optional, Union
from typing import Tuple

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from tqdm.auto import tqdm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.ar_model import ARResults
#NEW: exogenous data fetchers and aggregation
from helpers.temporal import monthly_to_quarterly_avg
from fetchers.dbnomics_pmi import ism_pmi_monthly, DBNOMICS_ISM_PMI_URL
from fetchers.eurostat_bcs import fetch_esi_sdmx, fetch_industry_conf_sdmx, to_monthly_series
from fetchers.nbs_pmi import load_nbs_pmi_csv, fetch_dbnomics_nbs
#NEW: diagnostics for residual normality summary in eval markdown
from diagnostics.residual_diagnostics import run_comprehensive_diagnostics
from datetime import datetime

#NEW: subprocess + sys + os for default-run orchestration
import sys
import subprocess
import os
import hashlib


# Initialize the global configuration manager
global config_manager
config_manager = None
CONFIG_AVAILABLE = False
try:
    from config import get_config  # use project configuration manager
    CONFIG_AVAILABLE = True
    logger.info("Configuration system detected: get_config")
except Exception as e:
    CONFIG_AVAILABLE = False
    logger.warning("Configuration system NOT detected: %s - using defaults", e)

# ---------- Helpers for ranges, intervals, and transforms ----------

def _parse_range_arg(s: Optional[str], default: str = "0-3", config_key: Optional[str] = None, args: Optional[argparse.Namespace] = None) -> list[int]:
    """
    Parse a CLI range argument like '0-3' or '0,1,2,3' into a list of ints.
    Now supports configuration system fallback.
    
    Parameters
    ----------
    s : str, optional
        CLI range argument
    default : str
        Default range if no CLI arg or config value
    config_key : str, optional
        Configuration key path for fallback value
    args : argparse.Namespace, optional
        CLI arguments for precedence checking
        
    Returns
    -------
    list[int]
        Parsed range as list of integers
    """
    # Use configuration-aware helper for getting the value
    if config_key:
        range_value = _get_config_value(config_key, default, args, None)
        # If we got a list from config, return it directly
        if isinstance(range_value, list):
            return sorted(set(int(x) for x in range_value))
        # Otherwise treat as string to parse
        txt = str(range_value).strip() if range_value is not None else default
    else:
        txt = (s or default).strip()
    
    out: list[int] = []
    if "-" in txt and "," not in txt:
        try:
            a, b = txt.split("-", 1)
            lo = int(a.strip())
            hi = int(b.strip())
            out = list(range(lo, hi + 1))
        except Exception:
            pass
    else:
        try:
            out = [int(x.strip()) for x in txt.split(",") if x.strip() != ""]
        except Exception:
            out = []
    
    if not out:
        out = [0, 1, 2, 3]
    return sorted(set(out))


def _parse_intervals_arg(s: Optional[str], default: str = "80,95") -> list[int]:
    """
    Parse a CLI intervals argument like '80,95' into sorted unique integer coverage levels.
    """
    txt = (s or default).strip()
    try:
        vals = sorted({int(x.strip()) for x in txt.split(",") if x.strip() != ""})
        vals = [v for v in vals if 1 <= v < 100]
        return vals or [80, 95]
    except Exception:
        return [80, 95]


def _apply_target_transform(series: pd.Series, transform: str) -> tuple[pd.Series, int]:
    """
    Apply target transform and return (transformed_series, suggested_differencing_d).
    - level: identity, d=1 (typical)
    - qoq: percent change * 100, d=0
    - yoy: percent change vs t-4 * 100, d=0
    - log_diff: 100 * diff(log), d=0
    
    Now supports configuration-driven transform mapping.
    """
    global config_manager
    
    # Get differencing mapping from configuration
    d_mapping = {}
    if config_manager:
        try:
            d_mapping = config_manager.get('model.target_transforms.differencing_mapping', {})
        except Exception:
            pass
    
    # Default mappings if config not available
    if not d_mapping:
        d_mapping = {
            'level': 1,
            'qoq': 0, 
            'yoy': 0,
            'log_diff': 0
        }
    
    if transform == "level":
        return series, d_mapping.get('level', 1)
    elif transform == "qoq":
        qoq_series = series.pct_change() * 100
        return qoq_series, d_mapping.get('qoq', 0)
    elif transform == "yoy":
        yoy_series = series.pct_change(periods=4) * 100
        return yoy_series, d_mapping.get('yoy', 0)
    elif transform == "log_diff":
        log_diff_series = np.log(series).diff() * 100
        return log_diff_series, d_mapping.get('log_diff', 0)
    else:
        logger.warning("Unknown transform '%s', using 'level'", transform)
        return series, d_mapping.get('level', 1)


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import math  # added for metrics and statistical functions
#Suppress non-critical warnings (e.g., statsmodels convergence messages) for cleaner CLI logs.
#Toggle by removing this line locally or by running with --log-level=DEBUG to inspect diagnostics.
warnings.filterwarnings("ignore")


#Ensure directory exists (Path-friendly)
def ensure_dir(path: Path) -> None:
    """Create the directory if it does not already exist.

    Parameters
    ----------
    path : Path
        Target directory path. All missing parents are created.

    Notes
    -----
    Side-effect only; returns None.
    """
    path.mkdir(parents=True, exist_ok=True)

# Evaluation markdown sink helpers
def _get_eval_md_path(args: Optional[argparse.Namespace], base_dir: Path) -> Path:
    try:
        override = getattr(args, "eval_md", None) if args is not None else None
    except Exception:
        override = None
    p = Path(override) if override else (base_dir / "analysis" / "eval_SARIMAX.md")
    ensure_dir(p.parent)
    return p

def _append_eval_md(eval_md_path: Path, title: str, body: str) -> None:
    try:
        ts = datetime.utcnow().isoformat()
        with eval_md_path.open("a", encoding="utf-8") as f:
# --- Stationarity auto-selection helpers (ADF-based) ---
def _safe_adf_pval(series: pd.Series) -> float:
    try:
        s = pd.Series(series).dropna()
        if len(s) < 12:
            return float("nan")
        return float(adfuller(s)[1])
    except Exception:
        return float("nan")

def _adf_select_d(series: pd.Series, alpha: float = 0.05) -> int:
    """
    Heuristic: if level is stationary (p&lt;alpha) then d=0 else d=1.
    """
    p_level = _safe_adf_pval(series)
    if np.isfinite(p_level) and p_level &lt; alpha:
        return 0
    return 1

def _adf_select_D(series: pd.Series, s: int = 4, alpha: float = 0.05) -> int:
    """
    Heuristic: if seasonal differenced series is stationary (p&lt;alpha) while level is non-stationary,
    select one seasonal difference; else D=0.
    """
    p_level = _safe_adf_pval(series)
    try:
        p_seasonal = _safe_adf_pval(pd.Series(series).diff(s))
    except Exception:
        p_seasonal = float("nan")
    if (not np.isfinite(p_level) or p_level &gt;= alpha) and np.isfinite(p_seasonal) and p_seasonal &lt; alpha:
        return 1
    return 0
            f.write(f"\n\n## {title}  \n")
            f.write(f"_timestamp: {ts}_\n\n")
            f.write(body.strip() + "\n")
    except Exception as e:
        logger.debug("Failed to append to eval markdown %s: %s", eval_md_path, e)

def _md_table_from_df(df: pd.DataFrame, max_rows: int = 10, columns: Optional[List[str]] = None) -> str:
    try:
        if columns is not None:
            keep = [c for c in columns if c in df.columns]
            if keep:
                df = df.loc[:, keep]
        df_disp = df.head(max_rows).copy()
        cols = list(df_disp.columns)
        if not cols:
            return ""
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        rows = []
        for _, row in df_disp.iterrows():
            vals = [row[c] for c in cols]
            rows.append("| " + " | ".join(str(v) for v in vals) + " |")
        return "\n".join([header, sep] + rows)
    except Exception as e:
        logger.debug("Failed to render markdown table: %s", e)
        return ""
#Initialize configuration system
def _initialize_config():
    """Initialize the global configuration manager."""
    global config_manager
    if CONFIG_AVAILABLE and config_manager is None:
        try:
            config_manager = get_config()
            validation_errors = config_manager.validate_configuration()
            if validation_errors:
                logger.warning("Configuration validation warnings: %s", validation_errors)
        except Exception as e:
            logger.error("Failed to initialize configuration: %s. Using defaults.", e)
            config_manager = None

def _get_config_value(key_path: str, default=None, args=None, cli_param=None):
    """Get configuration value with CLI override support.
    
    Parameters
    ----------
    key_path : str
        Dot-separated configuration path
    default : any
        Default value if not found in config
    args : argparse.Namespace, optional
        CLI arguments namespace
    cli_param : str, optional
        CLI parameter name that overrides config
        
    Returns
    -------
    any
        Configuration value with CLI override applied
    """
    # First priority: CLI argument
    if args and cli_param and hasattr(args, cli_param):
        cli_value = getattr(args, cli_param)
        if cli_value is not None:
            return cli_value
    
    # Second priority: Configuration file
    if config_manager:
        try:
            config_value = config_manager.get(key_path, default)
            if config_value is not None:
                return config_value
        except Exception as e:
            logger.debug("Error accessing config key '%s': %s", key_path, e)
    
    # Third priority: Default value
    return default

# Load US macroeconomic quarterly data.
# If data_path CSV exists, load from it; otherwise load from statsmodels and optionally persist to CSV.
def load_macro_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the US macro quarterly dataset (1959–2009).

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

# Plot Real GDP over time using index-based x-axis with year tick labels.
def plot_realgdp(df: pd.DataFrame, out_path: Path) -> None:
    """Render and save the Real GDP series plot.

    Parameters
    ----------
    df : pd.DataFrame
        Macro dataset containing the 'realgdp' column.
    out_path : Path
        File path to save the rendered PNG (parents are created if missing).

    Returns
    -------
    None
    """
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots()
    ax.plot(df["realgdp"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Real GDP (k$)")
    n = len(df)
    ax.set_xticks(np.arange(0, n, 16))
    ax.set_xticklabels(np.arange(1959, 2010, 4))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

# Plot six macroeconomic series (columns 2..7) in a 3x2 grid for quick inspection.
def plot_selected_columns(df: pd.DataFrame, out_path: Path) -> None:
    """Render and save a 3x2 panel of key macro series.

    Parameters
    ----------
    df : pd.DataFrame
        Macro dataset with at least 8 columns; uses df.columns[2:8] to plot six series
        (skipping 'year' and 'quarter').
    out_path : Path
        File path to save the PNG (parents are created if missing).

    Returns
    -------
    None
    """
    ensure_dir(out_path.parent)
    fig, axes = plt.subplots(nrows=3, ncols=2, dpi=300, figsize=(11, 6))
    cols = df.columns[2:8]  # Skip 'year', 'quarter'
    for i, ax in enumerate(axes.flatten()[:6]):
        col = cols[i]
        ax.plot(df[col], color="black", linewidth=1)
        ax.set_title(col)
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=6)
    for ax in axes.flatten():
        ax.set_xticks(np.arange(0, len(df), 8))
        ax.set_xticklabels(np.arange(1959, 2010, 2))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

# Augmented Dickey–Fuller test; return (statistic, p-value).
def adf_test(series: Union[pd.Series, np.ndarray]) -> tuple:
    """Run the Augmented Dickey–Fuller (ADF) test.

    Parameters
    ----------
    series : Union[pd.Series, np.ndarray]
        Input series. NaNs are dropped prior to testing.

    Returns
    -------
    tuple
        (statistic: float, p_value: float)

    Notes
    -----
    - ADF null: the series has a unit root (non-stationary).
    - We typically test both the level and the first difference to assess stationarity.
    """
    res = adfuller(pd.Series(series).dropna())
    return res[0], res[1]


# Grid-search SARIMAX hyperparameters over (p,q,P,Q) with fixed d,D,s; return AIC-ranked results.
def optimize_sarimax(endog: Union[pd.Series, list], exog: Union[pd.DataFrame, list], order_list: List[tuple], d: int, D: int, s: int) -> pd.DataFrame:
    """Grid-search SARIMAX hyperparameters and rank by AIC.

    Parameters
    ----------
    endog : Union[pd.Series, list]
        Endogenous (target) series.
    exog : Union[pd.DataFrame, list]
        Optional exogenous regressors aligned with endog (may be None for endog-only).
    order_list : List[tuple]
        List of (p, q, P, Q) tuples. Differencing orders d, D, and seasonal period s are fixed.
    d : int
        Non-seasonal differencing order (default usage: 1).
    D : int
        Seasonal differencing order (default usage: 0).
    s : int
        Seasonal period (quarterly data uses s=4).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['(p,q,P,Q)', 'AIC', 'BIC', 'HQIC'] sorted ascending by AIC.

    Notes
    -----
    - Model is fit with simple_differencing=False to retain internal differencing behavior.
    - Exceptions during fit are skipped to keep the search robust.
    """
    results: list[list[object]] = []
    for order in tqdm(order_list, desc="Grid search SARIMAX"):
        try:
            model = SARIMAX(
                endog,
                exog,
                order=(order[0], d, order[1]),
                seasonal_order=(order[2], D, order[3], s),
                simple_differencing=False,
            ).fit(disp=False)
        except Exception:
            continue
        aic = float(getattr(model, "aic", np.nan))
        bic = float(getattr(model, "bic", np.nan))
        hqic = float(getattr(model, "hqic", np.nan))
        results.append([order, aic, bic, hqic])
    result_df = pd.DataFrame(results, columns=["(p,q,P,Q)", "AIC", "BIC", "HQIC"])
    result_df = result_df.sort_values(by="AIC", ascending=True).reset_index(drop=True)
    return result_df

# Metrics and evaluation helpers (stabilized MAPE, DM test, residual metrics, p-value combiner)
def _to_1d_array(x: Union[List[float], np.ndarray, pd.Series]) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    return arr[np.isfinite(arr)]

def _mape_epsilon_from_train(y_train: Union[List[float], np.ndarray, pd.Series]) -> float:
    arr = _to_1d_array(y_train)
    if arr.size == 0:
        return 1e-8
    return float(max(1e-8, np.percentile(np.abs(arr), 10.0)))

def mape_eps(y_true: Union[List[float], np.ndarray, pd.Series], y_hat: Union[List[float], np.ndarray, pd.Series], eps: float) -> float:
    yt = _to_1d_array(y_true)
    yh = _to_1d_array(y_hat)
    n = min(len(yt), len(yh))
    if n == 0:
        return float("nan")
    yt = yt[:n]
    yh = yh[:n]
    denom = np.maximum(np.abs(yt), eps)
    return float(np.mean(np.abs(yh - yt) / denom) * 100.0)

def smape(y_true: Union[List[float], np.ndarray, pd.Series], y_hat: Union[List[float], np.ndarray, pd.Series], eps: float = 1e-12) -> float:
    yt = _to_1d_array(y_true)
    yh = _to_1d_array(y_hat)
    n = min(len(yt), len(yh))
    if n == 0:
        return float("nan")
    yt = yt[:n]
    yh = yh[:n]
    denom = np.maximum(np.abs(yt) + np.abs(yh), eps)
    return float(np.mean(2.0 * np.abs(yh - yt) / denom) * 100.0)

def mae(y_true: Union[List[float], np.ndarray, pd.Series], y_hat: Union[List[float], np.ndarray, pd.Series]) -> float:
    yt = _to_1d_array(y_true)
    yh = _to_1d_array(y_hat)
    n = min(len(yt), len(yh))
    if n == 0:
        return float("nan")
    return float(np.mean(np.abs(yh[:n] - yt[:n])))

def rmse(y_true: Union[List[float], np.ndarray, pd.Series], y_hat: Union[List[float], np.ndarray, pd.Series]) -> float:
    yt = _to_1d_array(y_true)
    yh = _to_1d_array(y_hat)
    n = min(len(yt), len(yh))
    if n == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yh[:n] - yt[:n]) ** 2)))

def mase_metric(y_true: Union[List[float], np.ndarray, pd.Series], y_hat: Union[List[float], np.ndarray, pd.Series], y_train: Union[List[float], np.ndarray, pd.Series], m: int = 4) -> float:
    yt = _to_1d_array(y_true)
    yh = _to_1d_array(y_hat)
    n = min(len(yt), len(yh))
    if n == 0:
        return float("nan")
    num = np.mean(np.abs(yh[:n] - yt[:n]))
    tr = _to_1d_array(y_train)
    if len(tr) <= m:
        return float("nan")
    denom = np.mean(np.abs(tr[m:] - tr[:-m]))
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return float(num / denom)

def theil_u1(y_true: Union[List[float], np.ndarray, pd.Series], y_hat: Union[List[float], np.ndarray, pd.Series]) -> float:
    yt = _to_1d_array(y_true)
    yh = _to_1d_array(y_hat)
    n = min(len(yt), len(yh))
    if n == 0:
        return float("nan")
    yt = yt[:n]
    yh = yh[:n]
    rmse_f = math.sqrt(float(np.mean((yh - yt) ** 2)))
    denom = math.sqrt(float(np.mean(yt ** 2))) + math.sqrt(float(np.mean(yh ** 2)))
    if denom <= 0.0:
        return float("nan")
    return float(rmse_f / denom)

def theil_u2(y_true: Union[List[float], np.ndarray, pd.Series], y_hat: Union[List[float], np.ndarray, pd.Series], y_hat_naive: Union[List[float], np.ndarray, pd.Series]) -> float:
    yt = _to_1d_array(y_true)
    yh = _to_1d_array(y_hat)
    yn = _to_1d_array(y_hat_naive)
    n = min(len(yt), len(yh), len(yn))
    if n == 0:
        return float("nan")
    rmse_f = math.sqrt(float(np.mean((yh[:n] - yt[:n]) ** 2)))
    rmse_n = math.sqrt(float(np.mean((yn[:n] - yt[:n]) ** 2)))
    if rmse_n == 0.0:
        return float("nan")
    return float(rmse_f / rmse_n)


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def _dm_newey_west_var(d: np.ndarray, h: int) -> float:
    # Spectral density at zero via Newey-West with lag L=h-1
    n = len(d)
    if n < 3:
        return float("nan")
    dbar = float(np.mean(d))
    e = d - dbar
    L = max(0, int(h) - 1)
    # Auto-covariances
    gamma0 = float(np.mean(e * e))
    s_hat = gamma0
    for k in range(1, L + 1):
        cov = float(np.mean(e[k:] * e[:-k]))
        w = 1.0 - (k / (L + 1.0))
        s_hat += 2.0 * w * cov
    var_dbar = s_hat / n
    return float(var_dbar) if var_dbar > 0.0 else float("nan")

def diebold_mariano(y_true: Union[List[float], np.ndarray, pd.Series], y_hat1: Union[List[float], np.ndarray, pd.Series], y_hat2: Union[List[float], np.ndarray, pd.Series], h: int = 1, power: int = 2) -> tuple[float, float]:
    yt = _to_1d_array(y_true)
    y1 = _to_1d_array(y_hat1)
    y2 = _to_1d_array(y_hat2)
    n = min(len(yt), len(y1), len(y2))
    if n < 3:
        return float("nan"), float("nan")
    yt = yt[:n]; y1 = y1[:n]; y2 = y2[:n]
    e1 = y1 - yt
    e2 = y2 - yt
    if power == 1:
        l1 = np.abs(e1)
        l2 = np.abs(e2)
    else:
        l1 = e1 ** 2
        l2 = e2 ** 2
    d = l1 - l2
    dbar = float(np.mean(d))
    var_dbar = _dm_newey_west_var(d, h=h)
    if not np.isfinite(var_dbar) or var_dbar <= 0.0:
        return float("nan"), float("nan")
    dm_t = dbar / math.sqrt(var_dbar)
    p = 2.0 * (1.0 - _norm_cdf(abs(dm_t)))
    return float(dm_t), float(min(max(p, 0.0), 1.0))


def _norm_ppf(p: float) -> float:
    # Acklam's inverse normal CDF approximation
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
           6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
          -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
           3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    q = p - 0.5
    r = q * q
    return(((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)

def combine_dm_pvalues(pvals: List[float], method: str = "fisher") -> float:
    vals = [float(p) for p in pvals if np.isfinite(p) and 0.0 <= p <= 1.0]
    if not vals:
        return float("nan")
    mth = (method or "fisher").lower()
    # Prefer Stouffer combination (no SciPy needed)
    if mth == "stouffer":
        zvals = [_norm_ppf(1.0 - v/2.0) for v in vals]
        z = float(np.sum(zvals)) / math.sqrt(len(zvals))
        return float(2.0 * (1.0 - _norm_cdf(abs(z))))
    # Fallback: approximate Fisher via Stouffer with a debug note
    try:
        zvals = [_norm_ppf(1.0 - v/2.0) for v in vals]
        z = float(np.sum(zvals)) / math.sqrt(len(zvals))
        return float(2.0 * (1.0 - _norm_cdf(abs(z))))
    except Exception:
        return float("nan")

def compute_metrics(
    y_true: Union[List[float], np.ndarray, pd.Series],
    y_hat: Union[List[float], np.ndarray, pd.Series],
    y_hat_naive: Union[List[float], np.ndarray, pd.Series],
    y_train: Union[List[float], np.ndarray, pd.Series],
    m: int = 4,
) -> dict:
    yt = _to_1d_array(y_true)
    yh = _to_1d_array(y_hat)
    yn = _to_1d_array(y_hat_naive)
    n = min(len(yt), len(yh), len(yn)) if len(yn) > 0 else min(len(yt), len(yh))
    yt = yt[:n]; yh = yh[:n]; yn = yn[:n] if len(yn) > 0 else np.array([], dtype=float)
    eps = _mape_epsilon_from_train(y_train)
    err = yh - yt
    res = {
        "ME": float(np.mean(err)) if n > 0 else float("nan"),
        "MAE": mae(yt, yh),
        "RMSE": rmse(yt, yh),
        "MAPE": mape_eps(yt, yh, eps),
        "sMAPE": smape(yt, yh),
        "median_APE": float(np.median(np.abs(err) / np.maximum(np.abs(yt), eps)) * 100.0) if n > 0 else float("nan"),
        "MASE": mase_metric(yt, yh, y_train, m=m),
        "TheilU1": theil_u1(yt, yh),
        "TheilU2": theil_u2(yt, yh, yn) if len(yn) > 0 else float("nan"),
    }
    if len(yn) > 0 and n > 0:
        dm_t, dm_p = diebold_mariano(yt, yh, yn, h=1, power=2)
        res["DM_t"] = dm_t
        res["DM_p"] = dm_p
    else:
        res["DM_t"] = float("nan")
        res["DM_p"] = float("nan")
    return res


# NEW: residual diagnostics helper (ACF/PACF, Ljung–Box table, ARCH LM summary)
def save_residual_diagnostics(residuals: Union[pd.Series, np.ndarray], out_dir: Path, fname_prefix: str = "Residuals") -> None:
    """
    Save residual diagnostics:
    - Combined ACF/PACF panel
    - Ljung–Box Q p-values table across lags
    - ARCH LM test summary

    Parameters
    ----------
    residuals : array-like
        Residual vector from a fitted model.
    out_dir : Path
        Output directory where artifacts will be written.
    fname_prefix : str
        Prefix for output filenames.
    """
    ensure_dir(out_dir)
    try:
        resid = pd.Series(residuals).dropna()
    except Exception:
        resid = pd.Series(np.asarray(residuals)).dropna()

    if resid.empty:
        logger.warning("Residual diagnostics skipped: empty residual series.")
        return

    # ACF/PACF panel
    try:
        lags = int(min(24, max(10, len(resid) // 4)))
    except Exception:
        lags = 24
    try:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), dpi=150)
        plot_acf(resid, ax=axes[0], lags=lags, zero=False)
        axes[0].set_title("Residual ACF")
        plot_pacf(resid, ax=axes[1], lags=lags, zero=False, method="ywmle")
        axes[1].set_title("Residual PACF")
        fig.tight_layout()
        fig.savefig(out_dir / f"{fname_prefix}_ACF_PACF.png", dpi=300)
        plt.close(fig)
    except Exception as e:
        logger.debug("Failed to render ACF/PACF diagnostics: %s", e)

    # Ljung–Box table (lags 1..min(24, n-1))
    try:
        max_lag = int(min(24, max(1, len(resid) - 1)))
        df_lb = acorr_ljungbox(resid, lags=np.arange(1, max_lag + 1), return_df=True)
        df_lb.to_csv(out_dir / f"{fname_prefix}_LjungBox.csv", index=True)
    except Exception as e:
        logger.debug("Ljung–Box diagnostics skipped: %s", e)

    # ARCH LM (heteroskedasticity in residuals)
    try:
        nlags = int(min(12, max(2, len(resid) // 10)))
        lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(resid, nlags=nlags)
        pd.DataFrame(
            {"lm_stat": [lm_stat], "lm_pvalue": [lm_pvalue], "f_stat": [f_stat], "f_pvalue": [f_pvalue]}
        ).to_csv(out_dir / f"{fname_prefix}_ARCH_LM.csv", index=False)
    except Exception as e:
        logger.debug("ARCH LM diagnostics skipped: %s", e)

    # CUSUM stability (approximate): standardized residuals cumulative sum with ~95% bounds
    try:
        mu = float(np.mean(resid))
        sd = float(np.std(resid, ddof=1))
        if not np.isfinite(sd) or sd <= 1e-12:
            raise ValueError("Degenerate residual std for CUSUM.")
        z = (resid - mu) / sd
        cusum = np.cumsum(z)
        t = np.arange(1, len(cusum) + 1)
        # Approximate 95% reference bounds for CUSUM; heuristic constant for visualization
        bound = 1.36 * np.sqrt(t)
        df_cusum = pd.DataFrame({"t": t, "cusum": cusum, "upper_95": bound, "lower_95": -bound})
        df_cusum.to_csv(out_dir / f"{fname_prefix}_CUSUM.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        ax.plot(t, cusum, label="CUSUM", color="tab:purple")
        ax.plot(t, bound, "--", color="gray", linewidth=1, label="±95% bounds")
        ax.plot(t, -bound, "--", color="gray", linewidth=1)
        ax.set_title("CUSUM stability (standardized residuals)")
        ax.set_xlabel("t")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{fname_prefix}_CUSUM.png", dpi=300)
        plt.close(fig)
    except Exception as e:
        logger.debug("CUSUM stability diagnostics skipped: %s", e)


def hash_forecast(seq: Union[List[float], np.ndarray, pd.Series]) -> str:
    arr = np.asarray(seq, dtype=np.float64)
    try:
        data = arr.tobytes()
    except Exception:
        data = np.asarray(list(seq), dtype=np.float64).tobytes()
    return hashlib.sha1(data).hexdigest()[:16]

# NEW: leakage-free rolling one-step prediction helpers for testing and reuse
def rolling_one_step_predictions(
    endog: pd.Series,
    exog_model: Optional[pd.DataFrame],
    best_order: tuple,
    d: int,
    D: int,
    s: int,
    coverage_levels: list[int],
    start_index: int,
) -> tuple[list[float], dict[int, tuple[list[float], list[float]]]]:
    """
    Compute leak-free rolling one-step-ahead predictions and predictive intervals.
    Returns (preds, conf_bounds) where conf_bounds[lvl] = (list[lo], list[hi]).
    """
    preds: list[float] = []
    conf_bounds: dict[int, tuple[list[float], list[float]]] = {lvl: ([], []) for lvl in coverage_levels}
    n = len(endog)
    for i in range(start_index, n):
        try:
            m = SARIMAX(
                endog.iloc[:i],
                (exog_model.iloc[:i] if exog_model is not None else None),
                order=(best_order[0], d, best_order[1]),
                seasonal_order=(best_order[2], D, best_order[3], s),
                simple_differencing=False,
            )
            r = m.fit(disp=False)
            fc = r.get_forecast(steps=1, exog=(exog_model.iloc[i : i + 1] if exog_model is not None else None))
            f_mean = float(fc.predicted_mean.iloc[0])
            for lvl in coverage_levels:
                alpha = 1.0 - (lvl / 100.0)
                ci = fc.conf_int(alpha=alpha)
                lo = float(pd.to_numeric(ci.iloc[0].values, errors="coerce").min())
                hi = float(pd.to_numeric(ci.iloc[0].values, errors="coerce").max())
                conf_bounds[lvl][0].append(lo)
                conf_bounds[lvl][1].append(hi)
        except Exception:
            # Fallback: naive last value when fit/forecast fails
            f_mean = float(endog.iloc[i - 1])
            for lvl in coverage_levels:
                conf_bounds[lvl][0].append(f_mean)
                conf_bounds[lvl][1].append(f_mean)
        preds.append(f_mean)
    return preds, conf_bounds


def one_step_forecast_at(
    endog: pd.Series,
    exog_model: Optional[pd.DataFrame],
    best_order: tuple,
    d: int,
    D: int,
    s: int,
    i: int,
) -> float:
    """
    Compute a single one-step-ahead forecast at index i using training data up to i-1.
    """
    if i <= 0 or i >= len(endog):
        raise ValueError("Index i must satisfy 1 <= i < len(endog).")
    try:
        m = SARIMAX(
            endog.iloc[:i],
            (exog_model.iloc[:i] if exog_model is not None else None),
            order=(best_order[0], d, best_order[1]),
            seasonal_order=(best_order[2], D, best_order[3], s),
            simple_differencing=False,
        )
        r = m.fit(disp=False)
        fc = r.get_forecast(steps=1, exog=(exog_model.iloc[i : i + 1] if exog_model is not None else None))
        return float(fc.predicted_mean.iloc[0])
    except Exception:
        return float(endog.iloc[i - 1])

# NEW: Append a single metrics row to CSV (create header on first write)
def _append_metrics_csv_row(csv_path: Optional[Path], row: dict, header: List[str]) -> None:
    if csv_path is None:
        return
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        exists = csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        logger.error("Failed to append metrics to %s: %s", csv_path, e)

# NEW: Build quarterly exogenous series for a region based on CLI args, aligned to endog index
def build_exog_for_country(country: str, args: argparse.Namespace, endog_index: pd.DatetimeIndex) -> Optional[pd.DataFrame]:
    """
    Construct a quarterly exogenous DataFrame aligned to endog_index with a single column 'EXOG'.
    Returns None if no exogenous source is configured or available.
    """
    if args is None or not getattr(args, "use_exog", False):
        return None

    code = (country or "").upper()
    monthly: Optional[pd.Series] = None

    try:
        if code == "US":
            src = getattr(args, "exog_source_us", "ism_pmi")
            if src == "ism_pmi":
                url = getattr(args, "us_ism_url", None) or DBNOMICS_ISM_PMI_URL
                monthly = ism_pmi_monthly(url=url)
            elif src == "industrial_production":
                # NEW: FRED Industrial Production option
                from fetchers.fred_indicators import fetch_us_industrial_production
                series_type = getattr(args, "us_ip_type", "total")  # total, manufacturing, capacity_util
                monthly = fetch_us_industrial_production(series_type=series_type)
            elif src == "manufacturers_orders":
                # NEW: FRED Manufacturers' Orders option  
                from fetchers.fred_indicators import fetch_us_manufacturers_orders
                monthly = fetch_us_manufacturers_orders()
            elif src == "oecd_cli":
                # NEW: OECD CLI harmonized option
                from fetchers.oecd_cli import fetch_oecd_cli
                monthly = fetch_oecd_cli("US")
        elif code in {"EU27_2020", "EA19", "EU", "EU27"}:
            src = getattr(args, "exog_source_eu", "esi")
            geo = getattr(args, "eu_geo", "EA19")
            if src == "esi":
                df = fetch_esi_sdmx(geo=geo, s_adj="SA")
                monthly = to_monthly_series(df, out_name="ESI")
            elif src == "industry":
                df = fetch_industry_conf_sdmx(geo=geo, s_adj="SA")
                monthly = to_monthly_series(df, out_name="ICI")
            elif src == "oecd_cli":
                # NEW: OECD CLI harmonized option
                from fetchers.oecd_cli import fetch_oecd_cli
                monthly = fetch_oecd_cli("EU")
        elif code == "CN":
            src = getattr(args, "exog_source_cn", "nbs_csv")
            if src == "nbs_csv":
                path = getattr(args, "exog_cn_csv", None)
                if path:
                    monthly = load_nbs_pmi_csv(path)
            elif src == "nbs_dbn":
                url = getattr(args, "exog_cn_url", None)
                if url:
                    monthly = fetch_dbnomics_nbs(url)
            elif src == "oecd_cli":
                # NEW: OECD CLI harmonized option
                from fetchers.oecd_cli import fetch_oecd_cli
                monthly = fetch_oecd_cli("CN")
    except Exception as e:
        logger.warning("Failed to fetch monthly exogenous series for %s: %s", country, e)
        monthly = None

    if monthly is None or monthly.empty:
        return None

    # Aggregate to quarter-end by mean and align to endog index
    exog_q = monthly_to_quarterly_avg(monthly, name="EXOG")
    exog_q = exog_q.reindex(endog_index)
    if exog_q.dropna().empty:
        return None
    return exog_q


# NEW: multi-fold rolling-origin backtest (expanding window), with DM aggregation
def run_endog_multifold(
    endog: pd.Series,
    exog_model: Optional[pd.DataFrame],
    best_order: tuple,
    d: int,
    D: int,
    s: int,
    figures_dir: Path,
    metrics_csv_path: Optional[Path],
    country: str,
    args: argparse.Namespace,
) -> None:
    """
    Perform multi-fold expanding-window backtesting.
    - folds: number of folds
    - fold_horizon: steps per fold
    Writes one aggregated metrics row with combined DM p-value across folds.
    """
    # compute_metrics and combine_dm_pvalues are defined locally

    folds = int(getattr(args, "folds", 3) or 3)
    fold_h = int(getattr(args, "fold_horizon", 8) or 8)
    n = len(endog)
    min_train = max(20, s * 4)  # require a minimal train size

    start_train = n - (folds * fold_h)
    if start_train < min_train:
        # shrink folds if series is too short
        folds = max(1, (n - min_train) // max(1, fold_h))
        start_train = n - (folds * fold_h)
        logger.warning("Adjusted folds to %d due to short series.", folds)
        if folds <= 0:
            raise SystemExit("Series too short for the requested multi-fold configuration.")

    all_true: list[float] = []
    all_pred: list[float] = []
    all_naive: list[float] = []
    dm_pvals: list[float] = []

    for k in range(folds):
        tr_end = start_train + k * fold_h
        te_end = tr_end + fold_h
        if te_end > n:
            break

        y_tr = endog.iloc[:tr_end]
        # X_tr not required explicitly; align exog to y_tr by index when fitting

        y_fold_true: list[float] = []
        y_fold_pred: list[float] = []
        y_fold_naive: list[float] = []

        for i in range(tr_end, te_end):
            try:
                m = SARIMAX(
                    y_tr,
                    (exog_model.loc[y_tr.index] if exog_model is not None else None),
                    order=(best_order[0], d, best_order[1]),
                    seasonal_order=(best_order[2], D, best_order[3], s),
                    simple_differencing=False,
                )
                r = m.fit(disp=False)
                fc = r.get_forecast(steps=1, exog=(exog_model.iloc[i : i + 1] if exog_model is not None else None))
                y_hat = float(fc.predicted_mean.iloc[0])
                y_tr = pd.concat([y_tr, pd.Series([endog.iloc[i]], index=[endog.index[i]])])
            except Exception:
                # Fallback to naive last value if fit/forecast fails
                y_hat = float(endog.iloc[i - 1])
            y_true_i = float(endog.iloc[i])
            y_naive_i = float(endog.iloc[i - 1])

            y_fold_true.append(y_true_i)
            y_fold_pred.append(y_hat)
            y_fold_naive.append(y_naive_i)

        # Compute metrics per fold
        met = compute_metrics(y_true=y_fold_true, y_hat=y_fold_pred, y_hat_naive=y_fold_naive, y_train=y_tr, m=4)
        if np.isfinite(met.get("DM_p", float("nan"))):
            dm_pvals.append(float(met.get("DM_p")))

        all_true.extend(y_fold_true)
        all_pred.extend(y_fold_pred)
        all_naive.extend(y_fold_naive)

    # Aggregate metrics
    agg = compute_metrics(y_true=all_true, y_hat=all_pred, y_hat_naive=all_naive, y_train=endog.iloc[:start_train], m=4)
    p_comb = combine_dm_pvalues(dm_pvals, method=str(getattr(args, "dm_method", "fisher")) if hasattr(args, "dm_method") else "fisher")

    # Export one aggregated row
    header = [
        "mode",
        "country",
        "n",
        "train_len",
        "test_len",
        "model",
        "ME",
        "MAE",
        "RMSE",
        "MAPE",
        "sMAPE",
        "median_APE",
        "MASE",
        "TheilU1",
        "TheilU2",
        "DM_t",
        "DM_p",
        "DM_p_combined",
        "hit_80",
        "hit_95",
        "folds",
        "fold_horizon",
        "hash_SARIMA",
        "hash_naive",
    ]
    try:
        sar_hash = hash_forecast(all_pred)
        nv_hash = hash_forecast(all_naive)
    except Exception:
        sar_hash = ""
        nv_hash = ""
    row = {
        "mode": "endog",
        "country": country,
        "n": len(all_true),
        "train_len": start_train,
        "test_len": len(all_true),
        "model": "SARIMA(best)_multifold",
        "ME": agg.get("ME"),
        "MAE": agg.get("MAE"),
        "RMSE": agg.get("RMSE"),
        "MAPE": agg.get("MAPE"),
        "sMAPE": agg.get("sMAPE"),
        "median_APE": agg.get("median_APE"),
        "MASE": agg.get("MASE"),
        "TheilU1": agg.get("TheilU1"),
        "TheilU2": agg.get("TheilU2"),
        "DM_t": agg.get("DM_t"),
        "DM_p": agg.get("DM_p"),
        "DM_p_combined": p_comb,
        "hit_80": "",
        "hit_95": "",
        "folds": folds,
        "fold_horizon": fold_h,
        "hash_SARIMA": sar_hash,
        "hash_naive": nv_hash,
    }
    _append_metrics_csv_row(metrics_csv_path, row, header)


def run_endog_only(series_path: Path, figures_dir: Path, metrics_csv_path: Optional[Path], args: Optional[argparse.Namespace] = None) -> None:
    """Execute an endog-only SARIMA workflow for a single GDP CSV.

    Parameters
    ----------
    series_path : Path
        Input CSV with columns ['date', 'gdp'] (date parseable to datetime).
    figures_dir : Path
        Output directory for endog-only figures (created if missing).
    metrics_csv_path : Optional[Path]
        If provided, append evaluation metrics to this CSV.

    Returns
    -------
    None

    Workflow
    --------
    - Load and validate the GDP series.
    - Grid search SARIMA over (p, q, P, Q) ∈ [0..3]^4 with d=1, D=0, s=4 (quarterly).
    - Fit best model by AIC; save diagnostics.
    - Rolling origin one-step evaluation on the last 16 quarters (adaptive for short series).
    - Compare SARIMA vs naive last-value; save overlay, APE curves, MAPE bar chart.
    - Append a metrics row for SARIMA and the naive baseline (m=4 for MASE scaling).

    Notes
    -----
    - Walk-forward forecast emulates real-time usage by refitting at each step.
    - MAPE can be sensitive to small denominators; see sMAPE/MASE for complementary insight.
    """
    if not series_path.exists():
        raise SystemExit(f"Series CSV not found: {series_path}")

    logger.info("Endog-only GDP CSV mode. Loading: %s", series_path)
    df_series = pd.read_csv(series_path)
    if "date" not in df_series.columns or "gdp" not in df_series.columns:
        raise SystemExit("Series CSV must contain 'date' and 'gdp' columns.")
    df_series["date"] = pd.to_datetime(df_series["date"], errors="coerce")
    df_series["gdp"] = pd.to_numeric(df_series["gdp"], errors="coerce")
    df_series = df_series.dropna(subset=["date", "gdp"]).sort_values("date").reset_index(drop=True)
    if df_series.empty:
        raise SystemExit("No valid rows found in series CSV after parsing.")

    # Use a DateIndex for modeling
    endog_raw = pd.Series(df_series["gdp"].values, index=df_series["date"], name="gdp")

    # Target transform handling
    target_transform = getattr(args, "target_transform", "level") if args is not None else "level"
    endog, d_for_transform = _apply_target_transform(endog_raw, target_transform)
    logger.info("Target transform: %s; resulting length=%d", target_transform, len(endog))
    # Drop NaNs introduced by transform (pre-exog alignment)
    _len_before = len(endog)
    endog = endog.dropna()
    if len(endog) < _len_before:
        logger.info("Dropped %d NaNs from transformed target after '%s'.", _len_before - len(endog), target_transform)

    # Optional: attach exogenous series if requested (align to transformed endog)
    country = _infer_country_from_series_path(series_path)
    exog_model: Optional[pd.DataFrame] = None
    if args is not None and getattr(args, "use_exog", False):
        exog_q = build_exog_for_country(country, args, endog.index)
        if exog_q is not None:
            df_joint = pd.concat([endog, exog_q["EXOG"]], axis=1).dropna()
            if not df_joint.empty:
                endog = df_joint.iloc[:, 0]
                exog_model = df_joint[["EXOG"]]
                logger.info("Using exogenous series for %s (aligned rows=%d).", country, len(endog))
            else:
                logger.info("Exogenous series for %s had no overlap with GDP; proceeding endog-only.", country)
        else:
            logger.info("No exogenous series available for %s; proceeding endog-only.", country)

    # Hyperparameter grid for SARIMA with fixed d=1, D=0, s=4; d depends on transform
    pL = _parse_range_arg(getattr(args, "p_range", None) if args is not None else None, 
                          config_key="model.search_space.p_range", args=args)
    qL = _parse_range_arg(getattr(args, "q_range", None) if args is not None else None,
                          config_key="model.search_space.q_range", args=args)
    PL = _parse_range_arg(getattr(args, "P_range", None) if args is not None else None,
                          config_key="model.search_space.P_range", args=args)
    QL = _parse_range_arg(getattr(args, "Q_range", None) if args is not None else None,
                          config_key="model.search_space.Q_range", args=args)
    
    # Get fixed parameters from configuration
    fixed_params = {}
    if config_manager:
        try:
            fixed_params = config_manager.get('model.fixed_parameters', {})
        except Exception:
            pass
    
    d = d_for_transform  # Use transform-specific differencing
    D = fixed_params.get('D', 0)  # Get from config or default
    s = fixed_params.get('s', 4)  # Get from config or default
    
    parameters_list = list(product(pL, qL, PL, QL))

    # Grid search (AIC-ranked) with optional caching
    result_df = optimize_sarimax(endog, exog_model, parameters_list, d, D, s)
    if result_df.empty:
        raise SystemExit("Grid search failed: no SARIMA configuration could be fit.")
    logger.info("Top 5 models by AIC:\n%s", result_df.head().to_string())

    aic_cache = getattr(args, "aic_cache", None) if args is not None else None
    if aic_cache:
        try:
            cache_path = Path(aic_cache)
            if not cache_path.is_absolute():
                cache_path = figures_dir / cache_path
            ensure_dir(cache_path.parent)
            result_df.to_csv(cache_path, index=False)
            logger.info("Saved AIC grid to %s", cache_path)
        except Exception as e:
            logger.warning("Failed to save AIC grid cache: %s", e)

    # Fit best model with robust errors support
    best_order = result_df.iloc[0]["(p,q,P,Q)"]
    logger.info("Selected (p,q,P,Q)=%s with AIC=%.3f", best_order, float(result_df.iloc[0]["AIC"]))
    
    # SARIMAX model fitting with robust error support
    fit_kwargs = {"disp": False}
    if config_manager:
        # Check for robust errors configuration
        robust_config = {}
        try:
            robust_config = config_manager.get('model.robust_errors', {})
        except Exception:
            pass
        if robust_config.get('enabled', False):
            fit_kwargs['cov_type'] = robust_config.get('cov_type', 'robust')
            cov_kwds = robust_config.get('cov_kwds', {})
            if cov_kwds:
                fit_kwargs['cov_kwds'] = cov_kwds
            logger.info("Using robust standard errors: %s", fit_kwargs.get('cov_type'))

    best_model = SARIMAX(
        endog,
        exog_model,
        order=(best_order[0], d, best_order[1]),
        seasonal_order=(best_order[2], D, best_order[3], s),
        simple_differencing=False,
    )
    best_res = best_model.fit(**fit_kwargs)
    logger.info("Model summary:\n%s", best_res.summary())

    # Diagnostics plot
    fig = best_res.plot_diagnostics(figsize=(10, 8))
    ensure_dir(figures_dir)
    fig.savefig(figures_dir / "Diagnostics.png", dpi=300)
    plt.close(fig)
    logger.info("Saved diagnostics: %s", figures_dir / "Diagnostics.png")

    # Additional residual diagnostics (ACF/PACF + tests)
    save_residual_diagnostics(best_res.resid, figures_dir, fname_prefix="Endog_Residuals")

    # If multi-fold requested, branch to multi-fold evaluation and return
    if args is not None and getattr(args, "multi_fold", False):
        run_endog_multifold(
            endog=endog,
            exog_model=exog_model,
            best_order=best_order,
            d=d,
            D=D,
            s=s,
            figures_dir=figures_dir,
            metrics_csv_path=metrics_csv_path,
            country=country,
            args=args,
        )
        return

    # Rolling-origin one-step evaluation on the last 16 quarters (adaptive for short series).
    TEST_LEN_DEFAULT = 16
    if len(endog) <= TEST_LEN_DEFAULT + 8:
        test_len = max(1, min(TEST_LEN_DEFAULT, len(endog) // 4))
        logger.warning("Short series detected. Using test_len=%d (default was %d).", test_len, TEST_LEN_DEFAULT)
    else:
        test_len = TEST_LEN_DEFAULT
    train_len = len(endog) - test_len

    # Collect one-step-ahead predictions and confidence intervals (leakage-free via helper)
    coverage_levels = _parse_intervals_arg(getattr(args, "intervals", None) if args is not None else None)
    preds, conf_bounds = rolling_one_step_predictions(
        endog=endog,
        exog_model=exog_model,
        best_order=best_order,
        d=d,
        D=D,
        s=s,
        coverage_levels=coverage_levels,
        start_index=train_len,
    )

    # True values over test window
    y_true = endog.iloc[train_len:]
    # MAPE logs (stabilized with training-based epsilon)
    _eps = _mape_epsilon_from_train(endog.iloc[:train_len])
    mape_val = mape_eps(y_true, preds, _eps)
    logger.info("Rolling-origin MAPEε over last %d periods (%s): %.6f%% (ε=%.6g)", len(y_true), target_transform, mape_val, _eps)
    naive_preds = [endog.iloc[i - 1] for i in range(train_len, len(endog))]
    mape_naive = mape_eps(y_true, naive_preds, _eps)
    logger.info("Rolling origin naive last-value MAPEε over last %d periods: %.6f%% (ε=%.6g)", len(y_true), mape_naive, _eps)

    # Interval hit-rates for selected levels
    hits: dict[int, float] = {}
    try:
        arr_y = np.asarray(y_true.values, dtype=float)
        for lvl in coverage_levels:
            lo_arr = np.asarray(conf_bounds[lvl][0])
            hi_arr = np.asarray(conf_bounds[lvl][1])
            hits[lvl] = float(np.mean((arr_y >= lo_arr) & (arr_y <= hi_arr)))
        # For legacy columns
        hit80 = hits.get(80, float("nan"))
        hit95 = hits.get(95, float("nan"))
        logger.info("Interval hit-rates: %s", ", ".join(f"{lvl}%={hits[lvl]:.3f}" for lvl in sorted(hits)))
    except Exception:
        hit80 = float("nan")
        hit95 = float("nan")

    # NEW: Calibration plot (empirical interval coverages vs requested levels)
    try:
        if hits:
            fig, ax = plt.subplots()
            lvls = sorted(hits.keys())
            vals = [hits[l] for l in lvls]
            ax.bar([f"{l}%" for l in lvls], vals, color="tab:purple", alpha=0.85)
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Empirical coverage")
            ax.set_title("Prediction interval calibration (endog-only)")
            for i, v in enumerate(vals):
                ax.text(i, min(0.98, v + 0.02), f"{v:.2f}", ha="center", fontsize=8)
            plt.tight_layout()
            fig.savefig(figures_dir / "Calibration_endog.png", dpi=300)
            plt.close(fig)
    except Exception as e:
        logger.debug("Calibration plot generation failed: %s", e)

    # Compute and export metrics (endog-only)
    country = _infer_country_from_series_path(series_path)
    try:
        sar_hash = hash_forecast(preds)
        nv_hash = hash_forecast(naive_preds)
        logger.info("Forecast hashes [%s] SARIMA=%s | naive=%s", country, sar_hash, nv_hash)
    except Exception as _e:
        sar_hash = ""
        nv_hash = ""
        logger.debug("Could not compute forecast hashes for %s: %s", country, _e)

    header = [
        "mode",
        "country",
        "n",
        "train_len",
        "test_len",
        "model",
        "ME",
        "MAE",
        "RMSE",
        "MAPE",
        "sMAPE",
        "median_APE",
        "MASE",
        "TheilU1",
        "TheilU2",
        "DM_t",
        "DM_p",
        "DM_p_combined",
        "hit_80",
        "hit_95",
        "folds",
        "fold_horizon",
        "hash_SARIMA",
        "hash_naive",
    ]
    n_obs = int(len(y_true))
    metrics_model = compute_metrics(
        y_true=y_true, y_hat=preds, y_hat_naive=naive_preds, y_train=endog.iloc[:train_len], m=4
    )
    row_model = {
        "mode": "endog",
        "country": country,
        "n": n_obs,
        "train_len": train_len,
        "test_len": n_obs,
        "model": "SARIMA(best)",
        "ME": metrics_model.get("ME"),
        "MAE": metrics_model.get("MAE"),
        "RMSE": metrics_model.get("RMSE"),
        "MAPE": metrics_model.get("MAPE"),
        "sMAPE": metrics_model.get("sMAPE"),
        "median_APE": metrics_model.get("median_APE"),
        "MASE": metrics_model.get("MASE"),
        "TheilU1": metrics_model.get("TheilU1"),
        "TheilU2": metrics_model.get("TheilU2"),
        "DM_t": metrics_model.get("DM_t"),
        "DM_p": metrics_model.get("DM_p"),
        "DM_p_combined": "",
        "hit_80": hits.get(80, "") if 80 in hits else "",
        "hit_95": hits.get(95, "") if 95 in hits else "",
        "folds": "",
        "fold_horizon": "",
        "hash_SARIMA": sar_hash,
        "hash_naive": nv_hash,
    }
    _append_metrics_csv_row(metrics_csv_path, row_model, header)

    metrics_naive = compute_metrics(
        y_true=y_true, y_hat=naive_preds, y_hat_naive=naive_preds, y_train=endog.iloc[:train_len], m=4
    )
    row_naive = {
        "mode": "endog",
        "country": country,
        "n": n_obs,
        "train_len": train_len,
        "test_len": n_obs,
        "model": "naive_last",
        "ME": metrics_naive.get("ME"),
        "MAE": metrics_naive.get("MAE"),
        "RMSE": metrics_naive.get("RMSE"),
        "MAPE": metrics_naive.get("MAPE"),
        "sMAPE": metrics_naive.get("sMAPE"),
        "median_APE": metrics_naive.get("median_APE"),
        "MASE": metrics_naive.get("MASE"),
        "TheilU1": metrics_naive.get("TheilU1"),
        "TheilU2": metrics_naive.get("TheilU2"),
        "DM_t": "",
        "DM_p": "",
        "DM_p_combined": "",
        "hit_80": "",
        "hit_95": "",
        "folds": "",
        "fold_horizon": "",
        "hash_SARIMA": "",
        "hash_naive": nv_hash,
    }
    _append_metrics_csv_row(metrics_csv_path, row_naive, header)

    # Comparable forecast plot (endog-only)
    fig, ax = plt.subplots()
    ax.plot(y_true.index, y_true.values, color="black", linewidth=1.5, label="actual")
    ax.plot(y_true.index, preds, color="tab:red", linestyle="--", label="SARIMA")
    ax.plot(y_true.index, naive_preds, color="tab:blue", linestyle="--", label="naive last")
    ax.set_ylabel("GDP" if target_transform == "level" else target_transform)
    ax.set_title("SARIMA vs naive predictions on endogenous data")
    ax.legend()
    plt.tight_layout()
    fig.savefig(figures_dir / "ComparisonForecast.png", dpi=300)
    plt.close(fig)
    return


def _infer_country_from_series_path(series_path: Path) -> str:
    """Infer a country code from a GDP CSV filename like 'gdp_US.csv'."""
    stem = series_path.stem
    return stem[4:] if stem.lower().startswith("gdp_") else (stem or "series")

# ------------------------
# Default-run orchestration and CLI
# ------------------------

# Ensure GDP series CSVs exist for specified countries via fetchers.fetch_gdp, return available list.
def ensure_series_csvs(base_dir: Path, countries: List[str], data_dir: Path) -> List[str]:
    ensure_dir(data_dir)
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

    available: List[str] = []
    for c in countries:
        p = data_dir / f"gdp_{c}.csv"
        if p.is_file():
            available.append(c)
        else:
            logger.warning("GDP CSV still missing for %s at %s", c, p)
    return available

# NEW: metrics CSV validation (warn-only) for schema and duplicates
def _validate_metrics_df(df: pd.DataFrame) -> bool:
    """
    Validate that metrics CSV has required fields and warn on duplicate (country, model) rows.

    Returns True if schema looks OK (even with duplicates), False if key fields are missing.
    """
    required = [
        "country",
        "model",
        "ME",
        "MAE",
        "RMSE",
        "MAPE",
        "sMAPE",
        "median_APE",
        "MASE",
        "TheilU1",
        "TheilU2",
        "DM_t",
        "DM_p",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning("metrics.csv missing required columns: %s", missing)
        return False

    try:
        # Check duplicates by (country, model)
        dups = (
            df.assign(_ord_=np.arange(len(df)))
            .groupby(["country", "model"], as_index=False)
            .agg(n=("model", "size"))
        )
        n_dup = int((dups["n"] > 1).sum())
        if n_dup > 0:
            logger.warning("metrics.csv has %d (country, model) groups with multiple rows; latest row will be used in summaries.", n_dup)
    except Exception as e:
        logger.debug("metrics.csv duplicate check failed: %s", e)

    # Duplicate-by-hash groups (detect identical forecast outputs across same (country, model))
    try:
        if {"hash_SARIMA", "hash_naive"}.issubset(set(df.columns)):
            dups_hash = (
                df.assign(_ord_=np.arange(len(df)))
                .groupby(["country", "model", "hash_SARIMA", "hash_naive"], as_index=False)
                .agg(n=("model", "size"))
            )
            n_dup_hash = int((dups_hash["n"] > 1).sum())
            if n_dup_hash > 0:
                logger.warning("metrics.csv has %d duplicate groups with identical (country, model, hash_SARIMA, hash_naive).", n_dup_hash)
    except Exception as e:
        logger.debug("metrics.csv duplicate-by-hash check failed: %s", e)

    # Optional fields for uncertainty (hit rates) are not required but recommended
    for opt in ["hit_80", "hit_95"]:
        if opt not in df.columns:
            logger.info("metrics.csv does not include optional column '%s' (hit rates); consider adding.", opt)

    return True

# Generate summary comparison figures from metrics.csv with extended metrics.
def generate_summary_figures(metrics_csv: Path, out_dir: Path) -> None:
    """Generate summary comparison figures from metrics.csv with extended metrics."""
    try:
        if not metrics_csv.exists() or metrics_csv.stat().st_size == 0:
            logger.warning("Metrics CSV missing or empty: %s; skipping summary figures.", metrics_csv)
            return
    except Exception:
        if not metrics_csv.exists():
            logger.warning("Metrics CSV not found: %s; skipping summary figures.", metrics_csv)
            return

    try:
        df = pd.read_csv(metrics_csv)
    except Exception as e:
        logger.warning("Failed to read metrics CSV %s: %s", metrics_csv, e)
        return

    if df.empty or "country" not in df.columns or "model" not in df.columns:
        logger.warning("Metrics CSV missing required data/columns; skipping summary figures.")
        return

    # NEW: validate schema and uniqueness expectations (warn-only)
    _validate_metrics_df(df)

    ensure_dir(out_dir)

    def _last_by_country_model(frm: pd.DataFrame) -> pd.DataFrame:
        d = frm.copy()
        d["_ord_"] = np.arange(len(d))
        return d.sort_values("_ord_").groupby(["country", "model"], as_index=False).tail(1)

    def _plot_grouped_metric(df_in: pd.DataFrame, metric: str, title: str, fname: str, ylabel: str) -> None:
        models = ["naive_last", "SARIMA(best)"]
        df_pairs = df_in[df_in["model"].isin(models)].copy()
        if df_pairs.empty or metric not in df_pairs.columns:
            return
        df_last = _last_by_country_model(df_pairs)
        df_last[metric] = pd.to_numeric(df_last[metric], errors="coerce")
        piv = df_last.pivot(index="country", columns="model", values=metric).sort_index()
        if piv.empty:
            return
        countries = list(piv.index)
        x = np.arange(len(countries))
        width = 0.38
        fig, ax = plt.subplots(figsize=(max(6, len(countries) * 1.2), 4))
        y_naive = piv.reindex(countries).get("naive_last")
        y_sar = piv.reindex(countries).get("SARIMA(best)")
        y_naive = y_naive.values if y_naive is not None else np.full(len(countries), np.nan)
        y_sar = y_sar.values if y_sar is not None else np.full(len(countries), np.nan)
        ax.bar(x - width / 2, y_naive, width=width, label="naive_last", color="tab:blue", alpha=0.75)
        ax.bar(x + width / 2, y_sar, width=width, label="SARIMA(best)", color="tab:red", alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels(countries, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        try:
            ymax = np.nanmax(np.concatenate([y_naive.astype(float), y_sar.astype(float)]))
            if np.isfinite(ymax):
                ax.set_ylim(0, ymax * 1.3)
        except Exception:
            pass
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=300)
        plt.close(fig)

    # Classic MAPE and RMSE
    _plot_grouped_metric(df, "MAPE", "MAPE by region (%)", "Summary_MAPE_by_region.png", "MAPE (%)")
    _plot_grouped_metric(df, "RMSE", "RMSE by region", "Summary_RMSE_by_region.png", "RMSE")
    # Extended summaries
    _plot_grouped_metric(df, "MAE", "MAE by region", "Summary_MAE_by_region.png", "MAE")
    _plot_grouped_metric(df, "MASE", "MASE by region", "Summary_MASE_by_region.png", "MASE")
    _plot_grouped_metric(df, "sMAPE", "sMAPE by region (%)", "Summary_sMAPE_by_region.png", "sMAPE (%)")
    _plot_grouped_metric(df, "median_APE", "Median APE by region (%)", "Summary_medianAPE_by_region.png", "Median APE (%)")

    # DM p-values (SARIMA(best) only)
    df_dm = df[df["model"] == "SARIMA(best)"].copy() if ("DM_p" in df.columns and "model" in df.columns) else pd.DataFrame()
    if not df_dm.empty and "DM_p" in df_dm.columns:
        df_dm["_ord_"] = np.arange(len(df_dm))
        df_last = df_dm.sort_values("_ord_").groupby("country", as_index=False).tail(1)
        df_last["DM_p"] = pd.to_numeric(df_last["DM_p"], errors="coerce")
        if not df_last.empty:
            df_last = df_last.sort_values("country")
            countries = df_last["country"].tolist()
            x = np.arange(len(countries))
            y = df_last["DM_p"].values
            fig, ax = plt.subplots(figsize=(max(6, len(countries) * 1.2), 4))
            ax.bar(x, y, width=0.5, color="tab:green", alpha=0.8, label="DM p-value (SARIMA vs naive)")
            ax.axhline(0.05, color="gray", linestyle="--", linewidth=1, label="p = 0.05")
            ax.set_xticks(x)
            ax.set_xticklabels(countries, rotation=45, ha="right")
            ax.set_ylabel("DM p-value (h=1)")
            ax.set_title("DM p-value by region for SARIMA(best)")
            ymax = np.nanmax(y.astype(float))
            if np.isfinite(ymax):
                ax.set_ylim(0, max(0.1, ymax * 1.15))
            ax.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "Summary_DMp_by_region.png", dpi=300)
            plt.close(fig)

    # NEW: Fold-aggregated DM (combined) summary for multi-fold runs
    df_dm_comb = df[df["model"] == "SARIMA(best)_multifold"].copy() if ("DM_p_combined" in df.columns and "model" in df.columns) else pd.DataFrame()
    if not df_dm_comb.empty:
        try:
            df_dm_comb["_ord_"] = np.arange(len(df_dm_comb))
            df_lastc = df_dm_comb.sort_values("_ord_").groupby("country", as_index=False).tail(1)
            df_lastc["DM_p_combined"] = pd.to_numeric(df_lastc["DM_p_combined"], errors="coerce")
            if not df_lastc.empty:
                df_lastc = df_lastc.sort_values("country")
                countries = df_lastc["country"].tolist()
                x = np.arange(len(countries))
                y = df_lastc["DM_p_combined"].values
                fig, ax = plt.subplots(figsize=(max(6, len(countries) * 1.2), 4))
                ax.bar(x, y, width=0.5, color="tab:orange", alpha=0.85, label="Combined DM p (multifold)")
                ax.axhline(0.05, color="gray", linestyle="--", linewidth=1, label="p = 0.05")
                ax.set_xticks(x)
                ax.set_xticklabels(countries, rotation=45, ha="right")
                ax.set_ylabel("Combined DM p-value")
                ax.set_title("Fold-aggregated DM p-values by region (multifold)")
                ymax = np.nanmax(y.astype(float))
                if np.isfinite(ymax):
                    ax.set_ylim(0, max(0.1, ymax * 1.15))
                ax.legend()
                plt.tight_layout()
                plt.savefig(out_dir / "Summary_DMpCombined_by_region.png", dpi=300)
                plt.close(fig)
        except Exception as e:
            logger.debug("Failed to render fold-aggregated DM summary: %s", e)

# ------------------------
# Default-run orchestration and CLI
# ------------------------

def default_run(base_dir: Path, figures_dir: Path, args: argparse.Namespace) -> None:
    """Run macro (figures/macro), ensure country CSVs, run endog-only for regions, and generate summaries."""
    ensure_dir(figures_dir)
    metrics_csv = figures_dir / "metrics.csv"
    script_path = Path(__file__).resolve()

    child_env = os.environ.copy()
    child_env["SKIP_DEFAULT_RUN"] = "1"

    # 1) Macro branch
    try:
        macro_dir = figures_dir / "macro"
        cmd = [
            sys.executable,
            str(script_path),
            "--figures-dir",
            str(macro_dir),
            "--log-level",
            str(args.log_level),
            "--metrics-csv",
            str(metrics_csv),
            "--data",
            str(base_dir / "data" / "us_macro_quarterly.csv"),
        ]
        subprocess.run(cmd, check=True, env=child_env)
    except Exception as e:
        logger.error("Macro subprocess failed in default-run: %s", e)

    # 2) Ensure GDP CSVs
    data_dir = base_dir / "data"
    countries = ["US", "EU27_2020", "CN"]
    available = ensure_series_csvs(base_dir, countries, data_dir)

    # 2b) Duplicate-series detection via SHA-1 of GDP vector
    try:
        series_hashes: dict[str, list[str]] = {}
        for c in available:
            p = data_dir / f"gdp_{c}.csv"
            df_c = pd.read_csv(p)
            g = pd.to_numeric(df_c.get("gdp", pd.Series(dtype=float)), errors="coerce")
            g = g[np.isfinite(g)]
            h = hash_forecast(g.values)
            series_hashes.setdefault(h, []).append(c)
        dup_groups = [v for v in series_hashes.values() if len(v) > 1]
        for grp in dup_groups:
            logger.warning("Duplicate GDP series detected across countries: %s", ", ".join(grp))
    except Exception as e:
        logger.debug("Duplicate-series hash detection failed: %s", e)

    # 3) Endog-only per available country (optionally with exog and multi-fold)
    for c in available:
        try:
            series_csv = data_dir / f"gdp_{c}.csv"
            fig_dir = figures_dir / c
            cmd = [
                sys.executable,
                str(script_path),
                "--series-csv",
                str(series_csv),
                "--figures-dir",
                str(fig_dir),
                "--log-level",
                str(args.log_level),
                "--metrics-csv",
                str(metrics_csv),
            ]
            if getattr(args, "default_run_exog", False):
                cmd.append("--use-exog")
                if c == "US":
                    if getattr(args, "exog_source_us", None):
                        cmd.extend(["--exog-source-US", str(args.exog_source_us)])
                    if getattr(args, "us_ism_url", None):
                        cmd.extend(["--us-ism-url", str(args.us_ism_url)])
                elif c in {"EU27_2020", "EA19"}:
                    if getattr(args, "exog_source_eu", None):
                        cmd.extend(["--exog-source-EU", str(args.exog_source_eu)])
                    if getattr(args, "eu_geo", None):
                        cmd.extend(["--eu-geo", str(args.eu_geo)])
                elif c == "CN":
                    if getattr(args, "exog_source_cn", None):
                        cmd.extend(["--exog-source-CN", str(args.exog_source_cn)])
                    if getattr(args, "exog_cn_csv", None):
                        cmd.extend(["--exog-cn-csv", str(args.exog_cn_csv)])
                    if getattr(args, "exog_cn_url", None):
                        cmd.extend(["--exog-cn-url", str(args.exog_cn_url)])
            if getattr(args, "multi_fold", False):
                cmd.append("--multi-fold")
                cmd.extend(["--folds", str(getattr(args, "folds", 3))])
                cmd.extend(["--fold-horizon", str(getattr(args, "fold_horizon", 8))])
            if getattr(args, "target_transform", None):
                cmd.extend(["--target-transform", str(args.target_transform)])
            if getattr(args, "intervals", None):
                cmd.extend(["--intervals", str(args.intervals)])
            if getattr(args, "p_range", None):
                cmd.extend(["--p-range", str(args.p_range)])
            if getattr(args, "q_range", None):
                cmd.extend(["--q-range", str(args.q_range)])
            if getattr(args, "P_range", None):
                cmd.extend(["--P-range", str(args.P_range)])
            if getattr(args, "Q_range", None):
                cmd.extend(["--Q-range", str(args.Q_range)])
            if getattr(args, "aic_cache", None):
                cmd.extend(["--aic-cache", str(args.aic_cache)])
            if getattr(args, "dm_method", None):
                cmd.extend(["--dm-method", str(args.dm_method)])
            subprocess.run(cmd, check=True, env=child_env)
        except Exception as e:
            logger.error("Endog subprocess failed for %s in default-run: %s", c, e)

    # 4) Summary figures
    try:
        generate_summary_figures(metrics_csv, figures_dir)
    except Exception as e:
        logger.error("Failed to generate summary figures: %s", e)


def main() -> None:
    """CLI entry point for the SARIMAX workflow."""
    
    # Initialize configuration system early
    _initialize_config()
    
    parser = argparse.ArgumentParser(description="SARIMAX modeling on US macroeconomic quarterly data (1959-2009).")
    
    # Data and output arguments
    parser.add_argument(
        "--data",
        type=str,
        default="data/us_macro_quarterly.csv",
        help="Path to CSV dataset. If missing, it will be created from statsmodels.macrodata.",
    )
    parser.add_argument("--figures-dir", type=str, default="figures", help="Directory to write figure files.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    parser.add_argument(
        "--series-csv",
        type=str,
        default=None,
        help="If provided, an endog-only GDP CSV with columns 'date' and 'gdp'. When specified, skips macro exogenous logic.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default=None,
        help="If provided, append evaluation metrics rows to this CSV (resolved relative to base_dir if not absolute).",
    )
    
    # Batch processing arguments  
    parser.add_argument("--batch-run", action="store_true", default=False, help="Iterate endog-only mode across --batch-countries.")
    parser.add_argument(
        "--batch-countries",
        type=str,
        default="US,EU27_2020,CN",
        help="Comma-separated list of countries for batch mode. Expects CSVs at data/gdp_{COUNTRY}.csv",
    )
    parser.add_argument(
        "--default-run",
        action="store_true",
        help="Run macro + endog batch (US, EU27_2020, CN), append metrics, and generate summary figures.",
    )
    
    # Default-run exogenous propagation
    parser.add_argument(
        "--default-run-exog",
        action="store_true",
        dest="default_run_exog",
        help="In default-run, propagate --use-exog and per-country exogenous flags to country subprocesses.",
    )
    
    # Exogenous source flags for per-country runs (now with config fallback)
    parser.add_argument("--use-exog", action="store_true", help="Enable exogenous regressors for country runs (PMI/ESI/NBS).")
    parser.add_argument("--exog-source-US", dest="exog_source_us", choices=["none", "ism_pmi"], default="ism_pmi")
    parser.add_argument("--us-ism-url", dest="us_ism_url", type=str, default=None, help="Override DB.NOMICS ISM PMI CSV URL.")
    parser.add_argument("--exog-source-EU", dest="exog_source_eu", choices=["none", "esi", "industry"], default="esi")
    parser.add_argument("--eu-geo", dest="eu_geo", choices=["EA19", "EU27_2020"], default="EA19", help="Euro area code for Eurostat BCS.")
    parser.add_argument("--exog-source-CN", dest="exog_source_cn", choices=["none", "nbs_csv", "nbs_dbn"], default="nbs_csv")
    parser.add_argument("--exog-cn-csv", dest="exog_cn_csv", type=str, default=None, help="Path to NBS PMI CSV for China.")
    parser.add_argument("--exog-cn-url", dest="exog_cn_url", type=str, default=None, help="DB.NOMICS CSV URL for NBS PMI (China).")

    # Target transform and intervals (now with config support)
    parser.add_argument("--target-transform", choices=["level", "qoq", "yoy", "log_diff"], default="level", help="Transform target series before modeling.")
    parser.add_argument("--intervals", type=str, default="80,95", help="Comma-separated predictive interval coverages (e.g., '80,95').")

    # Multi-fold backtesting controls (now with config support)
    parser.add_argument("--multi-fold", action="store_true", default=False, help="Enable multi-fold expanding-window backtest for endog-only mode.")
    parser.add_argument("--folds", type=int, default=None, help="Number of folds for multi-fold backtesting.")
    parser.add_argument("--fold-horizon", type=int, default=None, help="Test horizon (steps) per fold for multi-fold backtesting.")
    parser.add_argument("--dm-method", type=str, default="fisher", choices=["fisher", "stouffer"], help="Method to combine DM p-values across folds.")

    # Grid search controls with config support
    parser.add_argument("--p-range", type=str, default=None, help="Range or list for AR order p (e.g., '0-3' or '0,1,2'). Uses config default if not specified.")
    parser.add_argument("--q-range", type=str, default=None, help="Range or list for MA order q. Uses config default if not specified.")
    parser.add_argument("--P-range", type=str, default=None, help="Range or list for seasonal AR order P. Uses config default if not specified.")
    parser.add_argument("--Q-range", type=str, default=None, help="Range or list for seasonal MA order Q. Uses config default if not specified.")
    parser.add_argument("--aic-cache", type=str, default=None, help="Optional CSV path to save AIC grid results (relative to figures-dir if not absolute).")

    args = parser.parse_args()

    if not hasattr(args, "log_level"):
        args.log_level = "INFO"  # set default log level
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    # Gate warnings by log level (DEBUG -> show; otherwise silence non-critical noise)
    try:
        if args.log_level.upper() == "DEBUG":
            warnings.resetwarnings()
            warnings.filterwarnings("default")
        else:
            try:
                from statsmodels.tools.sm_exceptions import ConvergenceWarning  # type: ignore
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
            except Exception:
                pass
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
    except Exception:
        pass

    base_dir = Path(__file__).resolve().parent
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = base_dir / data_path
    figures_dir = Path(args.figures_dir)
    if not figures_dir.is_absolute():
        figures_dir = base_dir / figures_dir

    # Resolve metrics CSV path
    metrics_csv_path: Optional[Path] = None
    if args.metrics_csv:
        _m = Path(args.metrics_csv)
        metrics_csv_path = _m if _m.is_absolute() else (base_dir / _m)

    # Apply configuration defaults for backtesting parameters
    if args.folds is None:
        args.folds = _get_config_value('backtesting.cross_validation.n_splits', 3)
    if args.fold_horizon is None:
        args.fold_horizon = _get_config_value('backtesting.rolling_origin.forecast_horizon', 8)

    # Default composite run trigger
    if args.default_run or ((args.series_csv is None) and (not args.batch_run) and (os.environ.get("SKIP_DEFAULT_RUN") != "1")):
        default_run(base_dir, figures_dir, args)
        return

    # Batch mode (endog-only)
    if args.batch_run:
        countries = [c.strip() for c in (args.batch_countries or "").split(",") if c.strip()]
        if not countries:
            logger.warning("No valid countries provided to --batch-countries; nothing to run.")
            return
        for country in countries:
            series_path = base_dir / "data" / f"gdp_{country}.csv"
            fig_dir_country = figures_dir / country
            run_endog_only(series_path, fig_dir_country, metrics_csv_path, args)
        return

    # Endog-only CSV mode
    if args.series_csv:
        series_path = Path(args.series_csv)
        if not series_path.is_absolute():
            series_path = base_dir / series_path
        run_endog_only(series_path, figures_dir, metrics_csv_path, args)
        return


if __name__ == "__main__":
    main()
