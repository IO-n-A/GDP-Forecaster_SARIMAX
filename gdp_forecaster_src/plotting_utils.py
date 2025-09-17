# gdp_forecaster_src/plotting_utils.py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> None:
    """
    Create directory if it doesn't exist, including all parent directories.
    
    Parameters
    ----------
    path : Path
        Directory path to create
    """
    path.mkdir(parents=True, exist_ok=True)


def plot_realgdp(df: pd.DataFrame, out_path: Path) -> None:
    """
    Render and save the Real GDP series plot with proper time axis formatting.
    
    This function creates a time series plot of Real GDP data with appropriate
    axis labels and tick formatting for quarterly data spanning multiple decades.

    Parameters
    ----------
    df : pd.DataFrame
        Macro dataset containing the 'realgdp' column
    out_path : Path
        File path to save the rendered PNG (parents are created if missing)

    Returns
    -------
    None
        
    Notes
    -----
    The plot uses index-based x-axis with year tick labels for better readability.
    Assumes quarterly data starting from 1959.
    """
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots()
    ax.plot(df["realgdp"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Real GDP (k$)")
    n = len(df)
    ax.set_xticks(np.arange(0, n, 16))  # Every 4 years for quarterly data
    ax.set_xticklabels(np.arange(1959, 2010, 4))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_selected_columns(df: pd.DataFrame, out_path: Path) -> None:
    """
    Render and save a 3x2 panel of key macroeconomic series.
    
    This function creates a multi-panel plot showing six key macroeconomic indicators
    in a grid layout for quick visual inspection of the data.

    Parameters
    ----------
    df : pd.DataFrame
        Macro dataset with at least 8 columns; uses df.columns[2:8] to plot six series
        (skipping 'year' and 'quarter')
    out_path : Path
        File path to save the PNG (parents are created if missing)

    Returns
    -------
    None
        
    Notes
    -----
    Expects the dataframe to have 'year' and 'quarter' as the first two columns,
    followed by the series to be plotted.
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


def plot_forecast_comparison(y_true: pd.Series, 
                           forecasts: Dict[str, List[float]], 
                           out_path: Path,
                           title: str = "Forecast Comparison",
                           target_transform: str = "level") -> None:
    """
    Create a comparison plot of actual vs predicted values for multiple methods.
    
    This function generates a line plot comparing actual values against forecasts
    from different methods, useful for visual assessment of forecast accuracy.

    Parameters
    ----------
    y_true : pd.Series
        True values with datetime index
    forecasts : Dict[str, List[float]]
        Dictionary mapping method names to their forecast values
    out_path : Path
        Output file path for the plot
    title : str, default="Forecast Comparison"
        Plot title
    target_transform : str, default="level"
        Type of transformation applied to target (for y-axis label)

    Returns
    -------
    None
    """
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots()
    
    # Plot actual values
    ax.plot(y_true.index, y_true.values, color="black", linewidth=1.5, label="actual")
    
    # Plot forecasts from different methods
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
    for i, (method, values) in enumerate(forecasts.items()):
        color = colors[i % len(colors)]
        ax.plot(y_true.index, values, color=color, linestyle="--", label=method)
    
    ax.set_ylabel("GDP" if target_transform == "level" else target_transform)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_calibration_chart(hit_rates: Dict[int, float], 
                          out_path: Path,
                          title: str = "Prediction Interval Calibration") -> None:
    """
    Create a calibration plot showing empirical vs nominal coverage rates.
    
    This function generates a bar chart comparing requested confidence levels
    with their empirical coverage rates to assess interval forecast quality.

    Parameters
    ----------
    hit_rates : Dict[int, float]
        Dictionary mapping confidence levels (%) to empirical hit rates (0-1)
    out_path : Path
        Output file path for the plot
    title : str, default="Prediction Interval Calibration"
        Plot title

    Returns
    -------
    None
    """
    if not hit_rates:
        logger.warning("No hit rates provided for calibration plot")
        return
        
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots()
    
    lvls = sorted(hit_rates.keys())
    vals = [hit_rates[l] for l in lvls]
    
    ax.bar([f"{l}%" for l in lvls], vals, color="tab:purple", alpha=0.85)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Empirical coverage")
    ax.set_title(title)
    
    # Add value labels on bars
    for i, v in enumerate(vals):
        ax.text(i, min(0.98, v + 0.02), f"{v:.2f}", ha="center", fontsize=8)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_metric_comparison(df: pd.DataFrame, 
                          metric: str, 
                          out_path: Path,
                          title: str = None,
                          ylabel: str = None) -> None:
    """
    Create a grouped bar chart comparing a metric across countries and methods.
    
    This function generates comparison plots for evaluation metrics, typically
    used in summary analysis to compare forecast performance across regions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ['country', 'model', metric]
    metric : str
        Name of the metric column to plot
    out_path : Path
        Output file path for the plot
    title : str, optional
        Plot title (auto-generated if None)
    ylabel : str, optional
        Y-axis label (defaults to metric name)

    Returns
    -------
    None
    """
    if df.empty or metric not in df.columns:
        logger.warning("Cannot create metric comparison: missing data or metric column")
        return
    
    ensure_dir(out_path.parent)
    
    # Prepare data for plotting
    models = ["naive_last", "SARIMA(best)"]
    df_pairs = df[df["model"].isin(models)].copy()
    
    if df_pairs.empty:
        logger.warning("No data found for standard model comparison")
        return
    
    df_pairs[metric] = pd.to_numeric(df_pairs[metric], errors="coerce")
    piv = df_pairs.pivot(index="country", columns="model", values=metric).sort_index()
    
    if piv.empty:
        logger.warning("No valid data for pivot table")
        return
    
    countries = list(piv.index)
    x = np.arange(len(countries))
    width = 0.38
    
    fig, ax = plt.subplots(figsize=(max(6, len(countries) * 1.2), 4))
    
    y_naive = piv.get("naive_last", pd.Series(index=countries, dtype=float))
    y_sar = piv.get("SARIMA(best)", pd.Series(index=countries, dtype=float))
    
    y_naive = y_naive.values if y_naive is not None else np.full(len(countries), np.nan)
    y_sar = y_sar.values if y_sar is not None else np.full(len(countries), np.nan)
    
    ax.bar(x - width / 2, y_naive, width=width, label="naive_last", color="tab:blue", alpha=0.75)
    ax.bar(x + width / 2, y_sar, width=width, label="SARIMA(best)", color="tab:red", alpha=0.75)
    
    ax.set_xticks(x)
    ax.set_xticklabels(countries, rotation=45, ha="right")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or f"{metric} by region")
    
    # Set y-axis limits
    try:
        ymax = np.nanmax(np.concatenate([y_naive.astype(float), y_sar.astype(float)]))
        if np.isfinite(ymax):
            ax.set_ylim(0, ymax * 1.3)
    except Exception:
        pass
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_dm_pvalues(df: pd.DataFrame, 
                   out_path: Path,
                   title: str = "DM p-values by region",
                   combined: bool = False) -> None:
    """
    Create a bar chart of Diebold-Mariano p-values across countries.
    
    This function visualizes the statistical significance of forecast accuracy
    differences using DM test results.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing DM test results
    out_path : Path
        Output file path for the plot
    title : str, default="DM p-values by region"
        Plot title
    combined : bool, default=False
        Whether to use combined p-values from multi-fold analysis

    Returns
    -------
    None
    """
    p_col = "DM_p_combined" if combined else "DM_p"
    model_filter = "SARIMA(best)_multifold" if combined else "SARIMA(best)"
    
    df_dm = df[df["model"] == model_filter].copy() if p_col in df.columns else pd.DataFrame()
    
    if df_dm.empty:
        logger.warning("No DM test data found for plotting")
        return
    
    ensure_dir(out_path.parent)
    
    df_dm[p_col] = pd.to_numeric(df_dm[p_col], errors="coerce")
    df_last = df_dm.sort_values("country")
    
    countries = df_last["country"].tolist()
    x = np.arange(len(countries))
    y = df_last[p_col].values
    
    fig, ax = plt.subplots(figsize=(max(6, len(countries) * 1.2), 4))
    
    color = "tab:orange" if combined else "tab:green"
    label = "Combined DM p (multifold)" if combined else "DM p-value (SARIMA vs naive)"
    
    ax.bar(x, y, width=0.5, color=color, alpha=0.8, label=label)
    ax.axhline(0.05, color="gray", linestyle="--", linewidth=1, label="p = 0.05")
    
    ax.set_xticks(x)
    ax.set_xticklabels(countries, rotation=45, ha="right")
    ax.set_ylabel("DM p-value")
    ax.set_title(title)
    
    ymax = np.nanmax(y.astype(float))
    if np.isfinite(ymax):
        ax.set_ylim(0, max(0.1, ymax * 1.15))
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def generate_summary_figures(metrics_csv: Path, out_dir: Path) -> None:
    """
    Generate a comprehensive set of summary comparison figures from metrics CSV.
    
    This function creates multiple visualization comparing forecast performance
    across countries and methods, providing a dashboard-style overview.

    Parameters
    ----------
    metrics_csv : Path
        Path to CSV file containing evaluation metrics
    out_dir : Path
        Output directory for generated figures

    Returns
    -------
    None
        
    Notes
    -----
    Generates plots for MAPE, RMSE, MAE, MASE, sMAPE, median APE, and DM tests.
    Handles missing data gracefully and validates CSV structure.
    """
    try:
        if not metrics_csv.exists() or metrics_csv.stat().st_size == 0:
            logger.warning("Metrics CSV missing or empty: %s; skipping summary figures.", metrics_csv)
            return
    except Exception:
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

    ensure_dir(out_dir)

    # Helper function to get latest results per country-model combination
    def get_latest_by_country_model(frm: pd.DataFrame) -> pd.DataFrame:
        d = frm.copy()
        d["_ord_"] = np.arange(len(d))
        return d.sort_values("_ord_").groupby(["country", "model"], as_index=False).tail(1)

    # Generate metric comparison plots
    metrics_to_plot = [
        ("MAPE", "MAPE by region (%)", "Summary_MAPE_by_region.png", "MAPE (%)"),
        ("RMSE", "RMSE by region", "Summary_RMSE_by_region.png", "RMSE"),
        ("MAE", "MAE by region", "Summary_MAE_by_region.png", "MAE"),
        ("MASE", "MASE by region", "Summary_MASE_by_region.png", "MASE"),
        ("sMAPE", "sMAPE by region (%)", "Summary_sMAPE_by_region.png", "sMAPE (%)"),
        ("median_APE", "Median APE by region (%)", "Summary_medianAPE_by_region.png", "Median APE (%)")
    ]
    
    for metric, title, filename, ylabel in metrics_to_plot:
        try:
            df_latest = get_latest_by_country_model(df)
            plot_metric_comparison(df_latest, metric, out_dir / filename, title, ylabel)
        except Exception as e:
            logger.debug("Failed to generate %s: %s", filename, e)

    # DM p-value plots
    try:
        plot_dm_pvalues(df, out_dir / "Summary_DMp_by_region.png", 
                       "DM p-value by region for SARIMA(best)", combined=False)
    except Exception as e:
        logger.debug("Failed to generate DM p-value plot: %s", e)

    try:
        plot_dm_pvalues(df, out_dir / "Summary_DMpCombined_by_region.png",
                       "Fold-aggregated DM p-values by region (multifold)", combined=True)
    except Exception as e:
        logger.debug("Failed to generate combined DM p-value plot: %s", e)

    logger.info("Summary figures generated in %s", out_dir)