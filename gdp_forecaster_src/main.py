# gdp_forecaster_src/main.py

"""
SARIMAX modeling and evaluation on US macroeconomic quarterly data (1959–2009).

This is the main entry point for the refactored GDP forecasting system.
The original monolithic forecaster_SARIMAX.py has been split into multiple
focused utility modules for better maintainability and code organization.

Purpose
-------
- Load macro dataset (from CSV if provided, else statsmodels.macrodata)
- Visualize core series and run stationarity checks (ADF on levels and first differences)
- Grid-search SARIMAX orders by AIC over (p, q, P, Q) with fixed d=1, D=0, s=4 (quarterly seasonality)
- Fit selected configuration, save diagnostics and comparison figures to the figures/ directory
- Rolling-origin one-step forecasts vs a naive last-value baseline; export evaluation metrics

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
"""

import argparse
import logging
import os
import subprocess
import sys
import warnings
from itertools import product
from pathlib import Path
from typing import Optional

# Import utility modules
from .config_utils import initialize_config, get_config_value
from .data_utils import (
    load_macro_data, load_gdp_series_csv, infer_country_from_series_path, ensure_series_csvs
)
from .parsing_utils import (
    parse_range_arg, parse_intervals_arg, parse_batch_countries,
    validate_target_transform, validate_log_level
)
from .transform_utils import apply_target_transform
from .metrics_utils import compute_metrics, combine_dm_pvalues
from .plotting_utils import (
    plot_realgdp, plot_selected_columns, plot_forecast_comparison,
    plot_calibration_chart, generate_summary_figures
)
from .diagnostics_utils import save_residual_diagnostics
from .forecasting_utils import (
    adf_test, optimize_sarimax, rolling_one_step_predictions, hash_forecast,
    build_exog_for_country, run_multifold_evaluation, fit_best_sarimax_model,
    validate_sarimax_inputs
)
from .file_utils import (
    ensure_dir, append_metrics_csv_row, validate_metrics_df, safe_read_csv
)

logger = logging.getLogger(__name__)


def run_macro_workflow(data_path: Path, figures_dir: Path, metrics_csv_path: Optional[Path], args: argparse.Namespace) -> None:
    """
    Execute the traditional macro workflow with full dataset and exogenous variables.
    
    This function runs the original macro analysis workflow using the US macroeconomic
    dataset with visualization and model fitting on the full time series.
    
    Parameters
    ----------
    data_path : Path
        Path to macro dataset CSV
    figures_dir : Path
        Output directory for figures
    metrics_csv_path : Optional[Path]
        Path to metrics CSV file for results
    args : argparse.Namespace
        CLI arguments
    """
    logger.info("Starting macro workflow with data: %s", data_path)
    
    # Load macro data
    df = load_macro_data(data_path)
    logger.info("Loaded macro dataset with %d observations", len(df))
    
    # Generate visualizations
    ensure_dir(figures_dir)
    plot_realgdp(df, figures_dir / "RealGDP.png")
    plot_selected_columns(df, figures_dir / "MacroPanel.png")
    
    # ADF tests for stationarity
    gdp_series = df["realgdp"]
    adf_stat, adf_pval = adf_test(gdp_series)
    logger.info("ADF test on Real GDP: statistic=%.3f, p-value=%.3f", adf_stat, adf_pval)
    
    adf_stat_diff, adf_pval_diff = adf_test(gdp_series.diff().dropna())
    logger.info("ADF test on Real GDP (first diff): statistic=%.3f, p-value=%.3f", adf_stat_diff, adf_pval_diff)
    
    # For macro workflow, we could extend here with full SARIMAX modeling
    # For now, this demonstrates the modular structure
    logger.info("Macro workflow completed successfully")


def run_endog_only_workflow(series_path: Path, figures_dir: Path, metrics_csv_path: Optional[Path], args: Optional[argparse.Namespace] = None) -> None:
    """
    Execute an endog-only SARIMA workflow for a single GDP CSV.
    
    This is the main modeling workflow that loads a GDP series, performs grid search
    for optimal SARIMAX parameters, and evaluates forecast performance.

    Parameters
    ----------
    series_path : Path
        Input CSV with columns ['date', 'gdp'] (date parseable to datetime)
    figures_dir : Path
        Output directory for endog-only figures (created if missing)
    metrics_csv_path : Optional[Path]
        If provided, append evaluation metrics to this CSV
    args : Optional[argparse.Namespace]
        CLI arguments for configuration

    Returns
    -------
    None

    Workflow
    --------
    - Load and validate the GDP series
    - Grid search SARIMA over (p, q, P, Q) ∈ [0..3]^4 with d=1, D=0, s=4 (quarterly)
    - Fit best model by AIC; save diagnostics
    - Rolling origin one-step evaluation on the last 16 quarters (adaptive for short series)
    - Compare SARIMA vs naive last-value; save overlay, APE curves, MAPE bar chart
    - Append a metrics row for SARIMA and the naive baseline (m=4 for MASE scaling)
    """
    logger.info("Starting endog-only workflow for: %s", series_path)
    
    # Load and validate GDP series
    endog_raw = load_gdp_series_csv(series_path)
    
    # Apply target transformation
    target_transform = getattr(args, "target_transform", "level") if args is not None else "level"
    validate_target_transform(target_transform)
    endog, d_for_transform = apply_target_transform(endog_raw, target_transform)
    logger.info("Target transform: %s; resulting length=%d", target_transform, len(endog))
    
    # Drop NaNs introduced by transform
    _len_before = len(endog)
    endog = endog.dropna()
    if len(endog) < _len_before:
        logger.info("Dropped %d NaNs from transformed target after '%s'.", _len_before - len(endog), target_transform)
    
    # Optional: attach exogenous series if requested
    country = infer_country_from_series_path(series_path)
    exog_model: Optional = None
    if args is not None and getattr(args, "use_exog", False):
        exog_q = build_exog_for_country(country, args, endog.index)
        if exog_q is not None:
            import pandas as pd
            df_joint = pd.concat([endog, exog_q["EXOG"]], axis=1).dropna()
            if not df_joint.empty:
                endog = df_joint.iloc[:, 0]
                exog_model = df_joint[["EXOG"]]
                logger.info("Using exogenous series for %s (aligned rows=%d).", country, len(endog))
            else:
                logger.info("Exogenous series for %s had no overlap with GDP; proceeding endog-only.", country)
        else:
            logger.info("No exogenous series available for %s; proceeding endog-only.", country)
    
    # Validate inputs
    validate_sarimax_inputs(endog, exog_model)
    
    # Hyperparameter grid for SARIMA
    pL = parse_range_arg(getattr(args, "p_range", None) if args is not None else None, 
                        config_key="model.search_space.p_range", args=args)
    qL = parse_range_arg(getattr(args, "q_range", None) if args is not None else None,
                        config_key="model.search_space.q_range", args=args)
    PL = parse_range_arg(getattr(args, "P_range", None) if args is not None else None,
                        config_key="model.search_space.P_range", args=args)
    QL = parse_range_arg(getattr(args, "Q_range", None) if args is not None else None,
                        config_key="model.search_space.Q_range", args=args)
    
    # Get fixed parameters from configuration
    d = d_for_transform  # Use transform-specific differencing
    D = get_config_value('model.fixed_parameters.D', 0)
    s = get_config_value('model.fixed_parameters.s', 4)
    
    parameters_list = list(product(pL, qL, PL, QL))
    
    # Grid search (AIC-ranked)
    result_df = optimize_sarimax(endog, exog_model, parameters_list, d, D, s)
    if result_df.empty:
        raise SystemExit("Grid search failed: no SARIMA configuration could be fit.")
    logger.info("Top 5 models by AIC:\n%s", result_df.head().to_string())
    
    # Save AIC cache if requested
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
    
    # Fit best model
    best_order = result_df.iloc[0]["(p,q,P,Q)"]
    logger.info("Selected (p,q,P,Q)=%s with AIC=%.3f", best_order, float(result_df.iloc[0]["AIC"]))
    
    # SARIMAX model fitting with robust error support
    fit_kwargs = {"disp": False}
    robust_config = get_config_value('model.robust_errors', {})
    if robust_config.get('enabled', False):
        fit_kwargs['cov_type'] = robust_config.get('cov_type', 'robust')
        cov_kwds = robust_config.get('cov_kwds', {})
        if cov_kwds:
            fit_kwargs['cov_kwds'] = cov_kwds
        logger.info("Using robust standard errors: %s", fit_kwargs.get('cov_type'))
    
    best_res = fit_best_sarimax_model(endog, exog_model, best_order, d, D, s, fit_kwargs)
    logger.info("Model fitted successfully")
    
    # Diagnostics plot
    ensure_dir(figures_dir)
    fig = best_res.plot_diagnostics(figsize=(10, 8))
    fig.savefig(figures_dir / "Diagnostics.png", dpi=300)
    import matplotlib.pyplot as plt
    plt.close(fig)
    logger.info("Saved diagnostics: %s", figures_dir / "Diagnostics.png")
    
    # Additional residual diagnostics
    save_residual_diagnostics(best_res.resid, figures_dir, fname_prefix="Endog_Residuals")
    
    # Multi-fold evaluation if requested
    if args is not None and getattr(args, "multi_fold", False):
        folds = get_config_value('backtesting.cross_validation.n_splits', 3, args, 'folds')
        fold_horizon = get_config_value('backtesting.rolling_origin.forecast_horizon', 8, args, 'fold_horizon')
        
        multifold_results = run_multifold_evaluation(
            endog=endog,
            exog_model=exog_model,
            best_order=best_order,
            d=d, D=D, s=s,
            folds=folds,
            fold_horizon=fold_horizon
        )
        
        # Export multifold metrics
        _export_multifold_metrics(multifold_results, country, metrics_csv_path)
        return
    
    # Standard rolling-origin evaluation
    TEST_LEN_DEFAULT = 16
    if len(endog) <= TEST_LEN_DEFAULT + 8:
        test_len = max(1, min(TEST_LEN_DEFAULT, len(endog) // 4))
        logger.warning("Short series detected. Using test_len=%d (default was %d).", test_len, TEST_LEN_DEFAULT)
    else:
        test_len = TEST_LEN_DEFAULT
    train_len = len(endog) - test_len
    
    # Collect one-step-ahead predictions and confidence intervals
    coverage_levels = parse_intervals_arg(getattr(args, "intervals", None) if args is not None else None)
    preds, conf_bounds = rolling_one_step_predictions(
        endog=endog,
        exog_model=exog_model,
        best_order=best_order,
        d=d, D=D, s=s,
        coverage_levels=coverage_levels,
        start_index=train_len,
    )
    
    # Generate forecasts and evaluate
    y_true = endog.iloc[train_len:]
    naive_preds = [endog.iloc[i - 1] for i in range(train_len, len(endog))]
    
    # Calculate hit rates for prediction intervals
    hits = _calculate_hit_rates(y_true, conf_bounds, coverage_levels)
    
    # Generate calibration plot
    if hits:
        plot_calibration_chart(hits, figures_dir / "Calibration_endog.png", 
                             "Prediction interval calibration (endog-only)")
    
    # Export metrics
    _export_standard_metrics(y_true, preds, naive_preds, endog.iloc[:train_len], 
                           country, metrics_csv_path, hits)
    
    # Generate forecast comparison plot
    forecasts = {"SARIMA": preds, "naive last": naive_preds}
    plot_forecast_comparison(y_true, forecasts, figures_dir / "ComparisonForecast.png",
                           "SARIMA vs naive predictions on endogenous data", target_transform)
    
    logger.info("Endog-only workflow completed successfully")


def _calculate_hit_rates(y_true, conf_bounds, coverage_levels):
    """Calculate hit rates for prediction intervals."""
    import numpy as np
    hits = {}
    try:
        arr_y = np.asarray(y_true.values, dtype=float)
        for lvl in coverage_levels:
            lo_arr = np.asarray(conf_bounds[lvl][0])
            hi_arr = np.asarray(conf_bounds[lvl][1])
            hits[lvl] = float(np.mean((arr_y >= lo_arr) & (arr_y <= hi_arr)))
        logger.info("Interval hit-rates: %s", ", ".join(f"{lvl}%={hits[lvl]:.3f}" for lvl in sorted(hits)))
    except Exception:
        pass
    return hits


def _export_standard_metrics(y_true, preds, naive_preds, y_train, country, metrics_csv_path, hits):
    """Export standard evaluation metrics to CSV."""
    header = [
        "mode", "country", "n", "train_len", "test_len", "model",
        "ME", "MAE", "RMSE", "MAPE", "sMAPE", "median_APE", "MASE",
        "TheilU1", "TheilU2", "DM_t", "DM_p", "DM_p_combined",
        "hit_80", "hit_95", "folds", "fold_horizon", "hash_SARIMA", "hash_naive"
    ]
    
    n_obs = len(y_true)
    train_len = len(y_train)
    
    # Compute metrics
    metrics_model = compute_metrics(
        y_true=y_true, y_hat=preds, y_hat_naive=naive_preds, y_train=y_train, m=4
    )
    
    # Generate forecast hashes
    try:
        sar_hash = hash_forecast(preds)
        nv_hash = hash_forecast(naive_preds)
    except Exception:
        sar_hash = ""
        nv_hash = ""
    
    # SARIMA model row
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
    append_metrics_csv_row(metrics_csv_path, row_model, header)
    
    # Naive model row
    metrics_naive = compute_metrics(
        y_true=y_true, y_hat=naive_preds, y_hat_naive=naive_preds, y_train=y_train, m=4
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
    append_metrics_csv_row(metrics_csv_path, row_naive, header)


def _export_multifold_metrics(multifold_results, country, metrics_csv_path):
    """Export multifold evaluation metrics to CSV."""
    header = [
        "mode", "country", "n", "train_len", "test_len", "model",
        "ME", "MAE", "RMSE", "MAPE", "sMAPE", "median_APE", "MASE",
        "TheilU1", "TheilU2", "DM_t", "DM_p", "DM_p_combined",
        "hit_80", "hit_95", "folds", "fold_horizon", "hash_SARIMA", "hash_naive"
    ]
    
    all_true = multifold_results["true_values"]
    all_pred = multifold_results["predictions"]
    all_naive = multifold_results["naive_predictions"]
    
    # Aggregate metrics
    agg = compute_metrics(
        y_true=all_true, y_hat=all_pred, y_hat_naive=all_naive, 
        y_train=all_true[:multifold_results["train_size"]], m=4
    )
    
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
        "train_len": multifold_results["train_size"],
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
        "DM_p_combined": multifold_results["combined_dm_pvalue"],
        "hit_80": "",
        "hit_95": "",
        "folds": multifold_results["n_folds"],
        "fold_horizon": multifold_results["fold_horizon"],
        "hash_SARIMA": sar_hash,
        "hash_naive": nv_hash,
    }
    append_metrics_csv_row(metrics_csv_path, row, header)


def run_default_workflow(base_dir: Path, figures_dir: Path, args: argparse.Namespace) -> None:
    """
    Run the default composite workflow: macro + endog batch + summary figures.
    
    This function orchestrates the complete analysis pipeline including macro analysis,
    country-specific endog-only modeling, and summary figure generation.
    
    Parameters
    ----------
    base_dir : Path
        Base project directory
    figures_dir : Path
        Output directory for all figures
    args : argparse.Namespace
        CLI arguments
    """
    ensure_dir(figures_dir)
    metrics_csv = figures_dir / "metrics.csv"
    script_path = Path(__file__).resolve()
    
    child_env = os.environ.copy()
    child_env["SKIP_DEFAULT_RUN"] = "1"
    
    # 1) Macro branch
    try:
        macro_dir = figures_dir / "macro"
        run_macro_workflow(
            base_dir / "data" / "us_macro_quarterly.csv",
            macro_dir,
            metrics_csv,
            args
        )
    except Exception as e:
        logger.error("Macro workflow failed in default-run: %s", e)
    
    # 2) Ensure GDP CSVs and run endog-only per available country
    data_dir = base_dir / "data"
    countries = ["US", "EU27_2020", "CN"]
    available = ensure_series_csvs(base_dir, countries, data_dir)
    
    for c in available:
        try:
            series_csv = data_dir / f"gdp_{c}.csv"
            fig_dir = figures_dir / c
            run_endog_only_workflow(series_csv, fig_dir, metrics_csv, args)
        except Exception as e:
            logger.error("Endog workflow failed for %s in default-run: %s", c, e)
    
    # 3) Summary figures
    try:
        generate_summary_figures(metrics_csv, figures_dir)
    except Exception as e:
        logger.error("Failed to generate summary figures: %s", e)


def setup_cli_parser() -> argparse.ArgumentParser:
    """
    Set up the command-line argument parser.
    
    This function defines all CLI arguments and their configuration,
    maintaining compatibility with the original interface.
    
    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="SARIMAX modeling on US macroeconomic quarterly data (1959-2009)."
    )
    
    # Data and output arguments
    parser.add_argument(
        "--data", type=str, default="data/us_macro_quarterly.csv",
        help="Path to CSV dataset. If missing, it will be created from statsmodels.macrodata."
    )
    parser.add_argument(
        "--figures-dir", type=str, default="figures",
        help="Directory to write figure files."
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level."
    )
    parser.add_argument(
        "--series-csv", type=str, default=None,
        help="If provided, an endog-only GDP CSV with columns 'date' and 'gdp'. When specified, skips macro exogenous logic."
    )
    parser.add_argument(
        "--metrics-csv", type=str, default=None,
        help="If provided, append evaluation metrics rows to this CSV (resolved relative to base_dir if not absolute)."
    )
    
    # Batch processing arguments  
    parser.add_argument(
        "--batch-run", action="store_true", default=False,
        help="Iterate endog-only mode across --batch-countries."
    )
    parser.add_argument(
        "--batch-countries", type=str, default="US,EU27_2020,CN",
        help="Comma-separated list of countries for batch mode. Expects CSVs at data/gdp_{COUNTRY}.csv"
    )
    parser.add_argument(
        "--default-run", action="store_true",
        help="Run macro + endog batch (US, EU27_2020, CN), append metrics, and generate summary figures."
    )
    
    # Exogenous source flags
    parser.add_argument(
        "--use-exog", action="store_true",
        help="Enable exogenous regressors for country runs (PMI/ESI/NBS)."
    )
    parser.add_argument(
        "--exog-source-US", dest="exog_source_us", choices=["none", "ism_pmi"], default="ism_pmi"
    )
    parser.add_argument(
        "--us-ism-url", dest="us_ism_url", type=str, default=None,
        help="Override DB.NOMICS ISM PMI CSV URL."
    )
    parser.add_argument(
        "--exog-source-EU", dest="exog_source_eu", choices=["none", "esi", "industry"], default="esi"
    )
    parser.add_argument(
        "--eu-geo", dest="eu_geo", choices=["EA19", "EU27_2020"], default="EA19",
        help="Euro area code for Eurostat BCS."
    )
    parser.add_argument(
        "--exog-source-CN", dest="exog_source_cn", choices=["none", "nbs_csv", "nbs_dbn"], default="nbs_csv"
    )
    parser.add_argument(
        "--exog-cn-csv", dest="exog_cn_csv", type=str, default=None,
        help="Path to NBS PMI CSV for China."
    )
    parser.add_argument(
        "--exog-cn-url", dest="exog_cn_url", type=str, default=None,
        help="DB.NOMICS CSV URL for NBS PMI (China)."
    )

    # Target transform and intervals
    parser.add_argument(
        "--target-transform", choices=["level", "qoq", "yoy", "log_diff"], default="level",
        help="Transform target series before modeling."
    )
    parser.add_argument(
        "--intervals", type=str, default="80,95",
        help="Comma-separated predictive interval coverages (e.g., '80,95')."
    )

    # Multi-fold backtesting controls
    parser.add_argument(
        "--multi-fold", action="store_true", default=False,
        help="Enable multi-fold expanding-window backtest for endog-only mode."
    )
    parser.add_argument(
        "--folds", type=int, default=None,
        help="Number of folds for multi-fold backtesting."
    )
    parser.add_argument(
        "--fold-horizon", type=int, default=None,
        help="Test horizon (steps) per fold for multi-fold backtesting."
    )
    parser.add_argument(
        "--dm-method", type=str, default="fisher", choices=["fisher", "stouffer"],
        help="Method to combine DM p-values across folds."
    )

    # Grid search controls
    parser.add_argument(
        "--p-range", type=str, default=None,
        help="Range or list for AR order p (e.g., '0-3' or '0,1,2'). Uses config default if not specified."
    )
    parser.add_argument(
        "--q-range", type=str, default=None,
        help="Range or list for MA order q. Uses config default if not specified."
    )
    parser.add_argument(
        "--P-range", type=str, default=None,
        help="Range or list for seasonal AR order P. Uses config default if not specified."
    )
    parser.add_argument(
        "--Q-range", type=str, default=None,
        help="Range or list for seasonal MA order Q. Uses config default if not specified."
    )
    parser.add_argument(
        "--aic-cache", type=str, default=None,
        help="Optional CSV path to save AIC grid results (relative to figures-dir if not absolute)."
    )

    return parser


def setup_logging(log_level: str) -> None:
    """
    Configure logging with specified level and warning filters.
    
    Parameters
    ----------
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    level = validate_log_level(log_level)
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Configure warnings based on log level
    if level == "DEBUG":
        warnings.resetwarnings()
        warnings.filterwarnings("default")
    else:
        try:
            from statsmodels.tools.sm_exceptions import ConvergenceWarning
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
        except Exception:
            pass
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")


def main() -> None:
    """
    Main entry point for the SARIMAX forecasting application.
    
    This function orchestrates the entire workflow based on CLI arguments,
    supporting various execution modes including single series analysis,
    batch processing, and comprehensive default runs.
    """
    # Initialize configuration system early
    initialize_config()
    
    # Parse CLI arguments
    parser = setup_cli_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Resolve paths
    base_dir = Path(__file__).resolve().parent.parent  # Go up from gdp_forecaster_src to project root
    
    from .file_utils import resolve_path
    data_path = resolve_path(args.data, base_dir)
    figures_dir = resolve_path(args.figures_dir, base_dir)
    
    metrics_csv_path: Optional[Path] = None
    if args.metrics_csv:
        metrics_csv_path = resolve_path(args.metrics_csv, base_dir)
    
    # Apply configuration defaults for backtesting parameters
    if args.folds is None:
        args.folds = get_config_value('backtesting.cross_validation.n_splits', 3)
    if args.fold_horizon is None:
        args.fold_horizon = get_config_value('backtesting.rolling_origin.forecast_horizon', 8)
    
    # Determine execution mode
    if args.default_run or ((args.series_csv is None) and (not args.batch_run) and (os.environ.get("SKIP_DEFAULT_RUN") != "1")):
        run_default_workflow(base_dir, figures_dir, args)
        return
    
    # Batch mode (endog-only)
    if args.batch_run:
        countries = parse_batch_countries(args.batch_countries)
        if not countries:
            logger.warning("No valid countries provided to --batch-countries; nothing to run.")
            return
        for country in countries:
            series_path = base_dir / "data" / f"gdp_{country}.csv"
            fig_dir_country = figures_dir / country
            run_endog_only_workflow(series_path, fig_dir_country, metrics_csv_path, args)
        return
    
    # Endog-only CSV mode
    if args.series_csv:
        series_path = resolve_path(args.series_csv, base_dir)
        run_endog_only_workflow(series_path, figures_dir, metrics_csv_path, args)
        return
    
    # Default to macro workflow
    run_macro_workflow(data_path, figures_dir, metrics_csv_path, args)


if __name__ == "__main__":
    main()