# gdp_forecaster_src/forecasting_utils.py

import hashlib
import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union
from tqdm.auto import tqdm
import logging

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


def adf_test(series: Union[pd.Series, np.ndarray]) -> Tuple[float, float]:
    """
    Run the Augmented Dickey-Fuller (ADF) test for unit roots.
    
    This function performs the ADF test to assess whether a time series is stationary
    or contains a unit root, which is crucial for proper model specification.

    Parameters
    ----------
    series : Union[pd.Series, np.ndarray]
        Input series. NaNs are dropped prior to testing.

    Returns
    -------
    Tuple[float, float]
        (test_statistic, p_value)

    Notes
    -----
    - ADF null hypothesis: the series has a unit root (non-stationary)
    - We typically test both the level and first difference to assess stationarity
    - Lower p-values (< 0.05) suggest rejection of null (series is stationary)
    """
    res = adfuller(pd.Series(series).dropna())
    return res[0], res[1]


def optimize_sarimax(endog: Union[pd.Series, list], 
                    exog: Union[pd.DataFrame, list, None], 
                    order_list: List[Tuple], 
                    d: int, 
                    D: int, 
                    s: int) -> pd.DataFrame:
    """
    Grid-search SARIMAX hyperparameters and rank by AIC.
    
    This function performs an exhaustive grid search over specified SARIMAX parameter
    combinations to find the model with the best AIC score.

    Parameters
    ----------
    endog : Union[pd.Series, list]
        Endogenous (target) series
    exog : Union[pd.DataFrame, list, None]
        Optional exogenous regressors aligned with endog (may be None for endog-only)
    order_list : List[Tuple]
        List of (p, q, P, Q) tuples. Differencing orders d, D, and seasonal period s are fixed
    d : int
        Non-seasonal differencing order (typical usage: 1)
    D : int
        Seasonal differencing order (typical usage: 0)
    s : int
        Seasonal period (quarterly data uses s=4)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['(p,q,P,Q)', 'AIC', 'BIC', 'HQIC'] sorted ascending by AIC

    Notes
    -----
    - Model is fit with simple_differencing=False to retain internal differencing behavior
    - Exceptions during fit are skipped to keep the search robust
    - Progress is displayed via tqdm progress bar
    """
    results: List[List[object]] = []
    
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


def rolling_one_step_predictions(endog: pd.Series,
                                exog_model: Optional[pd.DataFrame],
                                best_order: Tuple,
                                d: int,
                                D: int,
                                s: int,
                                coverage_levels: List[int],
                                start_index: int) -> Tuple[List[float], Dict[int, Tuple[List[float], List[float]]]]:
    """
    Compute leak-free rolling one-step-ahead predictions and predictive intervals.
    
    This function performs true out-of-sample forecasting by refitting the model
    at each step using only historical data, preventing data leakage.
    
    Parameters
    ----------
    endog : pd.Series
        Endogenous time series
    exog_model : Optional[pd.DataFrame]
        Exogenous variables (if any)
    best_order : Tuple
        SARIMAX order (p, q, P, Q)
    d, D, s : int
        Differencing and seasonal parameters
    coverage_levels : List[int]
        Confidence levels for prediction intervals (e.g., [80, 95])
    start_index : int
        Index to start predictions from
        
    Returns
    -------
    Tuple[List[float], Dict[int, Tuple[List[float], List[float]]]]
        (predictions, confidence_bounds) where conf_bounds[level] = (lower, upper)
        
    Notes
    -----
    This is computationally intensive as it refits the model for each prediction,
    but ensures no data leakage and realistic forecast evaluation.
    """
    preds: List[float] = []
    conf_bounds: Dict[int, Tuple[List[float], List[float]]] = {
        lvl: ([], []) for lvl in coverage_levels
    }
    
    n = len(endog)
    for i in range(start_index, n):
        try:
            # Fit model using only data up to time i-1
            m = SARIMAX(
                endog.iloc[:i],
                (exog_model.iloc[:i] if exog_model is not None else None),
                order=(best_order[0], d, best_order[1]),
                seasonal_order=(best_order[2], D, best_order[3], s),
                simple_differencing=False,
            )
            r = m.fit(disp=False)
            
            # Generate one-step forecast
            fc = r.get_forecast(
                steps=1, 
                exog=(exog_model.iloc[i : i + 1] if exog_model is not None else None)
            )
            f_mean = float(fc.predicted_mean.iloc[0])
            
            # Generate confidence intervals for each coverage level
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


def one_step_forecast_at(endog: pd.Series,
                        exog_model: Optional[pd.DataFrame],
                        best_order: Tuple,
                        d: int,
                        D: int,
                        s: int,
                        i: int) -> float:
    """
    Compute a single one-step-ahead forecast at index i using training data up to i-1.
    
    This function generates a single forecast point, useful for specific
    forecast evaluations or when full rolling predictions are not needed.
    
    Parameters
    ----------
    endog : pd.Series
        Endogenous time series
    exog_model : Optional[pd.DataFrame]
        Exogenous variables (if any)
    best_order : Tuple
        SARIMAX order (p, q, P, Q)
    d, D, s : int
        Differencing and seasonal parameters
    i : int
        Index to forecast (must satisfy 1 <= i < len(endog))
        
    Returns
    -------
    float
        One-step forecast value
        
    Raises
    ------
    ValueError
        If index i is out of valid range
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
        fc = r.get_forecast(
            steps=1, 
            exog=(exog_model.iloc[i : i + 1] if exog_model is not None else None)
        )
        return float(fc.predicted_mean.iloc[0])
    except Exception:
        return float(endog.iloc[i - 1])


def hash_forecast(seq: Union[List[float], np.ndarray, pd.Series]) -> str:
    """
    Generate a hash fingerprint for a forecast sequence.
    
    This function creates a unique identifier for forecast outputs, useful for
    detecting duplicate runs or verifying forecast reproducibility.
    
    Parameters
    ----------
    seq : Union[List[float], np.ndarray, pd.Series]
        Forecast sequence to hash
        
    Returns
    -------
    str
        16-character SHA-1 hash of the forecast sequence
        
    Notes
    -----
    Uses first 16 characters of SHA-1 hash for reasonable uniqueness while
    maintaining readability in logs and output files.
    """
    arr = np.asarray(seq, dtype=np.float64)
    try:
        data = arr.tobytes()
    except Exception:
        data = np.asarray(list(seq), dtype=np.float64).tobytes()
    return hashlib.sha1(data).hexdigest()[:16]


def build_exog_for_country(country: str, 
                          args, 
                          endog_index: pd.DatetimeIndex) -> Optional[pd.DataFrame]:
    """
    Construct a quarterly exogenous DataFrame aligned to endog_index with a single column 'EXOG'.
    
    This function fetches and processes exogenous variables specific to each country/region,
    converting monthly data to quarterly and aligning with the target series timeline.
    
    Parameters
    ----------
    country : str
        Country code (US, EU, CN, etc.)
    args : argparse.Namespace
        CLI arguments containing exogenous data configuration
    endog_index : pd.DatetimeIndex
        Target series index for alignment
        
    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame with 'EXOG' column aligned to endog_index, or None if unavailable
        
    Notes
    -----
    Supports different data sources per country:
    - US: ISM PMI, Industrial Production, Manufacturers' Orders, OECD CLI
    - EU: Economic Sentiment Indicator, Industry Confidence, OECD CLI  
    - China: NBS PMI, OECD CLI
    """
    from helpers.temporal import monthly_to_quarterly_avg
    from fetchers.dbnomics_pmi import ism_pmi_monthly, DBNOMICS_ISM_PMI_URL
    from fetchers.eurostat_bcs import fetch_esi_sdmx, fetch_industry_conf_sdmx, to_monthly_series
    from fetchers.nbs_pmi import load_nbs_pmi_csv, fetch_dbnomics_nbs
    
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
                from fetchers.fred_indicators import fetch_us_industrial_production
                series_type = getattr(args, "us_ip_type", "total")
                monthly = fetch_us_industrial_production(series_type=series_type)
            elif src == "manufacturers_orders":
                from fetchers.fred_indicators import fetch_us_manufacturers_orders
                monthly = fetch_us_manufacturers_orders()
            elif src == "oecd_cli":
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
        
    return pd.DataFrame({"EXOG": exog_q})


def run_multifold_evaluation(endog: pd.Series,
                           exog_model: Optional[pd.DataFrame],
                           best_order: Tuple,
                           d: int,
                           D: int,
                           s: int,
                           folds: int = 3,
                           fold_horizon: int = 8) -> Dict[str, Union[List[float], float]]:
    """
    Perform multi-fold expanding-window backtesting with aggregated results.
    
    This function implements a robust evaluation scheme using multiple folds
    with expanding training windows, providing more reliable performance estimates.
    
    Parameters
    ----------
    endog : pd.Series
        Endogenous time series
    exog_model : Optional[pd.DataFrame]
        Exogenous variables (if any)
    best_order : Tuple
        SARIMAX order (p, q, P, Q)
    d, D, s : int
        Differencing and seasonal parameters
    folds : int, default=3
        Number of evaluation folds
    fold_horizon : int, default=8
        Number of steps per fold
        
    Returns
    -------
    Dict[str, Union[List[float], float]]
        Dictionary containing aggregated predictions, true values, naive forecasts,
        and combined statistics across all folds
        
    Notes
    -----
    Uses expanding window approach where each fold uses all previous data
    for training, simulating realistic forecasting scenarios.
    """
    from .metrics_utils import compute_metrics, combine_dm_pvalues
    
    n = len(endog)
    min_train = max(20, s * 4)  # Require minimal training size
    
    start_train = n - (folds * fold_horizon)
    if start_train < min_train:
        # Shrink folds if series is too short
        folds = max(1, (n - min_train) // max(1, fold_horizon))
        start_train = n - (folds * fold_horizon)
        logger.warning("Adjusted folds to %d due to short series.", folds)
        if folds <= 0:
            raise ValueError("Series too short for the requested multi-fold configuration.")

    all_true: List[float] = []
    all_pred: List[float] = []
    all_naive: List[float] = []
    dm_pvals: List[float] = []

    for k in range(folds):
        tr_end = start_train + k * fold_horizon
        te_end = tr_end + fold_horizon
        if te_end > n:
            break

        y_tr = endog.iloc[:tr_end]
        
        y_fold_true: List[float] = []
        y_fold_pred: List[float] = []
        y_fold_naive: List[float] = []

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
                fc = r.get_forecast(
                    steps=1, 
                    exog=(exog_model.iloc[i : i + 1] if exog_model is not None else None)
                )
                y_hat = float(fc.predicted_mean.iloc[0])
                y_tr = pd.concat([y_tr, pd.Series([endog.iloc[i]], index=[endog.index[i]])])
            except Exception:
                y_hat = float(endog.iloc[i - 1])  # Fallback to naive
                
            y_true_i = float(endog.iloc[i])
            y_naive_i = float(endog.iloc[i - 1])

            y_fold_true.append(y_true_i)
            y_fold_pred.append(y_hat)
            y_fold_naive.append(y_naive_i)

        # Compute metrics per fold
        met = compute_metrics(
            y_true=y_fold_true, 
            y_hat=y_fold_pred, 
            y_hat_naive=y_fold_naive, 
            y_train=y_tr, 
            m=4
        )
        
        if np.isfinite(met.get("DM_p", float("nan"))):
            dm_pvals.append(float(met.get("DM_p")))

        all_true.extend(y_fold_true)
        all_pred.extend(y_fold_pred)
        all_naive.extend(y_fold_naive)

    # Aggregate results
    p_comb = combine_dm_pvalues(dm_pvals, method="fisher")
    
    return {
        "true_values": all_true,
        "predictions": all_pred, 
        "naive_predictions": all_naive,
        "combined_dm_pvalue": p_comb,
        "n_folds": folds,
        "fold_horizon": fold_horizon,
        "train_size": start_train
    }


def fit_best_sarimax_model(endog: pd.Series,
                          exog: Optional[pd.DataFrame],
                          best_order: Tuple,
                          d: int,
                          D: int,
                          s: int,
                          fit_kwargs: Optional[Dict] = None):
    """
    Fit the best SARIMAX model with specified parameters.
    
    This function fits a SARIMAX model using the best order found through
    grid search, with optional robust standard errors and other fit options.
    
    Parameters
    ----------
    endog : pd.Series
        Endogenous time series
    exog : Optional[pd.DataFrame]
        Exogenous variables (if any)
    best_order : Tuple
        SARIMAX order (p, q, P, Q)
    d, D, s : int
        Differencing and seasonal parameters
    fit_kwargs : Optional[Dict]
        Additional keyword arguments for model fitting
        
    Returns
    -------
    SARIMAX results object
        Fitted model results
        
    Notes
    -----
    Supports robust standard errors and other advanced fitting options
    through the fit_kwargs parameter.
    """
    if fit_kwargs is None:
        fit_kwargs = {"disp": False}
    
    model = SARIMAX(
        endog,
        exog,
        order=(best_order[0], d, best_order[1]),
        seasonal_order=(best_order[2], D, best_order[3], s),
        simple_differencing=False,
    )
    
    return model.fit(**fit_kwargs)


def validate_sarimax_inputs(endog: pd.Series,
                           exog: Optional[pd.DataFrame] = None,
                           min_obs: int = 20) -> None:
    """
    Validate inputs for SARIMAX modeling.
    
    This function performs basic validation checks to ensure the data is
    suitable for SARIMAX modeling.
    
    Parameters
    ----------
    endog : pd.Series
        Endogenous time series to validate
    exog : Optional[pd.DataFrame]
        Exogenous variables to validate
    min_obs : int, default=20
        Minimum number of observations required
        
    Raises
    ------
    ValueError
        If validation fails
    """
    if endog.empty:
        raise ValueError("Endogenous series cannot be empty")
    
    if len(endog.dropna()) < min_obs:
        raise ValueError(f"Insufficient observations: {len(endog.dropna())} < {min_obs}")
    
    if exog is not None:
        if len(exog) != len(endog):
            raise ValueError("Exogenous variables must have same length as endogenous series")
        
        if exog.isnull().all().any():
            raise ValueError("Exogenous variables cannot be entirely missing")
    
    if not isinstance(endog.index, pd.DatetimeIndex):
        logger.warning("Endogenous series does not have DatetimeIndex")