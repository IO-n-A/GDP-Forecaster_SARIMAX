# gdp_forecaster_src/metrics_utils.py

import math
import numpy as np
import pandas as pd
from typing import Union, List, Dict
import logging

logger = logging.getLogger(__name__)


def to_1d_array(x: Union[List[float], np.ndarray, pd.Series]) -> np.ndarray:
    """
    Convert input to 1D numpy array, filtering out non-finite values.
    
    This utility function standardizes input data for metric calculations by
    converting various input types to a clean 1D numpy array.
    
    Parameters
    ----------
    x : Union[List[float], np.ndarray, pd.Series]
        Input data to convert
        
    Returns
    -------
    np.ndarray
        1D array containing only finite values
    """
    arr = np.asarray(x, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def mape_epsilon_from_train(y_train: Union[List[float], np.ndarray, pd.Series]) -> float:
    """
    Calculate epsilon value for stabilized MAPE computation from training data.
    
    This function computes an appropriate epsilon value to avoid division by zero
    in MAPE calculations, based on the 10th percentile of absolute training values.
    
    Parameters
    ----------
    y_train : Union[List[float], np.ndarray, pd.Series]
        Training data used to determine appropriate epsilon
        
    Returns
    -------
    float
        Epsilon value for MAPE stabilization (minimum 1e-8)
    """
    arr = to_1d_array(y_train)
    if arr.size == 0:
        return 1e-8
    return float(max(1e-8, np.percentile(np.abs(arr), 10.0)))


def mape_eps(y_true: Union[List[float], np.ndarray, pd.Series], 
            y_hat: Union[List[float], np.ndarray, pd.Series], 
            eps: float) -> float:
    """
    Calculate Mean Absolute Percentage Error with epsilon stabilization.
    
    This function computes MAPE while avoiding division by zero through the use
    of an epsilon parameter that provides a minimum denominator value.
    
    Parameters
    ----------
    y_true : Union[List[float], np.ndarray, pd.Series]
        True values
    y_hat : Union[List[float], np.ndarray, pd.Series]
        Predicted values
    eps : float
        Epsilon value to prevent division by zero
        
    Returns
    -------
    float
        MAPE as percentage (0-100+), or NaN if no valid data
    """
    yt = to_1d_array(y_true)
    yh = to_1d_array(y_hat)
    n = min(len(yt), len(yh))
    if n == 0:
        return float("nan")
    yt = yt[:n]
    yh = yh[:n]
    denom = np.maximum(np.abs(yt), eps)
    return float(np.mean(np.abs(yh - yt) / denom) * 100.0)


def smape(y_true: Union[List[float], np.ndarray, pd.Series], 
          y_hat: Union[List[float], np.ndarray, pd.Series], 
          eps: float = 1e-12) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error.
    
    SMAPE is less sensitive to outliers than MAPE and handles zero values better
    by using the average of actual and predicted values in the denominator.
    
    Parameters
    ----------
    y_true : Union[List[float], np.ndarray, pd.Series]
        True values
    y_hat : Union[List[float], np.ndarray, pd.Series]
        Predicted values
    eps : float, default=1e-12
        Small value to prevent division by zero
        
    Returns
    -------
    float
        sMAPE as percentage (0-200), or NaN if no valid data
    """
    yt = to_1d_array(y_true)
    yh = to_1d_array(y_hat)
    n = min(len(yt), len(yh))
    if n == 0:
        return float("nan")
    yt = yt[:n]
    yh = yh[:n]
    denom = np.maximum(np.abs(yt) + np.abs(yh), eps)
    return float(np.mean(2.0 * np.abs(yh - yt) / denom) * 100.0)


def mae(y_true: Union[List[float], np.ndarray, pd.Series], 
        y_hat: Union[List[float], np.ndarray, pd.Series]) -> float:
    """
    Calculate Mean Absolute Error.
    
    MAE provides a robust measure of prediction accuracy that is less sensitive
    to outliers compared to RMSE.
    
    Parameters
    ----------
    y_true : Union[List[float], np.ndarray, pd.Series]
        True values
    y_hat : Union[List[float], np.ndarray, pd.Series]
        Predicted values
        
    Returns
    -------
    float
        Mean absolute error, or NaN if no valid data
    """
    yt = to_1d_array(y_true)
    yh = to_1d_array(y_hat)
    n = min(len(yt), len(yh))
    if n == 0:
        return float("nan")
    return float(np.mean(np.abs(yh[:n] - yt[:n])))


def rmse(y_true: Union[List[float], np.ndarray, pd.Series], 
         y_hat: Union[List[float], np.ndarray, pd.Series]) -> float:
    """
    Calculate Root Mean Square Error.
    
    RMSE penalizes large errors more heavily than MAE, making it useful when
    large errors are particularly undesirable.
    
    Parameters
    ----------
    y_true : Union[List[float], np.ndarray, pd.Series]
        True values
    y_hat : Union[List[float], np.ndarray, pd.Series]
        Predicted values
        
    Returns
    -------
    float
        Root mean square error, or NaN if no valid data
    """
    yt = to_1d_array(y_true)
    yh = to_1d_array(y_hat)
    n = min(len(yt), len(yh))
    if n == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yh[:n] - yt[:n]) ** 2)))


def mase_metric(y_true: Union[List[float], np.ndarray, pd.Series], 
                y_hat: Union[List[float], np.ndarray, pd.Series], 
                y_train: Union[List[float], np.ndarray, pd.Series], 
                m: int = 4) -> float:
    """
    Calculate Mean Absolute Scaled Error.
    
    MASE scales the MAE by the MAE of a naive seasonal forecast, making it
    particularly useful for comparing forecasts across different time series.
    
    Parameters
    ----------
    y_true : Union[List[float], np.ndarray, pd.Series]
        True values
    y_hat : Union[List[float], np.ndarray, pd.Series]
        Predicted values
    y_train : Union[List[float], np.ndarray, pd.Series]
        Training data for scaling reference
    m : int, default=4
        Seasonal period for naive forecast (4 for quarterly data)
        
    Returns
    -------
    float
        MASE value, or NaN if computation is not possible
        
    Notes
    -----
    Values < 1 indicate the forecast is better than naive seasonal forecast.
    Values > 1 indicate the forecast is worse than naive seasonal forecast.
    """
    yt = to_1d_array(y_true)
    yh = to_1d_array(y_hat)
    n = min(len(yt), len(yh))
    if n == 0:
        return float("nan")
    
    num = np.mean(np.abs(yh[:n] - yt[:n]))
    tr = to_1d_array(y_train)
    if len(tr) <= m:
        return float("nan")
    
    denom = np.mean(np.abs(tr[m:] - tr[:-m]))
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return float(num / denom)


def theil_u1(y_true: Union[List[float], np.ndarray, pd.Series], 
             y_hat: Union[List[float], np.ndarray, pd.Series]) -> float:
    """
    Calculate Theil's U1 statistic (coefficient of inequality).
    
    U1 measures the relative accuracy of forecasts, with values closer to 0
    indicating better forecasts. It is scale-invariant and bounded between 0 and 1.
    
    Parameters
    ----------
    y_true : Union[List[float], np.ndarray, pd.Series]
        True values
    y_hat : Union[List[float], np.ndarray, pd.Series]
        Predicted values
        
    Returns
    -------
    float
        Theil's U1 statistic (0-1), or NaN if computation is not possible
    """
    yt = to_1d_array(y_true)
    yh = to_1d_array(y_hat)
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


def theil_u2(y_true: Union[List[float], np.ndarray, pd.Series], 
             y_hat: Union[List[float], np.ndarray, pd.Series], 
             y_hat_naive: Union[List[float], np.ndarray, pd.Series]) -> float:
    """
    Calculate Theil's U2 statistic (relative forecast accuracy).
    
    U2 compares forecast accuracy against a naive forecast. Values < 1 indicate
    the forecast is better than the naive benchmark.
    
    Parameters
    ----------
    y_true : Union[List[float], np.ndarray, pd.Series]
        True values
    y_hat : Union[List[float], np.ndarray, pd.Series]
        Predicted values
    y_hat_naive : Union[List[float], np.ndarray, pd.Series]
        Naive forecast values for comparison
        
    Returns
    -------
    float
        Theil's U2 statistic, or NaN if computation is not possible
    """
    yt = to_1d_array(y_true)
    yh = to_1d_array(y_hat)
    yn = to_1d_array(y_hat_naive)
    n = min(len(yt), len(yh), len(yn))
    if n == 0:
        return float("nan")
    
    rmse_f = math.sqrt(float(np.mean((yh[:n] - yt[:n]) ** 2)))
    rmse_n = math.sqrt(float(np.mean((yn[:n] - yt[:n]) ** 2)))
    if rmse_n == 0.0:
        return float("nan")
    return float(rmse_f / rmse_n)


def norm_cdf(z: float) -> float:
    """
    Calculate the cumulative distribution function of the standard normal distribution.
    
    This function provides a fast approximation of the normal CDF without requiring
    external dependencies like scipy.
    
    Parameters
    ----------
    z : float
        Standard normal variable
        
    Returns
    -------
    float
        Probability that a standard normal random variable is less than z
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def dm_newey_west_var(d: np.ndarray, h: int) -> float:
    """
    Calculate Newey-West variance estimator for Diebold-Mariano test.
    
    This function estimates the long-run variance of the loss differential series
    using the Newey-West HAC (Heteroskedasticity and Autocorrelation Consistent) estimator.
    
    Parameters
    ----------
    d : np.ndarray
        Array of loss differentials
    h : int
        Forecast horizon
        
    Returns
    -------
    float
        Variance estimate, or NaN if computation fails
    """
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


def diebold_mariano(y_true: Union[List[float], np.ndarray, pd.Series], 
                   y_hat1: Union[List[float], np.ndarray, pd.Series], 
                   y_hat2: Union[List[float], np.ndarray, pd.Series], 
                   h: int = 1, power: int = 2) -> tuple[float, float]:
    """
    Perform the Diebold-Mariano test for predictive accuracy.
    
    This test compares the forecast accuracy of two competing methods by testing
    whether the expected loss differential is significantly different from zero.
    
    Parameters
    ----------
    y_true : Union[List[float], np.ndarray, pd.Series]
        True values
    y_hat1 : Union[List[float], np.ndarray, pd.Series]
        Predictions from first method
    y_hat2 : Union[List[float], np.ndarray, pd.Series]
        Predictions from second method
    h : int, default=1
        Forecast horizon for variance adjustment
    power : int, default=2
        Power for loss function (1=MAE, 2=MSE)
        
    Returns
    -------
    tuple[float, float]
        (test_statistic, p_value), both NaN if test cannot be performed
        
    Notes
    -----
    Null hypothesis: both methods have equal predictive accuracy.
    Alternative: method 1 and method 2 have different predictive accuracy.
    """
    yt = to_1d_array(y_true)
    y1 = to_1d_array(y_hat1)
    y2 = to_1d_array(y_hat2)
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
    var_dbar = dm_newey_west_var(d, h=h)
    
    if not np.isfinite(var_dbar) or var_dbar <= 0.0:
        return float("nan"), float("nan")
    
    dm_t = dbar / math.sqrt(var_dbar)
    p = 2.0 * (1.0 - norm_cdf(abs(dm_t)))
    return float(dm_t), float(min(max(p, 0.0), 1.0))


def norm_ppf(p: float) -> float:
    """
    Calculate the percent point function (inverse CDF) of the standard normal distribution.
    
    This function uses Acklam's approximation algorithm to provide a fast and
    accurate inverse normal CDF without external dependencies.
    
    Parameters
    ----------
    p : float
        Probability value (0 < p < 1)
        
    Returns
    -------
    float
        Standard normal quantile corresponding to probability p
    """
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    
    # Acklam's coefficients
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
    """
    Combine multiple Diebold-Mariano p-values using meta-analysis methods.
    
    This function aggregates p-values from multiple DM tests, useful when
    conducting fold-based cross-validation with DM tests on each fold.
    
    Parameters
    ----------
    pvals : List[float]
        List of p-values to combine
    method : str, default="fisher"
        Method for combining p-values ("fisher" or "stouffer")
        
    Returns
    -------
    float
        Combined p-value, or NaN if combination is not possible
        
    Notes
    -----
    The Stouffer method is used as default since it doesn't require scipy.
    Fisher's method is approximated using Stouffer for simplicity.
    """
    vals = [float(p) for p in pvals if np.isfinite(p) and 0.0 <= p <= 1.0]
    if not vals:
        return float("nan")
    
    mth = (method or "fisher").lower()
    
    # Use Stouffer combination (no SciPy needed)
    try:
        zvals = [norm_ppf(1.0 - v/2.0) for v in vals]
        z = float(np.sum(zvals)) / math.sqrt(len(zvals))
        return float(2.0 * (1.0 - norm_cdf(abs(z))))
    except Exception:
        return float("nan")


def compute_metrics(y_true: Union[List[float], np.ndarray, pd.Series],
                   y_hat: Union[List[float], np.ndarray, pd.Series],
                   y_hat_naive: Union[List[float], np.ndarray, pd.Series],
                   y_train: Union[List[float], np.ndarray, pd.Series],
                   m: int = 4) -> Dict[str, float]:
    """
    Compute a comprehensive set of forecast evaluation metrics.
    
    This function calculates multiple metrics to provide a thorough assessment
    of forecast accuracy across different dimensions.
    
    Parameters
    ----------
    y_true : Union[List[float], np.ndarray, pd.Series]
        True values
    y_hat : Union[List[float], np.ndarray, pd.Series]
        Predicted values
    y_hat_naive : Union[List[float], np.ndarray, pd.Series]
        Naive forecast for comparison
    y_train : Union[List[float], np.ndarray, pd.Series]
        Training data for MAPE epsilon and MASE scaling
    m : int, default=4
        Seasonal period for MASE computation
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing all computed metrics
        
    Notes
    -----
    Includes: ME, MAE, RMSE, MAPE, sMAPE, median_APE, MASE, TheilU1, TheilU2, DM_t, DM_p
    """
    yt = to_1d_array(y_true)
    yh = to_1d_array(y_hat)
    yn = to_1d_array(y_hat_naive)
    n = min(len(yt), len(yh), len(yn)) if len(yn) > 0 else min(len(yt), len(yh))
    yt = yt[:n]; yh = yh[:n]; yn = yn[:n] if len(yn) > 0 else np.array([], dtype=float)
    
    eps = mape_epsilon_from_train(y_train)
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