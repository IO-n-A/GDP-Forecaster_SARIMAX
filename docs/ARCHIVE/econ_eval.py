# -*- coding: utf-8 -*-
"""Evaluation metrics for forecast accuracy and comparative tests.

Purpose
-------
Provide a compact set of forecast evaluation metrics for time series:
- Point-error metrics: ME (bias), MAE, RMSE
- Percentage metrics: MAPE (epsilon-guarded), sMAPE
- Scaled error: MASE (with seasonal period m, default 4 for quarterly)
- Theil’s inequality coefficients: U1, U2 (vs a naive comparator)
- Diebold–Mariano (DM) test for equal predictive accuracy at h=1

Design
------
- Robust to numpy/pandas inputs; NaNs are dropped consistently via alignment helpers.
- MAPE and sMAPE are returned in percent.
- MASE requires an in-sample y_train for scaling; returns NaN if scaling is ill-posed.
- DM test returns (t, p) using t-distribution (scipy) with a Normal fallback.

Notes
-----
- MAPE can distort when true values are near zero; inspect sMAPE and MASE alongside MAPE.
- DM test null: equal predictive accuracy. With squared loss, negative t implies the model
  improves upon the naive baseline (since d_t = l_model - l_naive).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

EPS: float = 1e-12


def _as_1d(a: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Convert input to a contiguous 1D float numpy array (copy-free when possible).

    Parameters
    ----------
    a : Sequence[float] | np.ndarray
        Input sequence or array.

    Returns
    -------
    np.ndarray
        One-dimensional array view/copy (dtype=float).
    """
    arr = np.asarray(a, dtype=float)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr


def _align_two(y_true: Sequence[float] | np.ndarray, y_hat: Sequence[float] | np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two sequences to equal length and drop NaNs consistently.

    Parameters
    ----------
    y_true : Sequence[float] | np.ndarray
        Observed values.
    y_hat : Sequence[float] | np.ndarray
        Predicted values.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Cleaned (y_true, y_hat) arrays with NaNs removed and lengths matched.
    """
    y = _as_1d(y_true)
    yh = _as_1d(y_hat)
    n = min(len(y), len(yh))
    y, yh = y[:n], yh[:n]
    mask = (~np.isnan(y)) & (~np.isnan(yh))
    return y[mask], yh[mask]


def _align_three(
    y_true: Sequence[float] | np.ndarray,
    y_hat: Sequence[float] | np.ndarray,
    y_naive: Sequence[float] | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align three sequences to equal length and drop NaNs consistently across all.

    Parameters
    ----------
    y_true : Sequence[float] | np.ndarray
        Observed values.
    y_hat : Sequence[float] | np.ndarray
        Model predictions.
    y_naive : Sequence[float] | np.ndarray
        Naive comparator predictions (e.g., last-observation).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Cleaned (y_true, y_hat, y_naive) arrays with NaNs removed and lengths matched.
    """
    y = _as_1d(y_true)
    yh = _as_1d(y_hat)
    yn = _as_1d(y_naive)
    n = min(len(y), len(yh), len(yn))
    y, yh, yn = y[:n], yh[:n], yn[:n]
    mask = (~np.isnan(y)) & (~np.isnan(yh)) & (~np.isnan(yn))
    return y[mask], yh[mask], yn[mask]


def mae(y_true: Sequence[float] | np.ndarray, y_hat: Sequence[float] | np.ndarray) -> float:
    """
    Mean Absolute Error.

    Parameters
    ----------
    y_true : Sequence[float] | np.ndarray
        Observed values.
    y_hat : Sequence[float] | np.ndarray
        Predicted values.

    Returns
    -------
    float
        MAE (NaN if no valid observations).
    """
    y, yh = _align_two(y_true, y_hat)
    if y.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y - yh)))


def rmse(y_true: Sequence[float] | np.ndarray, y_hat: Sequence[float] | np.ndarray) -> float:
    """
    Root Mean Squared Error.

    Parameters
    ----------
    y_true : Sequence[float] | np.ndarray
        Observed values.
    y_hat : Sequence[float] | np.ndarray
        Predicted values.

    Returns
    -------
    float
        RMSE (NaN if no valid observations).
    """
    y, yh = _align_two(y_true, y_hat)
    if y.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y - yh) ** 2)))


def mape(y_true: Sequence[float] | np.ndarray, y_hat: Sequence[float] | np.ndarray, eps: float = EPS) -> float:
    """
    Mean Absolute Percentage Error (percent), epsilon-guarded.

    Parameters
    ----------
    y_true : Sequence[float] | np.ndarray
        Observed values.
    y_hat : Sequence[float] | np.ndarray
        Predicted values.
    eps : float, optional
        Small constant used to bound the denominator away from zero (default: EPS=1e-12).

    Returns
    -------
    float
        MAPE in percent (NaN if no valid observations).

    Notes
    -----
    - Sensitive to near-zero y_true; consider sMAPE/MASE alongside MAPE.
    """
    y, yh = _align_two(y_true, y_hat)
    if y.size == 0:
        return float("nan")
    denom = np.maximum(np.abs(y), eps)
    return float(np.mean(np.abs((y - yh) / denom)) * 100.0)


def smape(y_true: Sequence[float] | np.ndarray, y_hat: Sequence[float] | np.ndarray, eps: float = EPS) -> float:
    """
    Symmetric MAPE (percent): 2|e| / (|y| + |ŷ|), epsilon-guarded.

    Parameters
    ----------
    y_true : Sequence[float] | np.ndarray
        Observed values.
    y_hat : Sequence[float] | np.ndarray
        Predicted values.
    eps : float, optional
        Small constant to avoid division by zero.

    Returns
    -------
    float
        sMAPE in percent (NaN if no valid observations).
    """
    y, yh = _align_two(y_true, y_hat)
    if y.size == 0:
        return float("nan")
    denom = np.maximum(np.abs(y) + np.abs(yh), eps)
    return float(np.mean(2.0 * np.abs(y - yh) / denom) * 100.0)


def median_ape(y_true: Sequence[float] | np.ndarray, y_hat: Sequence[float] | np.ndarray, eps: float = EPS) -> float:
    """
    Median Absolute Percentage Error (percent), epsilon-guarded.

    Parameters
    ----------
    y_true : Sequence[float] | np.ndarray
        Observed values.
    y_hat : Sequence[float] | np.ndarray
        Predicted values.
    eps : float, optional
        Small constant to bound denominators away from zero.

    Returns
    -------
    float
        Median APE in percent (NaN if no valid observations).
    """
    y, yh = _align_two(y_true, y_hat)
    if y.size == 0:
        return float("nan")
    denom = np.maximum(np.abs(y), eps)
    ape = np.abs((y - yh) / denom) * 100.0
    return float(np.median(ape))


def mean_error(y_true: Sequence[float] | np.ndarray, y_hat: Sequence[float] | np.ndarray) -> float:
    """
    Mean Error (bias) defined as mean(ŷ - y).

    Parameters
    ----------
    y_true : Sequence[float] | np.ndarray
        Observed values.
    y_hat : Sequence[float] | np.ndarray
        Predicted values.

    Returns
    -------
    float
        Bias (NaN if no valid observations).
    """
    y, yh = _align_two(y_true, y_hat)
    if y.size == 0:
        return float("nan")
    return float(np.mean(yh - y))


def mase(
    y_true: Sequence[float] | np.ndarray,
    y_hat: Sequence[float] | np.ndarray,
    y_train: Optional[Sequence[float] | np.ndarray] = None,
    m: int = 4,
) -> float:
    """
    Mean Absolute Scaled Error with seasonal period m.

    Parameters
    ----------
    y_true : Sequence[float] | np.ndarray
        Observed values (out-of-sample).
    y_hat : Sequence[float] | np.ndarray
        Predicted values (out-of-sample).
    y_train : Optional[Sequence[float] | np.ndarray], optional
        In-sample training series used to compute the seasonal naive scale:
        mean(|y_t - y_{t-m}|), t = m..n-1 (computed on y_train).
    m : int, optional
        Seasonal period (default 4 for quarterly GDP).

    Returns
    -------
    float
        MASE (NaN if scaling is undefined or inputs are empty).

    Notes
    -----
    - Returns NaN if y_train is None, too short, or yields a near-zero scale.
    """
    y, yh = _align_two(y_true, y_hat)
    if y.size == 0:
        return float("nan")
    if y_train is None:
        return float("nan")
    train = _as_1d(y_train)
    if train.size <= m:
        return float("nan")
    diffs = train[m:] - train[:-m]
    denom = float(np.nanmean(np.abs(diffs)))
    if not np.isfinite(denom) or denom < EPS:
        return float("nan")
    num = float(np.mean(np.abs(y - yh)))
    return float(num / denom)


def theil_u1(y_true: Sequence[float] | np.ndarray, y_hat: Sequence[float] | np.ndarray) -> float:
    """
    Theil’s U1 (inequality coefficient).

    Definition
    ----------
    U1 = RMSE(ŷ, y) / (sqrt(mean(y^2)) + sqrt(mean(ŷ^2)))

    Parameters
    ----------
    y_true : Sequence[float] | np.ndarray
        Observed values.
    y_hat : Sequence[float] | np.ndarray
        Predicted values.

    Returns
    -------
    float
        Theil’s U1 (NaN if denominator is degenerate).
    """
    y, yh = _align_two(y_true, y_hat)
    if y.size == 0:
        return float("nan")
    num = float(np.sqrt(np.mean((y - yh) ** 2)))
    den = float(np.sqrt(np.mean(y**2)) + np.sqrt(np.mean(yh**2)))
    if den < EPS:
        return float("nan")
    return float(num / den)


def theil_u2(
    y_true: Sequence[float] | np.ndarray,
    y_hat: Sequence[float] | np.ndarray,
    y_hat_naive: Sequence[float] | np.ndarray,
) -> float:
    """
    Theil’s U2 defined as RMSE_model / RMSE_naive.

    Parameters
    ----------
    y_true : Sequence[float] | np.ndarray
        Observed values.
    y_hat : Sequence[float] | np.ndarray
        Model predictions.
    y_hat_naive : Sequence[float] | np.ndarray
        Naive comparator predictions (aligned with y_true).

    Returns
    -------
    float
        Theil’s U2 (NaN if RMSE_naive is degenerate).
    """
    y, yh, yn = _align_three(y_true, y_hat, y_hat_naive)
    if y.size == 0:
        return float("nan")
    rmse_model = float(np.sqrt(np.mean((y - yh) ** 2)))
    rmse_naive = float(np.sqrt(np.mean((y - yn) ** 2)))
    denom = max(rmse_naive, EPS)
    return float(rmse_model / denom)


def diebold_mariano_h1(
    y_true: Sequence[float] | np.ndarray,
    y_hat: Sequence[float] | np.ndarray,
    y_hat_naive: Sequence[float] | np.ndarray,
    loss: str = "squared",
) -> Tuple[float, float]:
    """
    Diebold–Mariano test for equal predictive accuracy at horizon h=1.

    Parameters
    ----------
    y_true : Sequence[float] | np.ndarray
        Observed values.
    y_hat : Sequence[float] | np.ndarray
        Model predictions.
    y_hat_naive : Sequence[float] | np.ndarray
        Naive comparator predictions.
    loss : {'squared','absolute'}
        Loss function for d_t: l_model - l_naive.

    Returns
    -------
    Tuple[float, float]
        (t_statistic, p_value). Returns (NaN, NaN) if n < 5 or variance is degenerate.

    Raises
    ------
    ValueError
        If an unsupported loss is specified.

    Notes
    -----
    - Test statistic: t = mean(d) / sqrt(var(d)/n).
    - Two-sided p-value via Student’s t (df=n-1). Falls back to Normal if scipy is unavailable.
    """
    y, yh, yn = _align_three(y_true, y_hat, y_hat_naive)
    n = int(y.size)
    if n < 5:
        return float("nan"), float("nan")
    if loss == "squared":
        l_model = (y - yh) ** 2
        l_naive = (y - yn) ** 2
    elif loss == "absolute":
        l_model = np.abs(y - yh)
        l_naive = np.abs(y - yn)
    else:
        raise ValueError("Unsupported loss. Use 'squared' or 'absolute'.")
    d = l_model - l_naive
    mean_d = float(np.mean(d))
    var_d = float(np.var(d, ddof=1))
    if not np.isfinite(var_d) or var_d <= EPS:
        return float("nan"), float("nan")
    se = math.sqrt(var_d / n)
    if se <= EPS:
        return float("nan"), float("nan")
    t_stat = mean_d / se

    # Two-sided p-value computation
    # Prefer Student's t (df=n-1) when scipy is present; otherwise fall back to Normal.
    p_val: float
    try:
        from scipy import stats as st  # type: ignore

        p_val = float(2.0 * st.t.sf(abs(t_stat), df=n - 1))
    except Exception:
        # Normal fallback using error function
        z = abs(t_stat)
        # 2 * (1 - Phi(z)) where Phi is standard normal CDF
        p_val = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))))

    return float(t_stat), float(p_val)


def compute_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_hat: Sequence[float] | np.ndarray,
    y_hat_naive: Sequence[float] | np.ndarray,
    y_train: Optional[Sequence[float] | np.ndarray] = None,
    m: int = 4,
) -> Dict[str, float]:
    """
    Compute a suite of forecast metrics and comparative statistics.

    Parameters
    ----------
    y_true : Sequence[float] | np.ndarray
        Observed values.
    y_hat : Sequence[float] | np.ndarray
        Model predictions.
    y_hat_naive : Sequence[float] | np.ndarray
        Naive comparator predictions aligned to y_true.
    y_train : Optional[Sequence[float] | np.ndarray], optional
        In-sample training series for MASE scaling (optional).
    m : int, optional
        Seasonal period for MASE scaling (default 4 for quarterly series).

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
        ME, MAE, RMSE, MAPE, sMAPE, median_APE, MASE, TheilU1, TheilU2, DM_t, DM_p

    Notes
    -----
    - Drops NaNs consistently via alignment helpers.
    - DM test is computed at h=1 under squared error.
    """
    metrics: Dict[str, float] = {}

    # Core metrics on model predictions
    metrics["ME"] = mean_error(y_true, y_hat)
    metrics["MAE"] = mae(y_true, y_hat)
    metrics["RMSE"] = rmse(y_true, y_hat)
    metrics["MAPE"] = mape(y_true, y_hat)
    metrics["sMAPE"] = smape(y_true, y_hat)
    metrics["median_APE"] = median_ape(y_true, y_hat)
    metrics["MASE"] = mase(y_true, y_hat, y_train=y_train, m=m)
    metrics["TheilU1"] = theil_u1(y_true, y_hat)

    # Metrics requiring a naive comparator
    metrics["TheilU2"] = theil_u2(y_true, y_hat, y_hat_naive)

    # DM test (h=1, squared loss)
    t_stat, p_val = diebold_mariano_h1(y_true, y_hat, y_hat_naive, loss="squared")
    metrics["DM_t"] = t_stat
    metrics["DM_p"] = p_val

    return metrics


def combine_dm_pvalues(pvals: Sequence[float], method: str = "fisher") -> float:
    """
    Combine multiple two-sided p-values into a single aggregate p-value.

    Parameters
    ----------
    pvals : Sequence[float]
        Iterable of p-values in [0, 1]. NaN/inf/invalid values are dropped.
    method : {'fisher','stouffer'}
        - 'fisher': Fisher's method, chi-square with df=2k.
        - 'stouffer': Stouffer's Z with equal weights (two-sided).

    Returns
    -------
    float
        Combined p-value in [0, 1], or NaN if input contains no valid p-values.

    Notes
    -----
    - Requires scipy for distribution functions. If scipy is not available, returns NaN.
    - Input p-values are assumed two-sided. For Stouffer, they are converted to Z via
      the inverse survival function with a two-sided adjustment.
    """
    arr = np.asarray(list(pvals), dtype=float)
    # Keep finite values in [0,1]
    mask = np.isfinite(arr) & (arr >= 0.0) & (arr <= 1.0)
    arr = arr[mask]
    k = int(arr.size)
    if k == 0:
        return float("nan")

    method = str(method or "fisher").strip().lower()
    try:
        from scipy import stats as st  # type: ignore
    except Exception:
        return float("nan")

    if method == "fisher":
        # Fisher's statistic: -2 * sum(log(p_i)) ~ Chi^2_{2k}
        stat = -2.0 * float(np.sum(np.log(np.maximum(arr, EPS))))
        p_comb = float(1.0 - st.chi2.cdf(stat, df=2 * k))
        return p_comb
    elif method == "stouffer":
        # Convert two-sided p to Z via inverse Normal; combine and back to two-sided p
        # Use z_i = Phi^{-1}(1 - p_i/2); equal weights
        zi = st.norm.isf(arr / 2.0)
        z_comb = float(np.sum(zi) / np.sqrt(k))
        p_comb = float(2.0 * st.norm.sf(abs(z_comb)))
        return p_comb
    else:
        raise ValueError("Unsupported method. Use 'fisher' or 'stouffer'.")
