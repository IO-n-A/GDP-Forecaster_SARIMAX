# gdp_forecaster_src/transform_utils.py

import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def apply_target_transform(series: pd.Series, transform: str) -> Tuple[pd.Series, int]:
    """
    Apply target transformation and return transformed series with suggested differencing order.
    
    This function applies various transformations to prepare time series data for modeling:
    - level: identity transformation, typically requires d=1 differencing
    - qoq: quarter-over-quarter percent change * 100, typically d=0
    - yoy: year-over-year percent change * 100, typically d=0  
    - log_diff: 100 * diff(log), typically d=0
    
    Parameters
    ----------
    series : pd.Series
        Input time series to transform
    transform : str
        Type of transformation to apply ('level', 'qoq', 'yoy', 'log_diff')
        
    Returns
    -------
    Tuple[pd.Series, int]
        Tuple of (transformed_series, suggested_differencing_d)
        
    Notes
    -----
    The function also supports configuration-driven transform mapping for differencing
    parameters, falling back to sensible defaults if configuration is unavailable.
    
    Examples
    --------
    >>> series = pd.Series([100, 102, 104, 106])
    >>> transformed, d = apply_target_transform(series, "qoq")
    >>> # Returns quarter-over-quarter percentage changes and d=0
    """
    from .config_utils import config_manager
    
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


def safe_adf_pval(series: pd.Series) -> float:
    """
    Safely compute ADF test p-value with error handling.
    
    This function performs the Augmented Dickey-Fuller test on a time series
    while handling edge cases like insufficient data or numerical issues.
    
    Parameters
    ----------
    series : pd.Series
        Time series to test for stationarity
        
    Returns
    -------
    float
        ADF test p-value, or NaN if test cannot be performed
        
    Notes
    -----
    Requires at least 12 observations to perform the test reliably.
    """
    from statsmodels.tsa.stattools import adfuller
    
    try:
        s = pd.Series(series).dropna()
        if len(s) < 12:
            return float("nan")
        return float(adfuller(s)[1])
    except Exception:
        return float("nan")


def adf_select_d(series: pd.Series, alpha: float = 0.05) -> int:
    """
    Select non-seasonal differencing order using ADF test heuristic.
    
    This function uses a simple heuristic: if the level series is stationary 
    (p < alpha) then d=0, else d=1.
    
    Parameters
    ----------
    series : pd.Series
        Time series to analyze
    alpha : float, default=0.05
        Significance level for stationarity test
        
    Returns
    -------
    int
        Suggested differencing order (0 or 1)
        
    Notes
    -----
    This is a heuristic approach and may not be optimal for all series.
    Consider using more sophisticated unit root tests for critical applications.
    """
    p_level = safe_adf_pval(series)
    if np.isfinite(p_level) and p_level < alpha:
        return 0
    return 1


def adf_select_D(series: pd.Series, s: int = 4, alpha: float = 0.05) -> int:
    """
    Select seasonal differencing order using ADF test heuristic.
    
    This function uses a heuristic: if seasonal differenced series is stationary 
    (p < alpha) while level is non-stationary, select one seasonal difference; else D=0.
    
    Parameters
    ----------
    series : pd.Series
        Time series to analyze
    s : int, default=4
        Seasonal period (4 for quarterly data)
    alpha : float, default=0.05
        Significance level for stationarity test
        
    Returns
    -------
    int
        Suggested seasonal differencing order (0 or 1)
        
    Notes
    -----
    This heuristic works well for quarterly data but may need adjustment for other frequencies.
    """
    p_level = safe_adf_pval(series)
    try:
        p_seasonal = safe_adf_pval(pd.Series(series).diff(s))
    except Exception:
        p_seasonal = float("nan")
    
    if (not np.isfinite(p_level) or p_level >= alpha) and np.isfinite(p_seasonal) and p_seasonal < alpha:
        return 1
    return 0


def validate_series_for_transform(series: pd.Series, transform: str) -> pd.Series:
    """
    Validate and prepare a time series for transformation.
    
    This function performs basic validation checks to ensure the series is
    suitable for the requested transformation.
    
    Parameters
    ----------
    series : pd.Series
        Input time series
    transform : str
        Transformation type to validate against
        
    Returns
    -------
    pd.Series
        Validated and cleaned series
        
    Raises
    ------
    ValueError
        If series is unsuitable for the transformation
        
    Notes
    -----
    Different transformations have different requirements:
    - log_diff requires positive values
    - All transformations require sufficient data points
    """
    if series.empty:
        raise ValueError("Cannot transform empty series")
    
    if len(series.dropna()) < 4:
        raise ValueError("Insufficient data points for transformation")
    
    if transform == "log_diff":
        if (series <= 0).any():
            raise ValueError("log_diff transformation requires all positive values")
    
    return series.dropna()


def get_transform_description(transform: str) -> str:
    """
    Get a human-readable description of a transformation.
    
    Parameters
    ----------
    transform : str
        Transformation type
        
    Returns
    -------
    str
        Description of the transformation
    """
    descriptions = {
        "level": "Original series (no transformation)",
        "qoq": "Quarter-over-quarter percentage change",
        "yoy": "Year-over-year percentage change", 
        "log_diff": "Log difference (approximate percentage change)"
    }
    return descriptions.get(transform, f"Unknown transformation: {transform}")