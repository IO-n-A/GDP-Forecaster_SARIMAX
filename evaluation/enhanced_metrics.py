"""Enhanced evaluation metrics with stability filtering and deprecation handling.

This module provides a refined metrics calculation system that prioritizes
stable and robust metrics for economic forecasting while handling deprecated
metrics appropriately.

Features:
- Stable primary metrics: MAE, RMSE, MASE, Theil's U
- Robust secondary metrics: Median AE, Directional Accuracy  
- Deprecation warnings for unstable metrics (MAPE/sMAPE)
- Performance assessment against configurable thresholds
- Metric stability analysis and filtering
- Integration with configuration system
- Enhanced reporting with metric prioritization
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

# Remove sklearn dependency - use numpy implementations instead
def mean_absolute_error(y_true, y_pred):
    """Numpy implementation of mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    """Numpy implementation of mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

# Configuration integration
try:
    from config import get_config, ConfigurationManager
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetricStability(Enum):
    """Stability classification for evaluation metrics."""
    STABLE = "stable"           # Reliable across all conditions
    ROBUST = "robust"           # Generally reliable, some edge cases
    UNSTABLE = "unstable"       # Known issues with certain data patterns
    DEPRECATED = "deprecated"   # No longer recommended for headline results


class PerformanceLevel(Enum):
    """Performance assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"


@dataclass
class MetricResult:
    """Result from a single metric calculation."""
    
    name: str
    value: float
    stability: MetricStability
    description: str
    
    # Performance assessment
    performance_level: Optional[PerformanceLevel] = None
    threshold_info: Optional[Dict[str, float]] = None
    
    # Computation metadata
    n_observations: Optional[int] = None
    computation_warnings: List[str] = field(default_factory=list)
    
    # Additional statistics
    confidence_interval: Optional[Tuple[float, float]] = None
    bootstrap_std: Optional[float] = None
    
    def __post_init__(self):
        """Validate metric result."""
        if np.isnan(self.value) or np.isinf(self.value):
            self.computation_warnings.append(f"Invalid metric value: {self.value}")
        
        if self.stability == MetricStability.DEPRECATED:
            warning_msg = f"Metric '{self.name}' is deprecated due to known stability issues"
            self.computation_warnings.append(warning_msg)
            warnings.warn(warning_msg, DeprecationWarning, stacklevel=3)


class EnhancedMetricsCalculator:
    """Enhanced metrics calculator with stability filtering and performance assessment."""
    
    def __init__(self, config_manager: Optional = None):
        """Initialize the enhanced metrics calculator.
        
        Parameters
        ----------
        config_manager : ConfigurationManager, optional
            Configuration manager for metric settings
        """
        self.config_manager = config_manager
        
        # Load configuration
        self.metrics_config = self._load_metrics_config()
        
        # Define metric stability classifications
        self.metric_stability = {
            'mae': MetricStability.STABLE,
            'rmse': MetricStability.STABLE,
            'mse': MetricStability.STABLE,
            'mase': MetricStability.STABLE,
            'theil_u1': MetricStability.STABLE,
            'theil_u2': MetricStability.STABLE,
            'median_ae': MetricStability.ROBUST,
            'directional_accuracy': MetricStability.ROBUST,
            'mape': MetricStability.DEPRECATED,
            'smape': MetricStability.DEPRECATED,
            'symmetric_mape': MetricStability.DEPRECATED
        }
    
    def _load_metrics_config(self) -> Dict[str, Any]:
        """Load metrics configuration from config manager."""
        default_config = {
            'primary_metrics': ['mae', 'rmse', 'mase', 'theil_u1', 'median_ae'],
            'secondary_metrics': ['directional_accuracy', 'theil_u2'],
            'deprecated_metrics': ['mape', 'smape'],
            'precision': 4,
            'performance_thresholds': {
                'mae': {'excellent': 1.0, 'good': 2.0, 'acceptable': 5.0},
                'rmse': {'excellent': 1.5, 'good': 3.0, 'acceptable': 7.0},
                'mase': {'excellent': 0.6, 'good': 0.8, 'acceptable': 1.2}
            }
        }
        
        if self.config_manager and CONFIG_AVAILABLE:
            try:
                eval_config = self.config_manager.get_evaluation_config()
                output_config = eval_config.get('output', {})
                
                default_config.update({
                    'primary_metrics': output_config.get('primary_metrics', default_config['primary_metrics']),
                    'secondary_metrics': output_config.get('secondary_metrics', default_config['secondary_metrics']),
                    'deprecated_metrics': output_config.get('deprecated_metrics', default_config['deprecated_metrics']),
                    'precision': output_config.get('precision', default_config['precision']),
                    'performance_thresholds': output_config.get('performance_thresholds', default_config['performance_thresholds'])
                })
                
                logger.debug("Loaded metrics configuration from config manager")
                
            except Exception as e:
                logger.warning("Failed to load metrics config, using defaults: %s", e)
        
        return default_config
    
    def compute_enhanced_metrics(self, 
                                y_true: pd.Series,
                                y_pred: pd.Series,
                                y_naive: Optional[pd.Series] = None,
                                include_deprecated: bool = False) -> Dict[str, MetricResult]:
        """Compute enhanced evaluation metrics with stability assessment.
        
        Parameters
        ----------
        y_true : pd.Series
            True values
        y_pred : pd.Series
            Predicted values
        y_naive : pd.Series, optional
            Naive benchmark predictions for MASE calculation
        include_deprecated : bool, default False
            Whether to include deprecated metrics (with warnings)
            
        Returns
        -------
        dict
            Dictionary of MetricResult objects
        """
        logger.debug("Computing enhanced metrics for %d observations", len(y_true))
        
        # Validate inputs
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        
        results = {}
        
        # Primary stable metrics
        primary_metrics = self.metrics_config['primary_metrics']
        for metric_name in primary_metrics:
            try:
                result = self._compute_single_metric(metric_name, y_true, y_pred, y_naive)
                results[metric_name] = result
            except Exception as e:
                logger.error("Failed to compute %s: %s", metric_name, e)
                continue
        
        # Secondary robust metrics
        secondary_metrics = self.metrics_config['secondary_metrics']
        for metric_name in secondary_metrics:
            try:
                result = self._compute_single_metric(metric_name, y_true, y_pred, y_naive)
                results[metric_name] = result
            except Exception as e:
                logger.error("Failed to compute %s: %s", metric_name, e)
                continue
        
        # Deprecated metrics (with warnings)
        if include_deprecated:
            deprecated_metrics = self.metrics_config['deprecated_metrics']
            for metric_name in deprecated_metrics:
                try:
                    result = self._compute_single_metric(metric_name, y_true, y_pred, y_naive)
                    results[metric_name] = result
                except Exception as e:
                    logger.warning("Failed to compute deprecated metric %s: %s", metric_name, e)
                    continue
        
        logger.info("Computed %d metrics (%d stable, %d robust, %d deprecated)",
                   len(results),
                   len([r for r in results.values() if r.stability == MetricStability.STABLE]),
                   len([r for r in results.values() if r.stability == MetricStability.ROBUST]),
                   len([r for r in results.values() if r.stability == MetricStability.DEPRECATED]))
        
        return results
    
    def _validate_inputs(self, y_true: pd.Series, y_pred: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Validate and align input series."""
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
        
        if len(y_true) == 0:
            raise ValueError("Cannot compute metrics on empty series")
        
        # Align indices and handle missing values
        aligned_data = pd.DataFrame({'true': y_true, 'pred': y_pred}).dropna()
        
        if len(aligned_data) == 0:
            raise ValueError("No valid observations after removing NaN values")
        
        if len(aligned_data) < len(y_true):
            logger.warning("Removed %d NaN observations", len(y_true) - len(aligned_data))
        
        return aligned_data['true'], aligned_data['pred']
    
    def _compute_single_metric(self, 
                              metric_name: str,
                              y_true: pd.Series,
                              y_pred: pd.Series,
                              y_naive: Optional[pd.Series] = None) -> MetricResult:
        """Compute a single metric with full metadata."""
        stability = self.metric_stability.get(metric_name, MetricStability.UNSTABLE)
        warnings_list = []
        
        # Compute metric value
        if metric_name == 'mae':
            value = float(mean_absolute_error(y_true, y_pred))
            description = "Mean Absolute Error"
            
        elif metric_name == 'rmse':
            value = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            description = "Root Mean Squared Error"
            
        elif metric_name == 'mse':
            value = float(mean_squared_error(y_true, y_pred))
            description = "Mean Squared Error"
            
        elif metric_name == 'mase':
            value, mase_warnings = self._compute_mase(y_true, y_pred, y_naive)
            warnings_list.extend(mase_warnings)
            description = "Mean Absolute Scaled Error"
            
        elif metric_name in ['theil_u1', 'theil_u']:
            value = self._compute_theil_u1(y_true, y_pred)
            description = "Theil's U Statistic (U1)"
            
        elif metric_name == 'theil_u2':
            value = self._compute_theil_u2(y_true, y_pred)
            description = "Theil's U Statistic (U2)"
            
        elif metric_name == 'median_ae':
            value = float(np.median(np.abs(y_true - y_pred)))
            description = "Median Absolute Error (robust)"
            
        elif metric_name == 'directional_accuracy':
            value = self._compute_directional_accuracy(y_true, y_pred)
            description = "Directional Accuracy"
            
        elif metric_name == 'mape':
            value, mape_warnings = self._compute_mape(y_true, y_pred)
            warnings_list.extend(mape_warnings)
            warnings_list.append("MAPE deprecated due to instability with values near zero")
            description = "Mean Absolute Percentage Error (deprecated)"
            
        elif metric_name in ['smape', 'symmetric_mape']:
            value, smape_warnings = self._compute_smape(y_true, y_pred)
            warnings_list.extend(smape_warnings)
            warnings_list.append("sMAPE deprecated due to instability and asymmetry issues")
            description = "Symmetric Mean Absolute Percentage Error (deprecated)"
            
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        # Assess performance level
        performance_level = self._assess_performance_level(metric_name, value)
        threshold_info = self.metrics_config['performance_thresholds'].get(metric_name)
        
        return MetricResult(
            name=metric_name,
            value=value,
            stability=stability,
            description=description,
            performance_level=performance_level,
            threshold_info=threshold_info,
            n_observations=len(y_true),
            computation_warnings=warnings_list
        )
    
    def _compute_mase(self, y_true: pd.Series, y_pred: pd.Series, y_naive: Optional[pd.Series]) -> Tuple[float, List[str]]:
        """Compute Mean Absolute Scaled Error."""
        warnings_list = []
        
        if y_naive is None:
            # Use seasonal naive (lag-4 for quarterly data) as fallback
            if len(y_true) > 4:
                naive_mae = np.mean(np.abs(y_true.iloc[4:].values - y_true.iloc[:-4].values))
                warnings_list.append("Using seasonal naive (lag-4) for MASE scaling")
            else:
                # Use simple naive for very short series
                naive_mae = np.mean(np.abs(y_true.iloc[1:].values - y_true.iloc[:-1].values))
                warnings_list.append("Using simple naive for MASE scaling (insufficient data for seasonal)")
        else:
            # Align y_naive with y_true
            aligned = pd.DataFrame({'true': y_true, 'naive': y_naive}).dropna()
            if len(aligned) == 0:
                raise ValueError("No valid observations for MASE calculation")
            naive_mae = np.mean(np.abs(aligned['true'] - aligned['naive']))
        
        if naive_mae == 0:
            warnings_list.append("Zero naive MAE - MASE may be unreliable")
            naive_mae = 1e-8  # Small epsilon to avoid division by zero
        
        forecast_mae = np.mean(np.abs(y_true - y_pred))
        mase = forecast_mae / naive_mae
        
        return float(mase), warnings_list
    
    def _compute_theil_u1(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """Compute Theil's U1 statistic."""
        mse = np.mean((y_true - y_pred) ** 2)
        mean_true_sq = np.mean(y_true ** 2)
        mean_pred_sq = np.mean(y_pred ** 2)
        
        if mean_true_sq + mean_pred_sq == 0:
            return 0.0
        
        return float(np.sqrt(mse) / (np.sqrt(mean_true_sq) + np.sqrt(mean_pred_sq)))
    
    def _compute_theil_u2(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """Compute Theil's U2 statistic (relative to naive forecast)."""
        if len(y_true) < 2:
            return np.nan
        
        # Forecast RMSE
        forecast_mse = np.mean((y_true.iloc[1:] - y_pred.iloc[1:]) ** 2)
        
        # Naive RMSE (random walk)
        naive_mse = np.mean((y_true.iloc[1:] - y_true.iloc[:-1]) ** 2)
        
        if naive_mse == 0:
            return 0.0 if forecast_mse == 0 else np.inf
        
        return float(np.sqrt(forecast_mse / naive_mse))
    
    def _compute_directional_accuracy(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """Compute directional accuracy (proportion of correctly predicted directions)."""
        if len(y_true) < 2:
            return np.nan
        
        true_direction = np.sign(y_true.diff().iloc[1:])
        pred_direction = np.sign(y_pred.diff().iloc[1:])
        
        correct_directions = (true_direction == pred_direction).sum()
        total_directions = len(true_direction)
        
        return float(correct_directions / total_directions)
    
    def _compute_mape(self, y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, List[str]]:
        """Compute MAPE with stability warnings."""
        warnings_list = []
        
        # Check for values near zero
        near_zero = np.abs(y_true) < 1e-8
        if near_zero.any():
            warnings_list.append(f"Found {near_zero.sum()} values near zero - MAPE may be unstable")
        
        # Compute MAPE excluding zero values
        non_zero_mask = y_true != 0
        if not non_zero_mask.any():
            warnings_list.append("All true values are zero - MAPE undefined")
            return np.inf, warnings_list
        
        y_true_nz = y_true[non_zero_mask]
        y_pred_nz = y_pred[non_zero_mask]
        
        mape = np.mean(np.abs((y_true_nz - y_pred_nz) / y_true_nz)) * 100
        
        if len(y_true_nz) < len(y_true):
            warnings_list.append(f"Excluded {len(y_true) - len(y_true_nz)} zero values from MAPE calculation")
        
        return float(mape), warnings_list
    
    def _compute_smape(self, y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, List[str]]:
        """Compute sMAPE with stability warnings."""
        warnings_list = []
        
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        zero_denom = denominator == 0
        
        if zero_denom.any():
            warnings_list.append(f"Found {zero_denom.sum()} cases where both true and predicted values are zero")
        
        # Exclude cases where denominator is zero
        valid_mask = ~zero_denom
        if not valid_mask.any():
            warnings_list.append("All denominators are zero - sMAPE undefined")
            return np.nan, warnings_list
        
        smape = np.mean(np.abs(y_true[valid_mask] - y_pred[valid_mask]) / denominator[valid_mask]) * 100
        
        if len(y_true[valid_mask]) < len(y_true):
            warnings_list.append(f"Excluded {len(y_true) - len(y_true[valid_mask])} undefined cases from sMAPE")
        
        return float(smape), warnings_list
    
    def _assess_performance_level(self, metric_name: str, value: float) -> Optional[PerformanceLevel]:
        """Assess performance level based on metric value and thresholds."""
        thresholds = self.metrics_config['performance_thresholds'].get(metric_name)
        
        if thresholds is None or np.isnan(value) or np.isinf(value):
            return None
        
        # For error metrics (lower is better)
        if metric_name in ['mae', 'rmse', 'mse', 'mape', 'smape', 'median_ae']:
            if value <= thresholds.get('excellent', 0):
                return PerformanceLevel.EXCELLENT
            elif value <= thresholds.get('good', 0):
                return PerformanceLevel.GOOD
            elif value <= thresholds.get('acceptable', float('inf')):
                return PerformanceLevel.ACCEPTABLE
            else:
                return PerformanceLevel.POOR
        
        # For scale-independent metrics (around 1.0 is neutral)
        elif metric_name in ['mase', 'theil_u1', 'theil_u2']:
            if value <= thresholds.get('excellent', 0.6):
                return PerformanceLevel.EXCELLENT
            elif value <= thresholds.get('good', 0.8):
                return PerformanceLevel.GOOD
            elif value <= thresholds.get('acceptable', 1.2):
                return PerformanceLevel.ACCEPTABLE
            else:
                return PerformanceLevel.POOR
        
        # For accuracy metrics (higher is better)
        elif metric_name == 'directional_accuracy':
            if value >= 0.7:
                return PerformanceLevel.EXCELLENT
            elif value >= 0.6:
                return PerformanceLevel.GOOD
            elif value >= 0.5:
                return PerformanceLevel.ACCEPTABLE
            else:
                return PerformanceLevel.POOR
        
        return None


def compute_enhanced_metrics(y_true: pd.Series,
                           y_pred: pd.Series,
                           y_naive: Optional[pd.Series] = None,
                           include_deprecated: bool = False,
                           config_manager: Optional = None) -> Dict[str, MetricResult]:
    """Convenience function to compute enhanced metrics.
    
    Parameters
    ----------
    y_true : pd.Series
        True values
    y_pred : pd.Series
        Predicted values
    y_naive : pd.Series, optional
        Naive benchmark for scaling metrics
    include_deprecated : bool, default False
        Whether to include deprecated metrics with warnings
    config_manager : optional
        Configuration manager
        
    Returns
    -------
    dict
        Dictionary of MetricResult objects
    """
    calculator = EnhancedMetricsCalculator(config_manager)
    return calculator.compute_enhanced_metrics(y_true, y_pred, y_naive, include_deprecated)


def assess_metric_stability(metric_results: Dict[str, MetricResult]) -> Dict[str, int]:
    """Assess overall stability of metric results.
    
    Parameters
    ----------
    metric_results : dict
        Dictionary of MetricResult objects
        
    Returns
    -------
    dict
        Count of metrics by stability level
    """
    stability_counts = {
        'stable': 0,
        'robust': 0,
        'unstable': 0,
        'deprecated': 0
    }
    
    for result in metric_results.values():
        stability_counts[result.stability.value] += 1
    
    return stability_counts


def create_metrics_report(metric_results: Dict[str, MetricResult],
                         title: str = "Enhanced Metrics Report") -> str:
    """Create a formatted metrics report.
    
    Parameters
    ----------
    metric_results : dict
        Dictionary of MetricResult objects
    title : str
        Report title
        
    Returns
    -------
    str
        Formatted report string
    """
    lines = []
    lines.append("=" * len(title))
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    
    # Group by stability
    stable_metrics = [r for r in metric_results.values() if r.stability == MetricStability.STABLE]
    robust_metrics = [r for r in metric_results.values() if r.stability == MetricStability.ROBUST]
    deprecated_metrics = [r for r in metric_results.values() if r.stability == MetricStability.DEPRECATED]
    
    # Primary (stable) metrics
    if stable_metrics:
        lines.append("Primary Metrics (Stable):")
        lines.append("-" * 30)
        for result in stable_metrics:
            perf_str = f" ({result.performance_level.value})" if result.performance_level else ""
            lines.append(f"  {result.name.upper()}: {result.value:.4f}{perf_str}")
            if result.computation_warnings:
                for warning in result.computation_warnings:
                    lines.append(f"    ⚠ {warning}")
        lines.append("")
    
    # Secondary (robust) metrics
    if robust_metrics:
        lines.append("Secondary Metrics (Robust):")
        lines.append("-" * 30)
        for result in robust_metrics:
            perf_str = f" ({result.performance_level.value})" if result.performance_level else ""
            lines.append(f"  {result.name.upper()}: {result.value:.4f}{perf_str}")
        lines.append("")
    
    # Deprecated metrics (if any)
    if deprecated_metrics:
        lines.append("Deprecated Metrics (Not recommended for headline results):")
        lines.append("-" * 60)
        for result in deprecated_metrics:
            lines.append(f"  {result.name.upper()}: {result.value:.4f} [DEPRECATED]")
            for warning in result.computation_warnings:
                lines.append(f"    ⚠ {warning}")
        lines.append("")
    
    # Summary
    lines.append("Summary:")
    lines.append("-" * 15)
    lines.append(f"Total metrics computed: {len(metric_results)}")
    lines.append(f"Stable metrics: {len(stable_metrics)}")
    lines.append(f"Robust metrics: {len(robust_metrics)}")
    lines.append(f"Deprecated metrics: {len(deprecated_metrics)}")
    
    return "\n".join(lines)