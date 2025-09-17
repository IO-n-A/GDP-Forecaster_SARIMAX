"""Metric refinement engine for filtering stable metrics and handling deprecation.

This module provides tools to refine evaluation metrics by filtering out unstable
metrics, handling deprecation appropriately, and integrating with existing
evaluation workflows.

Features:
- Filter metrics by stability level
- Handle metric deprecation with appropriate warnings
- Performance assessment and ranking
- Integration with existing econ_eval module
- Backward compatibility with legacy metrics
- Enhanced reporting with stability annotations
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

# Import existing evaluation function for backward compatibility
try:
    from econ_eval import compute_metrics as legacy_compute_metrics
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Legacy econ_eval module not available")

# Configuration integration
try:
    from config import get_config, ConfigurationManager
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

from .enhanced_metrics import (
    EnhancedMetricsCalculator, 
    MetricResult, 
    MetricStability, 
    PerformanceLevel
)

logger = logging.getLogger(__name__)


@dataclass
class PerformanceAssessment:
    """Overall performance assessment for a model."""
    
    primary_metric_score: float
    primary_metric_name: str
    performance_level: PerformanceLevel
    
    # Detailed breakdown
    stable_metrics_count: int
    robust_metrics_count: int
    deprecated_metrics_count: int
    
    # Rankings
    overall_score: float  # Composite score based on multiple metrics
    stability_score: float  # Score based on metric stability
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass 
class RefinedMetricsResult:
    """Result from refined metrics computation."""
    
    # Refined metrics (stable + robust only)
    refined_metrics: Dict[str, MetricResult]
    
    # Legacy metrics (for backward compatibility) 
    legacy_metrics: Optional[Dict[str, float]] = None
    
    # Deprecated metrics (with warnings)
    deprecated_metrics: Optional[Dict[str, MetricResult]] = None
    
    # Overall assessment
    performance_assessment: Optional[PerformanceAssessment] = None
    
    # Integration metadata
    computation_metadata: Dict[str, Any] = field(default_factory=dict)


class MetricRefinementEngine:
    """Engine for refining evaluation metrics and filtering unstable metrics."""
    
    def __init__(self, config_manager: Optional = None):
        """Initialize the metric refinement engine.
        
        Parameters
        ----------
        config_manager : ConfigurationManager, optional
            Configuration manager for metric settings
        """
        self.config_manager = config_manager
        self.enhanced_calculator = EnhancedMetricsCalculator(config_manager)
        
        # Load refinement configuration
        self.refinement_config = self._load_refinement_config()
    
    def _load_refinement_config(self) -> Dict[str, Any]:
        """Load metric refinement configuration."""
        default_config = {
            'include_deprecated_by_default': False,
            'warn_on_deprecated_usage': True,
            'stability_threshold': MetricStability.ROBUST,
            'performance_weights': {
                'mae': 0.3,
                'rmse': 0.3,
                'mase': 0.25,
                'theil_u1': 0.15
            },
            'enable_legacy_integration': True
        }
        
        if self.config_manager and CONFIG_AVAILABLE:
            try:
                eval_config = self.config_manager.get_evaluation_config()
                output_config = eval_config.get('output', {})
                
                # Update configuration from file
                default_config['include_deprecated_by_default'] = 'mape' in output_config.get('summary_metrics', [])
                
                logger.debug("Loaded refinement configuration from config manager")
                
            except Exception as e:
                logger.warning("Failed to load refinement config, using defaults: %s", e)
        
        return default_config
    
    def refine_evaluation_metrics(self,
                                 y_true: pd.Series,
                                 y_pred: pd.Series,
                                 y_naive: Optional[pd.Series] = None,
                                 include_legacy: bool = True,
                                 stability_filter: Optional[MetricStability] = None) -> RefinedMetricsResult:
        """Refine evaluation metrics by filtering stable metrics and handling deprecation.
        
        Parameters
        ----------
        y_true : pd.Series
            True values
        y_pred : pd.Series
            Predicted values
        y_naive : pd.Series, optional
            Naive benchmark for scaling metrics
        include_legacy : bool, default True
            Whether to include legacy metric computation for backward compatibility
        stability_filter : MetricStability, optional
            Minimum stability level to include (default: ROBUST)
            
        Returns
        -------
        RefinedMetricsResult
            Comprehensive refined metrics result
        """
        logger.info("Refining evaluation metrics for %d observations", len(y_true))
        
        if stability_filter is None:
            stability_filter = self.refinement_config['stability_threshold']
        
        # Compute all metrics (including deprecated with warnings)
        include_deprecated = self.refinement_config['include_deprecated_by_default']
        all_metrics = self.enhanced_calculator.compute_enhanced_metrics(
            y_true, y_pred, y_naive, include_deprecated=True
        )
        
        # Separate metrics by stability
        refined_metrics = {}
        deprecated_metrics = {}
        
        for name, result in all_metrics.items():
            if result.stability.value in ['stable', 'robust']:
                refined_metrics[name] = result
            elif result.stability == MetricStability.DEPRECATED:
                deprecated_metrics[name] = result
                if self.refinement_config['warn_on_deprecated_usage']:
                    warnings.warn(f"Metric '{name}' is deprecated and excluded from headline results",
                                DeprecationWarning, stacklevel=2)
        
        # Compute legacy metrics for backward compatibility
        legacy_metrics = None
        if include_legacy and self.refinement_config['enable_legacy_integration']:
            legacy_metrics = self._compute_legacy_metrics(y_true, y_pred)
        
        # Perform performance assessment
        performance_assessment = self._assess_overall_performance(refined_metrics, deprecated_metrics)
        
        # Compilation metadata
        metadata = {
            'total_metrics_computed': len(all_metrics),
            'refined_metrics_count': len(refined_metrics),
            'deprecated_metrics_count': len(deprecated_metrics),
            'stability_filter': stability_filter.value,
            'legacy_integration_enabled': include_legacy and self.refinement_config['enable_legacy_integration']
        }
        
        result = RefinedMetricsResult(
            refined_metrics=refined_metrics,
            legacy_metrics=legacy_metrics,
            deprecated_metrics=deprecated_metrics if deprecated_metrics else None,
            performance_assessment=performance_assessment,
            computation_metadata=metadata
        )
        
        logger.info("Metric refinement completed: %d refined metrics, %d deprecated metrics",
                   len(refined_metrics), len(deprecated_metrics))
        
        return result
    
    def _compute_legacy_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> Optional[Dict[str, float]]:
        """Compute legacy metrics for backward compatibility."""
        if not LEGACY_AVAILABLE:
            logger.warning("Legacy econ_eval module not available")
            return None
        
        try:
            # Convert to numpy arrays for legacy function
            y_true_array = y_true.values
            y_pred_array = y_pred.values
            
            legacy_result = legacy_compute_metrics(y_true_array, y_pred_array)
            
            # Add deprecation warnings for unstable metrics
            if 'MAPE' in legacy_result:
                warnings.warn("Legacy MAPE metric is deprecated - use enhanced metrics instead",
                            DeprecationWarning, stacklevel=3)
            
            logger.debug("Computed %d legacy metrics for backward compatibility", len(legacy_result))
            return legacy_result
            
        except Exception as e:
            logger.error("Failed to compute legacy metrics: %s", e)
            return None
    
    def _assess_overall_performance(self,
                                  refined_metrics: Dict[str, MetricResult],
                                  deprecated_metrics: Dict[str, MetricResult]) -> PerformanceAssessment:
        """Assess overall model performance based on refined metrics."""
        if not refined_metrics:
            logger.warning("No refined metrics available for performance assessment")
            return PerformanceAssessment(
                primary_metric_score=np.inf,
                primary_metric_name="none",
                performance_level=PerformanceLevel.POOR,
                stable_metrics_count=0,
                robust_metrics_count=0,
                deprecated_metrics_count=len(deprecated_metrics),
                overall_score=0.0,
                stability_score=0.0,
                recommendations=["No stable metrics available for assessment"],
                warnings=["Insufficient metrics for reliable performance assessment"]
            )
        
        # Get primary metric (usually MAE)
        primary_metric_name = 'mae'
        if primary_metric_name not in refined_metrics:
            primary_metric_name = list(refined_metrics.keys())[0]
        
        primary_metric = refined_metrics[primary_metric_name]
        
        # Count metrics by stability
        stable_count = sum(1 for m in refined_metrics.values() if m.stability == MetricStability.STABLE)
        robust_count = sum(1 for m in refined_metrics.values() if m.stability == MetricStability.ROBUST)
        
        # Compute composite score
        overall_score = self._compute_composite_score(refined_metrics)
        
        # Compute stability score
        total_refined = len(refined_metrics)
        stability_score = (stable_count * 1.0 + robust_count * 0.8) / total_refined if total_refined > 0 else 0.0
        
        # Generate recommendations
        recommendations = []
        warnings_list = []
        
        if primary_metric.performance_level == PerformanceLevel.EXCELLENT:
            recommendations.append("Model shows excellent performance on primary metrics")
        elif primary_metric.performance_level == PerformanceLevel.POOR:
            recommendations.append("Consider model re-specification or additional features")
            warnings_list.append("Poor performance on primary metric")
        
        if stability_score < 0.7:
            warnings_list.append("Limited availability of stable metrics")
            recommendations.append("Consider computing additional stable metrics for robustness")
        
        if len(deprecated_metrics) > 0:
            warnings_list.append(f"{len(deprecated_metrics)} deprecated metrics excluded from assessment")
            recommendations.append("Transition to stable metrics for consistent evaluation")
        
        return PerformanceAssessment(
            primary_metric_score=primary_metric.value,
            primary_metric_name=primary_metric_name,
            performance_level=primary_metric.performance_level or PerformanceLevel.ACCEPTABLE,
            stable_metrics_count=stable_count,
            robust_metrics_count=robust_count,
            deprecated_metrics_count=len(deprecated_metrics),
            overall_score=overall_score,
            stability_score=stability_score,
            recommendations=recommendations,
            warnings=warnings_list
        )
    
    def _compute_composite_score(self, metrics: Dict[str, MetricResult]) -> float:
        """Compute composite performance score from multiple metrics."""
        weights = self.refinement_config['performance_weights']
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric_name, weight in weights.items():
            if metric_name in metrics:
                metric_result = metrics[metric_name]
                
                # Normalize metric value (lower is better for error metrics)
                if metric_name in ['mae', 'rmse', 'mse']:
                    # Inverse score for error metrics
                    normalized_score = 1.0 / (1.0 + metric_result.value)
                elif metric_name in ['mase', 'theil_u1', 'theil_u2']:
                    # For scale-independent metrics, 1.0 is neutral
                    normalized_score = 2.0 / (1.0 + metric_result.value)
                else:
                    # Default normalization
                    normalized_score = 1.0 / (1.0 + metric_result.value)
                
                weighted_score += weight * normalized_score
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_score / total_weight


def refine_evaluation_metrics(y_true: pd.Series,
                            y_pred: pd.Series,
                            y_naive: Optional[pd.Series] = None,
                            config_manager: Optional = None,
                            include_legacy: bool = True) -> RefinedMetricsResult:
    """Convenience function to refine evaluation metrics.
    
    Parameters
    ----------
    y_true : pd.Series
        True values
    y_pred : pd.Series
        Predicted values
    y_naive : pd.Series, optional
        Naive benchmark
    config_manager : optional
        Configuration manager
    include_legacy : bool, default True
        Include legacy metrics for backward compatibility
        
    Returns
    -------
    RefinedMetricsResult
        Refined metrics result
    """
    engine = MetricRefinementEngine(config_manager)
    return engine.refine_evaluation_metrics(y_true, y_pred, y_naive, include_legacy)


def filter_stable_metrics(metric_results: Dict[str, MetricResult],
                         min_stability: MetricStability = MetricStability.ROBUST) -> Dict[str, MetricResult]:
    """Filter metrics by minimum stability level.
    
    Parameters
    ----------
    metric_results : dict
        Dictionary of MetricResult objects
    min_stability : MetricStability, default ROBUST
        Minimum stability level to include
        
    Returns
    -------
    dict
        Filtered metrics dictionary
    """
    stability_order = {
        MetricStability.STABLE: 3,
        MetricStability.ROBUST: 2,
        MetricStability.UNSTABLE: 1,
        MetricStability.DEPRECATED: 0
    }
    
    min_level = stability_order[min_stability]
    
    filtered_metrics = {}
    for name, result in metric_results.items():
        if stability_order.get(result.stability, 0) >= min_level:
            filtered_metrics[name] = result
    
    logger.debug("Filtered %d metrics to %d based on stability >= %s",
                len(metric_results), len(filtered_metrics), min_stability.value)
    
    return filtered_metrics


def create_refinement_report(result: RefinedMetricsResult,
                           title: str = "Refined Metrics Evaluation Report") -> str:
    """Create a comprehensive refinement report.
    
    Parameters
    ----------
    result : RefinedMetricsResult
        Refined metrics result
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
    
    # Performance Assessment
    if result.performance_assessment:
        perf = result.performance_assessment
        lines.append("Performance Assessment:")
        lines.append("-" * 25)
        lines.append(f"Primary Metric ({perf.primary_metric_name.upper()}): {perf.primary_metric_score:.4f}")
        lines.append(f"Performance Level: {perf.performance_level.value.upper()}")
        lines.append(f"Overall Score: {perf.overall_score:.3f}")
        lines.append(f"Stability Score: {perf.stability_score:.3f}")
        lines.append("")
    
    # Metric Counts
    lines.append("Metrics Summary:")
    lines.append("-" * 20)
    lines.append(f"Refined (Stable + Robust): {len(result.refined_metrics)}")
    if result.deprecated_metrics:
        lines.append(f"Deprecated: {len(result.deprecated_metrics)}")
    if result.legacy_metrics:
        lines.append(f"Legacy (Backward Compatibility): {len(result.legacy_metrics)}")
    lines.append("")
    
    # Refined Metrics
    if result.refined_metrics:
        lines.append("Refined Metrics (Recommended for Headline Results):")
        lines.append("-" * 55)
        for name, metric in result.refined_metrics.items():
            perf_str = f" ({metric.performance_level.value})" if metric.performance_level else ""
            stability_str = f"[{metric.stability.value.upper()}]"
            lines.append(f"  {name.upper()}: {metric.value:.4f}{perf_str} {stability_str}")
        lines.append("")
    
    # Deprecated Metrics (if included)
    if result.deprecated_metrics:
        lines.append("Deprecated Metrics (Not Recommended):")
        lines.append("-" * 40)
        for name, metric in result.deprecated_metrics.items():
            lines.append(f"  {name.upper()}: {metric.value:.4f} [DEPRECATED]")
            for warning in metric.computation_warnings:
                lines.append(f"    ⚠ {warning}")
        lines.append("")
    
    # Recommendations and Warnings
    if result.performance_assessment:
        perf = result.performance_assessment
        if perf.recommendations:
            lines.append("Recommendations:")
            lines.append("-" * 16)
            for rec in perf.recommendations:
                lines.append(f"  • {rec}")
            lines.append("")
        
        if perf.warnings:
            lines.append("Warnings:")
            lines.append("-" * 10)
            for warn in perf.warnings:
                lines.append(f"  ⚠ {warn}")
            lines.append("")
    
    # Metadata
    if result.computation_metadata:
        lines.append("Computation Metadata:")
        lines.append("-" * 22)
        for key, value in result.computation_metadata.items():
            lines.append(f"  {key}: {value}")
    
    return "\n".join(lines)