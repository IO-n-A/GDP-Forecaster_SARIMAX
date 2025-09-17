"""Enhanced evaluation metrics for GDP-ForecasterSARIMAX.

This package provides refined evaluation metrics that address instability issues
with MAPE/sMAPE near zero values and prioritizes robust metrics for economic
forecasting.

Key Features:
- Stable primary metrics: MAE, RMSE, MASE, Theil's U
- Robust secondary metrics: Median AE, Directional Accuracy
- Deprecation handling for unstable metrics (MAPE/sMAPE)
- Performance thresholds and quality assessment
- Integration with configuration system
- Enhanced reporting with metric prioritization
"""

from .enhanced_metrics import (
    EnhancedMetricsCalculator,
    MetricResult,
    MetricStability,
    PerformanceLevel,
    compute_enhanced_metrics,
    assess_metric_stability,
    create_metrics_report
)

from .metric_refinement import (
    MetricRefinementEngine,
    RefinedMetricsResult,
    PerformanceAssessment,
    refine_evaluation_metrics,
    filter_stable_metrics,
    create_refinement_report
)

__all__ = [
    # Enhanced metrics computation
    'EnhancedMetricsCalculator',
    'MetricResult',
    'MetricStability',
    'PerformanceLevel',
    'compute_enhanced_metrics',
    'assess_metric_stability',
    'create_metrics_report',
    
    # Metric refinement
    'MetricRefinementEngine',
    'RefinedMetricsResult',
    'PerformanceAssessment',
    'refine_evaluation_metrics',
    'filter_stable_metrics',
    'create_refinement_report'
]

# Version info
__version__ = '1.0.0'