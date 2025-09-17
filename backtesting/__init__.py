"""Rolling-origin cross-validation backtesting for GDP-ForecasterSARIMAX.

This package provides comprehensive time series backtesting including:
- Rolling-origin cross-validation with configurable folds
- Strict out-of-sample forecast evaluation
- Per-fold and aggregated performance metrics
- Statistical significance testing (Diebold-Mariano)
- Integration with configuration system
- Prevention of data leakage
"""

from .rolling_origin import (
    RollingOriginValidator,
    BacktestResult,
    FoldResult,
    BacktestConfig,
    run_rolling_origin_backtest
)

from .metrics_aggregation import (
    MetricsAggregator,
    AggregatedMetrics,
    StatisticalTest,
    AggregationMethod,
    PValueCombination,
    aggregate_fold_results,
    compute_diebold_mariano_combined,
    combine_pvalues,
    create_performance_summary
)

from .evaluation_pipeline import (
    BacktestingPipeline,
    BacktestError,
    run_comprehensive_backtest,
    create_sarimax_model_factory,
    create_naive_benchmark_factory
)

__all__ = [
    # Core backtesting
    'RollingOriginValidator',
    'BacktestResult',
    'FoldResult', 
    'BacktestConfig',
    'run_rolling_origin_backtest',
    
    # Metrics aggregation
    'MetricsAggregator',
    'AggregatedMetrics',
    'StatisticalTest',
    'AggregationMethod',
    'PValueCombination',
    'aggregate_fold_results',
    'compute_diebold_mariano_combined',
    'combine_pvalues',
    'create_performance_summary',
    
    # Pipeline
    'BacktestingPipeline',
    'BacktestError', 
    'run_comprehensive_backtest',
    'create_sarimax_model_factory',
    'create_naive_benchmark_factory'
]

# Version info
__version__ = '1.0.0'