"""Comprehensive backtesting evaluation pipeline.

This module provides a high-level interface for running complete backtesting
workflows, integrating rolling-origin cross-validation, metrics aggregation,
and statistical testing into a unified pipeline.

Features:
- End-to-end backtesting workflow
- Configuration system integration
- Model factory pattern support
- Benchmark model integration
- Comprehensive result reporting
- Error handling and recovery
- Progress tracking and logging
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Internal imports
from .rolling_origin import (
    RollingOriginValidator, 
    BacktestResult, 
    BacktestConfig,
    run_rolling_origin_backtest
)
from .metrics_aggregation import (
    MetricsAggregator,
    AggregatedMetrics,
    aggregate_fold_results,
    compute_diebold_mariano_combined,
    create_performance_summary,
    PValueCombination
)

# Configuration integration
try:
    from config import get_config, ConfigurationManager
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)


class BacktestError(Exception):
    """Exception raised during backtesting operations."""
    pass


@dataclass
class BacktestingPipeline:
    """Comprehensive backtesting evaluation pipeline."""
    
    def __init__(self, config_manager: Optional = None):
        """Initialize the backtesting pipeline.
        
        Parameters
        ----------
        config_manager : ConfigurationManager, optional
            Configuration manager for backtest settings
        """
        self.config_manager = config_manager
        
        # Load configuration if available
        if CONFIG_AVAILABLE and config_manager is None:
            try:
                self.config_manager = get_config()
                logger.info("Configuration manager loaded for backtesting pipeline")
            except Exception as e:
                logger.warning("Failed to load configuration manager: %s", e)
    
    def run_comprehensive_backtest(self,
                                 endog: pd.Series,
                                 model_factory: Callable,
                                 region: str,
                                 exog: Optional[pd.DataFrame] = None,
                                 benchmark_factory: Optional[Callable] = None,
                                 custom_config: Optional[BacktestConfig] = None,
                                 save_results: bool = True,
                                 results_dir: Optional[Path] = None) -> Tuple[BacktestResult, AggregatedMetrics]:
        """Run comprehensive backtesting evaluation.
        
        Parameters
        ----------
        endog : pd.Series
            Endogenous time series (target variable)
        model_factory : callable
            Function that creates and fits models: (endog_train, exog_train) -> fitted_model
        region : str
            Region identifier for configuration and reporting
        exog : pd.DataFrame, optional
            Exogenous variables
        benchmark_factory : callable, optional
            Function that creates benchmark forecasts: (endog_train, n_periods) -> forecasts
        custom_config : BacktestConfig, optional
            Custom backtesting configuration (overrides config file)
        save_results : bool, default True
            Whether to save detailed results to files
        results_dir : Path, optional
            Directory to save results (default: results/backtest/)
            
        Returns
        -------
        tuple
            (BacktestResult, AggregatedMetrics)
        """
        start_time = datetime.now()
        logger.info("Starting comprehensive backtesting for region %s", region)
        
        try:
            # Get configuration
            config = self._get_backtest_config(region, custom_config)
            logger.info("Using backtest configuration: %d folds, %d forecast horizon, %s window",
                       config.n_folds, config.forecast_horizon, config.window_type)
            
            # Validate inputs
            self._validate_inputs(endog, exog, model_factory)
            
            # Run rolling-origin cross-validation
            logger.info("Running rolling-origin cross-validation...")
            backtest_result = self._run_rolling_origin_cv(
                endog, model_factory, exog, benchmark_factory, config, region
            )
            
            logger.info("Cross-validation completed: %d/%d folds successful", 
                       backtest_result.successful_folds, config.n_folds)
            
            # Aggregate metrics across folds
            logger.info("Aggregating metrics across folds...")
            aggregated_metrics = self._aggregate_fold_metrics(backtest_result)
            
            # Add combined statistical tests
            self._add_combined_statistical_tests(backtest_result, aggregated_metrics)
            
            # Generate comprehensive summary
            summary = create_performance_summary(aggregated_metrics)
            logger.info("Backtesting summary:\n%s", summary)
            
            # Save results if requested
            if save_results:
                self._save_backtest_results(backtest_result, aggregated_metrics, region, results_dir)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info("Comprehensive backtesting completed in %.2f seconds", execution_time)
            
            return backtest_result, aggregated_metrics
            
        except Exception as e:
            logger.error("Comprehensive backtesting failed for region %s: %s", region, e)
            raise BacktestError(f"Backtesting failed for {region}: {e}") from e
    
    def _get_backtest_config(self, region: str, custom_config: Optional[BacktestConfig]) -> BacktestConfig:
        """Get backtesting configuration with region-specific overrides."""
        if custom_config is not None:
            return custom_config
        
        # Load from configuration manager
        if self.config_manager:
            try:
                config = BacktestConfig.from_config_manager(self.config_manager, region)
                logger.debug("Loaded backtest configuration from config manager")
                return config
            except Exception as e:
                logger.warning("Failed to load configuration for %s, using defaults: %s", region, e)
        
        # Use defaults
        return BacktestConfig()
    
    def _validate_inputs(self, endog: pd.Series, exog: Optional[pd.DataFrame], model_factory: Callable) -> None:
        """Validate inputs for backtesting."""
        if endog.empty:
            raise BacktestError("Endogenous series cannot be empty")
        
        if not isinstance(endog.index, pd.DatetimeIndex):
            raise BacktestError("Endogenous series must have DatetimeIndex")
        
        if exog is not None:
            if len(exog) != len(endog):
                raise BacktestError("Exogenous variables must have same length as endogenous series")
            
            if not isinstance(exog.index, pd.DatetimeIndex):
                raise BacktestError("Exogenous variables must have DatetimeIndex")
        
        if not callable(model_factory):
            raise BacktestError("Model factory must be callable")
    
    def _run_rolling_origin_cv(self, 
                              endog: pd.Series,
                              model_factory: Callable,
                              exog: Optional[pd.DataFrame],
                              benchmark_factory: Optional[Callable],
                              config: BacktestConfig,
                              region: str) -> BacktestResult:
        """Run the rolling-origin cross-validation."""
        validator = RollingOriginValidator(config)
        
        return validator.validate(
            endog=endog,
            model_func=model_factory,
            exog=exog,
            benchmark_func=benchmark_factory,
            region=region
        )
    
    def _aggregate_fold_metrics(self, backtest_result: BacktestResult) -> AggregatedMetrics:
        """Aggregate metrics across all folds."""
        # Get confidence levels from configuration
        confidence_levels = [80, 90, 95]
        if self.config_manager:
            try:
                config = self.config_manager.get_backtesting_config()
                eval_config = config.get('evaluation', {})
                confidence_levels = eval_config.get('confidence_levels', confidence_levels)
            except Exception:
                pass
        
        aggregator = MetricsAggregator(confidence_levels=confidence_levels)
        return aggregator.aggregate_fold_results(backtest_result.fold_results)
    
    def _add_combined_statistical_tests(self, backtest_result: BacktestResult, aggregated_metrics: AggregatedMetrics) -> None:
        """Add combined statistical tests to aggregated metrics."""
        # Combined Diebold-Mariano test
        dm_method = PValueCombination.FISHER
        if self.config_manager:
            try:
                config = self.config_manager.get_backtesting_config()
                method_str = config.get('evaluation', {}).get('diebold_mariano', {}).get('combination_method', 'fisher')
                dm_method = PValueCombination(method_str.lower())
            except Exception:
                pass
        
        combined_dm_test = compute_diebold_mariano_combined(backtest_result.fold_results, dm_method)
        if combined_dm_test:
            aggregated_metrics.statistical_tests['combined_diebold_mariano'] = combined_dm_test
    
    def _save_backtest_results(self,
                              backtest_result: BacktestResult,
                              aggregated_metrics: AggregatedMetrics,
                              region: str,
                              results_dir: Optional[Path]) -> None:
        """Save comprehensive backtest results to files."""
        if results_dir is None:
            results_dir = Path("results") / "backtest"
        
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Save fold-wise results
            fold_results_file = results_dir / f"{region}_fold_results_{timestamp}.csv"
            if aggregated_metrics.fold_metrics is not None:
                aggregated_metrics.fold_metrics.to_csv(fold_results_file)
                logger.info("Saved fold results to: %s", fold_results_file)
            
            # Save aggregated metrics
            agg_metrics_file = results_dir / f"{region}_aggregated_metrics_{timestamp}.json"
            import json
            
            agg_data = {
                'region': region,
                'timestamp': timestamp,
                'n_folds': aggregated_metrics.n_folds,
                'successful_folds': aggregated_metrics.successful_folds,
                'aggregation_method': aggregated_metrics.aggregation_method.value,
                'metrics': aggregated_metrics.metrics,
                'confidence_intervals': {
                    metric: {str(level): [float(ci[0]), float(ci[1])] for level, ci in levels.items()}
                    for metric, levels in aggregated_metrics.confidence_intervals.items()
                },
                'statistical_tests': {
                    name: {
                        'test_name': test.test_name,
                        'statistic': test.statistic,
                        'p_value': test.p_value,
                        'significance_level': test.significance_level,
                        'is_significant': test.is_significant
                    }
                    for name, test in aggregated_metrics.statistical_tests.items()
                }
            }
            
            with open(agg_metrics_file, 'w') as f:
                json.dump(agg_data, f, indent=2)
            logger.info("Saved aggregated metrics to: %s", agg_metrics_file)
            
            # Save performance summary
            summary_file = results_dir / f"{region}_performance_summary_{timestamp}.txt"
            summary = create_performance_summary(aggregated_metrics)
            with open(summary_file, 'w') as f:
                f.write(summary)
            logger.info("Saved performance summary to: %s", summary_file)
            
        except Exception as e:
            logger.error("Failed to save backtest results: %s", e)


def run_comprehensive_backtest(endog: pd.Series,
                             model_factory: Callable,
                             region: str,
                             exog: Optional[pd.DataFrame] = None,
                             benchmark_factory: Optional[Callable] = None,
                             config_manager: Optional = None,
                             custom_config: Optional[BacktestConfig] = None,
                             save_results: bool = True) -> Tuple[BacktestResult, AggregatedMetrics]:
    """Convenience function for comprehensive backtesting.
    
    Parameters
    ----------
    endog : pd.Series
        Endogenous time series
    model_factory : callable
        Model factory function
    region : str
        Region identifier
    exog : pd.DataFrame, optional
        Exogenous variables
    benchmark_factory : callable, optional
        Benchmark model factory
    config_manager : optional
        Configuration manager
    custom_config : BacktestConfig, optional
        Custom configuration
    save_results : bool
        Whether to save results
        
    Returns
    -------
    tuple
        (BacktestResult, AggregatedMetrics)
    """
    pipeline = BacktestingPipeline(config_manager)
    return pipeline.run_comprehensive_backtest(
        endog=endog,
        model_factory=model_factory,
        region=region,
        exog=exog,
        benchmark_factory=benchmark_factory,
        custom_config=custom_config,
        save_results=save_results
    )


def create_sarimax_model_factory(order: Tuple[int, int, int],
                                seasonal_order: Tuple[int, int, int, int],
                                robust_errors: bool = True) -> Callable:
    """Create a SARIMAX model factory function.
    
    Parameters
    ----------
    order : tuple
        SARIMAX order (p, d, q)
    seasonal_order : tuple
        Seasonal order (P, D, Q, s)
    robust_errors : bool
        Whether to use robust standard errors
        
    Returns
    -------
    callable
        Model factory function
    """
    def model_factory(endog_train: pd.Series, exog_train: Optional[pd.DataFrame]) -> SARIMAX:
        """SARIMAX model factory."""
        fit_kwargs = {"disp": False}
        if robust_errors:
            fit_kwargs["cov_type"] = "robust"
        
        model = SARIMAX(
            endog_train,
            exog=exog_train,
            order=order,
            seasonal_order=seasonal_order,
            simple_differencing=False
        )
        
        return model.fit(**fit_kwargs)
    
    return model_factory


def create_naive_benchmark_factory() -> Callable:
    """Create a naive benchmark forecast factory.
    
    Returns
    -------
    callable
        Naive benchmark factory function
    """
    def naive_benchmark(endog_train: pd.Series, n_periods: int) -> pd.Series:
        """Naive benchmark: last value carried forward."""
        last_value = endog_train.iloc[-1]
        forecast_index = pd.date_range(
            start=endog_train.index[-1] + endog_train.index.freq,
            periods=n_periods,
            freq=endog_train.index.freq
        )
        return pd.Series([last_value] * n_periods, index=forecast_index, name='naive_forecast')
    
    return naive_benchmark