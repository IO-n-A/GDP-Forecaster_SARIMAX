"""Rolling-origin cross-validation for time series forecasting.

This module implements robust rolling-origin cross-validation for time series
models, ensuring strict out-of-sample evaluation and preventing data leakage.

Features:
- Configurable number of folds with expanding/rolling windows
- Minimum training window size enforcement
- Multiple forecast horizons per fold
- Integration with configuration system
- Comprehensive result tracking per fold
- Statistical significance testing preparation
- SARIMAX model integration
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Configuration integration
try:
    from config import get_config, ConfigurationManager
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# Metrics computation
from econ_eval import compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for rolling-origin backtesting."""
    
    # Core parameters
    n_folds: int = 8                    # Number of CV folds
    min_train_size: int = 40            # Minimum training observations
    forecast_horizon: int = 8           # Steps ahead to forecast
    step_size: int = 1                  # Steps between fold origins
    
    # Window configuration
    window_type: str = "expanding"      # "expanding" or "rolling"
    max_train_size: Optional[int] = None # Max training size for rolling window
    
    # Model retraining
    refit: bool = True                  # Refit model at each fold
    reoptimize: bool = False            # Re-optimize hyperparameters at each fold
    
    # Evaluation settings
    metrics: List[str] = field(default_factory=lambda: [
        'mae', 'rmse', 'mase', 'theil_u1', 'directional_accuracy'
    ])
    confidence_intervals: List[int] = field(default_factory=lambda: [80, 95])
    
    # Statistical tests
    diebold_mariano: bool = True
    significance_level: float = 0.05
    
    @classmethod
    def from_config_manager(cls, config_manager: Optional = None, region: Optional[str] = None) -> 'BacktestConfig':
        """Create BacktestConfig from configuration manager.
        
        Parameters
        ----------
        config_manager : ConfigurationManager, optional
            Configuration manager instance
        region : str, optional
            Region for region-specific overrides
            
        Returns
        -------
        BacktestConfig
            Configured backtest configuration
        """
        config = cls()  # Start with defaults
        
        if config_manager:
            try:
                # Get base backtesting configuration
                base_config = config_manager.get_backtesting_config()
                rolling_config = base_config.get('rolling_origin', {})
                eval_config = base_config.get('evaluation', {})
                
                # Apply base configuration
                config.min_train_size = rolling_config.get('min_train_size', config.min_train_size)
                config.forecast_horizon = rolling_config.get('forecast_horizon', config.forecast_horizon)
                config.step_size = rolling_config.get('step_size', config.step_size)
                config.window_type = rolling_config.get('window_type', config.window_type)
                config.max_train_size = rolling_config.get('max_train_size', config.max_train_size)
                config.refit = rolling_config.get('refit', config.refit)
                config.reoptimize = rolling_config.get('reoptimize', config.reoptimize)
                
                # Get cross-validation settings
                cv_config = base_config.get('cross_validation', {})
                config.n_folds = cv_config.get('n_splits', config.n_folds)
                
                # Get evaluation settings
                config.metrics = eval_config.get('metrics', config.metrics)
                config.significance_level = base_config.get('robustness', {}).get('structural_breaks', {}).get('significance_level', config.significance_level)
                
                # Apply region-specific overrides
                if region:
                    region_overrides = base_config.get('regional_overrides', {}).get(region, {})
                    if region_overrides:
                        config.min_train_size = region_overrides.get('min_train_size', config.min_train_size)
                        config.forecast_horizon = region_overrides.get('forecast_horizon', config.forecast_horizon)
                        
                logger.info("Loaded backtest configuration from config manager")
                
            except Exception as e:
                logger.warning("Failed to load configuration, using defaults: %s", e)
        
        return config


@dataclass
class FoldResult:
    """Results from a single backtest fold."""
    
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_size: int
    test_size: int
    
    # Forecasts and actuals
    forecasts: pd.Series
    actuals: pd.Series
    forecast_intervals: Optional[Dict[int, Tuple[pd.Series, pd.Series]]] = None
    
    # Model information
    model_params: Optional[Dict[str, Any]] = None
    model_summary: Optional[str] = None
    convergence_info: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    metrics: Optional[Dict[str, float]] = None
    
    # Statistical tests
    diebold_mariano_stat: Optional[float] = None
    diebold_mariano_pvalue: Optional[float] = None
    
    # Timing information
    fit_time: Optional[float] = None
    forecast_time: Optional[float] = None
    
    def __post_init__(self):
        """Validate fold result consistency."""
        if len(self.forecasts) != len(self.actuals):
            raise ValueError("Forecasts and actuals must have same length")
        
        if self.test_size != len(self.forecasts):
            logger.warning("Test size inconsistent with forecast length: %d vs %d", 
                         self.test_size, len(self.forecasts))


@dataclass
class BacktestResult:
    """Complete results from rolling-origin backtesting."""
    
    config: BacktestConfig
    fold_results: List[FoldResult]
    
    # Aggregated metrics
    aggregated_metrics: Optional[Dict[str, Dict[str, float]]] = None  # metric -> {mean, std, etc.}
    combined_forecasts: Optional[pd.Series] = None
    combined_actuals: Optional[pd.Series] = None
    
    # Overall statistical tests
    combined_dm_stat: Optional[float] = None
    combined_dm_pvalue: Optional[float] = None
    
    # Execution metadata
    total_execution_time: Optional[float] = None
    successful_folds: Optional[int] = None
    failed_folds: Optional[List[int]] = None
    
    @property
    def n_folds(self) -> int:
        """Number of completed folds."""
        return len(self.fold_results)
    
    @property
    def success_rate(self) -> float:
        """Proportion of successful folds."""
        if self.successful_folds is None:
            return 1.0 if self.fold_results else 0.0
        return self.successful_folds / self.config.n_folds
    
    def get_metric_series(self, metric_name: str) -> pd.Series:
        """Get time series of a metric across folds."""
        values = []
        fold_ids = []
        
        for fold in self.fold_results:
            if fold.metrics and metric_name in fold.metrics:
                values.append(fold.metrics[metric_name])
                fold_ids.append(fold.fold_id)
        
        return pd.Series(values, index=fold_ids, name=metric_name)


class RollingOriginValidator:
    """Rolling-origin cross-validator for time series models."""
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """Initialize the validator.
        
        Parameters
        ----------
        config : BacktestConfig, optional
            Backtesting configuration. If None, uses defaults.
        """
        self.config = config or BacktestConfig()
        
        # Load configuration from config manager if available
        if CONFIG_AVAILABLE and config is None:
            try:
                config_manager = get_config()
                self.config = BacktestConfig.from_config_manager(config_manager)
            except Exception as e:
                logger.warning("Failed to load config manager, using defaults: %s", e)
    
    def validate(self, 
                endog: pd.Series,
                model_func: Callable,
                exog: Optional[pd.DataFrame] = None,
                benchmark_func: Optional[Callable] = None,
                region: Optional[str] = None) -> BacktestResult:
        """Run rolling-origin cross-validation.
        
        Parameters
        ----------
        endog : pd.Series
            Endogenous time series (target variable)
        model_func : callable
            Function that takes (endog_train, exog_train) and returns fitted model
        exog : pd.DataFrame, optional
            Exogenous variables
        benchmark_func : callable, optional
            Function for benchmark model (e.g., naive forecast)
        region : str, optional
            Region identifier for logging
            
        Returns
        -------
        BacktestResult
            Complete backtesting results
        """
        start_time = datetime.now()
        region_str = f" for {region}" if region else ""
        logger.info("Starting rolling-origin cross-validation%s with %d folds", 
                   region_str, self.config.n_folds)
        
        # Validate inputs
        self._validate_inputs(endog, exog)
        
        # Generate fold boundaries
        fold_boundaries = self._generate_fold_boundaries(endog)
        logger.info("Generated %d fold boundaries", len(fold_boundaries))
        
        # Run folds
        fold_results = []
        failed_folds = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(fold_boundaries):
            try:
                logger.debug("Running fold %d: train=%s to %s, test=%s to %s", 
                           i+1, train_start.date(), train_end.date(), 
                           test_start.date(), test_end.date())
                
                fold_result = self._run_single_fold(
                    fold_id=i+1,
                    endog=endog,
                    exog=exog,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    model_func=model_func,
                    benchmark_func=benchmark_func
                )
                
                fold_results.append(fold_result)
                logger.debug("Fold %d completed successfully", i+1)
                
            except Exception as e:
                logger.error("Fold %d failed: %s", i+1, e)
                failed_folds.append(i+1)
                continue
        
        # Aggregate results
        execution_time = (datetime.now() - start_time).total_seconds()
        
        result = BacktestResult(
            config=self.config,
            fold_results=fold_results,
            total_execution_time=execution_time,
            successful_folds=len(fold_results),
            failed_folds=failed_folds if failed_folds else None
        )
        
        # Compute aggregated metrics and statistical tests
        self._compute_aggregated_results(result, benchmark_func is not None)
        
        logger.info("Rolling-origin cross-validation completed: %d/%d folds successful (%.1f%%)",
                   len(fold_results), self.config.n_folds, result.success_rate * 100)
        
        return result
    
    def _validate_inputs(self, endog: pd.Series, exog: Optional[pd.DataFrame]) -> None:
        """Validate input data for backtesting."""
        if endog.empty:
            raise ValueError("Endogenous series cannot be empty")
        
        if not isinstance(endog.index, pd.DatetimeIndex):
            logger.warning("Endogenous series does not have DatetimeIndex")
        
        total_obs = len(endog)
        min_required = self.config.min_train_size + self.config.forecast_horizon * self.config.n_folds
        
        if total_obs < min_required:
            raise ValueError(f"Insufficient data: {total_obs} obs, need at least {min_required}")
        
        if exog is not None:
            if len(exog) != len(endog):
                raise ValueError("Exogenous variables must have same length as endogenous series")
    
    def _generate_fold_boundaries(self, endog: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate fold boundaries for rolling-origin CV."""
        boundaries = []
        
        total_obs = len(endog)
        min_train = self.config.min_train_size
        horizon = self.config.forecast_horizon
        step = self.config.step_size
        
        # Start with minimum training size
        for fold in range(self.config.n_folds):
            # Calculate training window
            if self.config.window_type == "expanding":
                train_start_idx = 0
                train_end_idx = min_train + fold * step - 1
            else:  # rolling
                train_size = self.config.max_train_size or min_train
                train_end_idx = min_train + fold * step - 1
                train_start_idx = max(0, train_end_idx - train_size + 1)
            
            # Calculate test window
            test_start_idx = train_end_idx + 1
            test_end_idx = min(test_start_idx + horizon - 1, total_obs - 1)
            
            # Check if we have enough data for this fold
            if test_end_idx >= total_obs:
                logger.warning("Fold %d would exceed data bounds, stopping at %d folds", 
                             fold + 1, fold)
                break
            
            if test_end_idx < test_start_idx:
                logger.warning("Invalid test window for fold %d, stopping", fold + 1)
                break
            
            # Convert indices to timestamps
            train_start = endog.index[train_start_idx]
            train_end = endog.index[train_end_idx]
            test_start = endog.index[test_start_idx]
            test_end = endog.index[test_end_idx]
            
            boundaries.append((train_start, train_end, test_start, test_end))
        
        return boundaries
    
    def _run_single_fold(self,
                        fold_id: int,
                        endog: pd.Series,
                        exog: Optional[pd.DataFrame],
                        train_start: pd.Timestamp,
                        train_end: pd.Timestamp,
                        test_start: pd.Timestamp,
                        test_end: pd.Timestamp,
                        model_func: Callable,
                        benchmark_func: Optional[Callable]) -> FoldResult:
        """Run a single fold of cross-validation."""
        import time
        
        # Split data
        endog_train = endog[train_start:train_end]
        endog_test = endog[test_start:test_end]
        
        exog_train = None
        exog_test = None
        if exog is not None:
            exog_train = exog[train_start:train_end]
            exog_test = exog[test_start:test_end]
        
        # Fit model
        fit_start = time.time()
        try:
            model = model_func(endog_train, exog_train)
            model_params = self._extract_model_params(model)
            model_summary = str(model.summary()) if hasattr(model, 'summary') else None
        except Exception as e:
            logger.error("Model fitting failed in fold %d: %s", fold_id, e)
            raise
        
        fit_time = time.time() - fit_start
        
        # Generate forecasts
        forecast_start = time.time()
        try:
            forecast_result = model.get_forecast(steps=len(endog_test), exog=exog_test)
            forecasts = pd.Series(
                forecast_result.predicted_mean,
                index=endog_test.index,
                name='forecast'
            )
            
            # Get confidence intervals
            conf_int = forecast_result.conf_int()
            forecast_intervals = {}
            for ci in self.config.confidence_intervals:
                alpha = 1 - (ci / 100)
                ci_result = model.get_forecast(steps=len(endog_test), exog=exog_test).conf_int(alpha=alpha)
                forecast_intervals[ci] = (
                    pd.Series(ci_result.iloc[:, 0], index=endog_test.index),
                    pd.Series(ci_result.iloc[:, 1], index=endog_test.index)
                )
                
        except Exception as e:
            logger.error("Forecasting failed in fold %d: %s", fold_id, e)
            raise
        
        forecast_time = time.time() - forecast_start
        
        # Compute metrics
        metrics = compute_metrics(endog_test, forecasts)
        
        # Compute Diebold-Mariano test if benchmark available
        dm_stat = None
        dm_pvalue = None
        if benchmark_func is not None:
            try:
                benchmark_forecasts = benchmark_func(endog_train, len(endog_test))
                dm_stat, dm_pvalue = self._diebold_mariano_test(endog_test, forecasts, benchmark_forecasts)
            except Exception as e:
                logger.warning("Diebold-Mariano test failed in fold %d: %s", fold_id, e)
        
        return FoldResult(
            fold_id=fold_id,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_size=len(endog_train),
            test_size=len(endog_test),
            forecasts=forecasts,
            actuals=endog_test,
            forecast_intervals=forecast_intervals,
            model_params=model_params,
            model_summary=model_summary,
            metrics=metrics,
            diebold_mariano_stat=dm_stat,
            diebold_mariano_pvalue=dm_pvalue,
            fit_time=fit_time,
            forecast_time=forecast_time
        )
    
    def _extract_model_params(self, model) -> Dict[str, Any]:
        """Extract parameters from fitted model."""
        params = {}
        try:
            if hasattr(model, 'params'):
                params['coefficients'] = model.params.to_dict() if hasattr(model.params, 'to_dict') else dict(model.params)
            
            if hasattr(model, 'aic'):
                params['aic'] = float(model.aic)
            if hasattr(model, 'bic'):
                params['bic'] = float(model.bic)
            if hasattr(model, 'hqic'):
                params['hqic'] = float(model.hqic)
            if hasattr(model, 'llf'):
                params['log_likelihood'] = float(model.llf)
                
            # SARIMAX specific
            if hasattr(model, 'model'):
                if hasattr(model.model, 'order'):
                    params['order'] = model.model.order
                if hasattr(model.model, 'seasonal_order'):
                    params['seasonal_order'] = model.model.seasonal_order
        except Exception as e:
            logger.debug("Failed to extract some model parameters: %s", e)
        
        return params
    
    def _diebold_mariano_test(self, actuals: pd.Series, forecast1: pd.Series, forecast2: pd.Series) -> Tuple[float, float]:
        """Compute Diebold-Mariano test statistic."""
        # Compute loss differentials (using squared error loss)
        loss1 = (actuals - forecast1) ** 2
        loss2 = (actuals - forecast2) ** 2
        d = loss1 - loss2
        
        # DM test statistic
        d_mean = d.mean()
        d_var = d.var()
        n = len(d)
        
        if d_var == 0:
            return 0.0, 1.0
        
        dm_stat = d_mean / np.sqrt(d_var / n)
        
        # Approximate p-value (two-tailed)
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
        
        return float(dm_stat), float(p_value)
    
    def _compute_aggregated_results(self, result: BacktestResult, has_benchmark: bool) -> None:
        """Compute aggregated metrics and statistical tests."""
        if not result.fold_results:
            return
        
        # Combine all forecasts and actuals
        all_forecasts = []
        all_actuals = []
        
        for fold in result.fold_results:
            all_forecasts.append(fold.forecasts)
            all_actuals.append(fold.actuals)
        
        result.combined_forecasts = pd.concat(all_forecasts)
        result.combined_actuals = pd.concat(all_actuals)
        
        # Aggregate metrics across folds
        aggregated_metrics = {}
        metric_names = set()
        
        # Collect all metric names
        for fold in result.fold_results:
            if fold.metrics:
                metric_names.update(fold.metrics.keys())
        
        # Aggregate each metric
        for metric_name in metric_names:
            metric_values = []
            for fold in result.fold_results:
                if fold.metrics and metric_name in fold.metrics:
                    metric_values.append(fold.metrics[metric_name])
            
            if metric_values:
                aggregated_metrics[metric_name] = {
                    'mean': float(np.mean(metric_values)),
                    'std': float(np.std(metric_values, ddof=1)) if len(metric_values) > 1 else 0.0,
                    'median': float(np.median(metric_values)),
                    'min': float(np.min(metric_values)),
                    'max': float(np.max(metric_values)),
                    'count': len(metric_values)
                }
        
        result.aggregated_metrics = aggregated_metrics
        
        # Combined Diebold-Mariano test
        if has_benchmark and self.config.diebold_mariano:
            dm_stats = [f.diebold_mariano_stat for f in result.fold_results if f.diebold_mariano_stat is not None]
            dm_pvals = [f.diebold_mariano_pvalue for f in result.fold_results if f.diebold_mariano_pvalue is not None]
            
            if dm_stats and dm_pvals:
                # Fisher's method for combining p-values
                from scipy.stats import chi2
                combined_chi2 = -2 * np.sum([np.log(max(p, 1e-16)) for p in dm_pvals])
                combined_pval = 1 - chi2.cdf(combined_chi2, 2 * len(dm_pvals))
                
                result.combined_dm_stat = float(np.mean(dm_stats))
                result.combined_dm_pvalue = float(combined_pval)


def run_rolling_origin_backtest(endog: pd.Series,
                               model_func: Callable,
                               exog: Optional[pd.DataFrame] = None,
                               config: Optional[BacktestConfig] = None,
                               benchmark_func: Optional[Callable] = None,
                               region: Optional[str] = None) -> BacktestResult:
    """Convenience function to run rolling-origin backtesting.
    
    Parameters
    ----------
    endog : pd.Series
        Endogenous time series
    model_func : callable
        Function that fits and returns a model
    exog : pd.DataFrame, optional
        Exogenous variables
    config : BacktestConfig, optional
        Backtesting configuration
    benchmark_func : callable, optional
        Benchmark model function
    region : str, optional
        Region identifier
        
    Returns
    -------
    BacktestResult
        Complete backtesting results
    """
    validator = RollingOriginValidator(config)
    return validator.validate(endog, model_func, exog, benchmark_func, region)