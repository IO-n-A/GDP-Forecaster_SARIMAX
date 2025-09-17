"""Heteroskedasticity testing and robust inference for SARIMAX models.

This module provides comprehensive tools for detecting and addressing
heteroskedasticity in time series models, particularly SARIMAX models
used for GDP forecasting.

Features:
- ARCH-LM test for heteroskedasticity in residuals
- Breusch-Pagan test for heteroskedasticity
- HAC/Newey-West robust standard error computation
- GARCH model integration for heteroskedastic residuals
- Configuration-driven robust inference settings
- Integration with SARIMAX model fitting
- Diagnostic plotting and reporting
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import het_arch, het_breuschpagan
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.sandwich_covariance import cov_hac

# Configuration integration
try:
    from config import get_config, ConfigurationManager
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)


class HeteroskedasticityTest(Enum):
    """Types of heteroskedasticity tests."""
    ARCH_LM = "arch_lm"
    BREUSCH_PAGAN = "breusch_pagan"
    WHITE = "white"
    GOLDFELD_QUANDT = "goldfeld_quandt"


class RobustCovarianceType(Enum):
    """Types of robust covariance estimators."""
    HAC = "hac"                    # Heteroskedasticity and Autocorrelation Consistent
    NEWEY_WEST = "newey-west"      # Newey-West HAC estimator
    WHITE = "white"                # White heteroskedasticity-robust
    ROBUST = "robust"              # Generic robust (typically HAC)


@dataclass
class HeteroskedasticityResult:
    """Results from heteroskedasticity testing."""
    
    test_name: str
    test_statistic: float
    p_value: float
    critical_value: Optional[float] = None
    significance_level: float = 0.05
    
    # Additional test information
    degrees_of_freedom: Optional[int] = None
    auxiliary_regression_r2: Optional[float] = None
    test_description: Optional[str] = None
    
    @property
    def is_heteroskedastic(self) -> bool:
        """Check if heteroskedasticity is detected."""
        return self.p_value < self.significance_level
    
    @property 
    def interpretation(self) -> str:
        """Get interpretation of test result."""
        if self.is_heteroskedastic:
            return f"Heteroskedasticity detected (p={self.p_value:.4f} < {self.significance_level})"
        else:
            return f"No heteroskedasticity detected (p={self.p_value:.4f} >= {self.significance_level})"


class HeteroskedasticityTester:
    """Comprehensive heteroskedasticity testing for time series models."""
    
    def __init__(self, significance_level: float = 0.05):
        """Initialize the heteroskedasticity tester.
        
        Parameters
        ----------
        significance_level : float, default 0.05
            Significance level for hypothesis tests
        """
        self.significance_level = significance_level
    
    def test_arch_lm(self, residuals: pd.Series, lags: int = 4) -> HeteroskedasticityResult:
        """Test for ARCH effects using Lagrange Multiplier test.
        
        Parameters
        ----------
        residuals : pd.Series
            Model residuals
        lags : int, default 4
            Number of lags to include in the test
            
        Returns
        -------
        HeteroskedasticityResult
            ARCH-LM test results
        """
        logger.debug("Running ARCH-LM test with %d lags", lags)
        
        try:
            # Run ARCH-LM test
            lm_stat, lm_pval, f_stat, f_pval = het_arch(residuals, nlags=lags)
            
            return HeteroskedasticityResult(
                test_name="ARCH-LM Test",
                test_statistic=float(lm_stat),
                p_value=float(lm_pval),
                degrees_of_freedom=lags,
                significance_level=self.significance_level,
                test_description=f"Test for ARCH effects in residuals (H0: No ARCH effects, lags={lags})"
            )
            
        except Exception as e:
            logger.error("ARCH-LM test failed: %s", e)
            raise
    
    def test_breusch_pagan(self, residuals: pd.Series, 
                          regressors: Optional[pd.DataFrame] = None) -> HeteroskedasticityResult:
        """Test for heteroskedasticity using Breusch-Pagan test.
        
        Parameters
        ----------
        residuals : pd.Series
            Model residuals
        regressors : pd.DataFrame, optional
            Regressors for heteroskedasticity test. If None, uses fitted values.
            
        Returns
        -------
        HeteroskedasticityResult
            Breusch-Pagan test results
        """
        logger.debug("Running Breusch-Pagan test")
        
        try:
            if regressors is None:
                # Use a simple trend as regressor if none provided
                n = len(residuals)
                regressors = pd.DataFrame({
                    'const': np.ones(n),
                    'trend': np.arange(1, n + 1)
                }, index=residuals.index)
            
            # Ensure regressors include constant
            if 'const' not in regressors.columns:
                regressors = regressors.copy()
                regressors.insert(0, 'const', 1.0)
            
            # Run Breusch-Pagan test
            bp_stat, bp_pval, _, _ = het_breuschpagan(residuals, regressors)
            
            return HeteroskedasticityResult(
                test_name="Breusch-Pagan Test",
                test_statistic=float(bp_stat),
                p_value=float(bp_pval),
                degrees_of_freedom=regressors.shape[1] - 1,  # Exclude constant
                significance_level=self.significance_level,
                test_description="Test for heteroskedasticity (H0: Homoskedasticity)"
            )
            
        except Exception as e:
            logger.error("Breusch-Pagan test failed: %s", e)
            raise
    
    def test_white(self, residuals: pd.Series, fitted_values: pd.Series) -> HeteroskedasticityResult:
        """Test for heteroskedasticity using White's test.
        
        Parameters
        ----------
        residuals : pd.Series
            Model residuals
        fitted_values : pd.Series
            Fitted values from the model
            
        Returns
        -------
        HeteroskedasticityResult
            White test results
        """
        logger.debug("Running White's heteroskedasticity test")
        
        try:
            # White test: regress squared residuals on fitted values and their squares
            y_hat = fitted_values
            y_hat2 = y_hat ** 2
            
            # Create regressors
            regressors = pd.DataFrame({
                'const': 1.0,
                'fitted': y_hat,
                'fitted_sq': y_hat2
            }, index=residuals.index)
            
            # Auxiliary regression: residuals^2 on regressors
            squared_residuals = residuals ** 2
            aux_model = OLS(squared_residuals, regressors).fit()
            
            # White test statistic: n * R^2
            n = len(residuals)
            r_squared = aux_model.rsquared
            white_stat = n * r_squared
            
            # Chi-square test
            df = regressors.shape[1] - 1  # Exclude constant
            p_value = 1 - stats.chi2.cdf(white_stat, df)
            
            return HeteroskedasticityResult(
                test_name="White's Test",
                test_statistic=float(white_stat),
                p_value=float(p_value),
                degrees_of_freedom=df,
                auxiliary_regression_r2=r_squared,
                significance_level=self.significance_level,
                test_description="White's test for heteroskedasticity (H0: Homoskedasticity)"
            )
            
        except Exception as e:
            logger.error("White's test failed: %s", e)
            raise
    
    def comprehensive_heteroskedasticity_test(self, 
                                            residuals: pd.Series,
                                            fitted_values: Optional[pd.Series] = None,
                                            arch_lags: int = 4) -> Dict[str, HeteroskedasticityResult]:
        """Run comprehensive heteroskedasticity testing.
        
        Parameters
        ----------
        residuals : pd.Series
            Model residuals
        fitted_values : pd.Series, optional
            Model fitted values for White's test
        arch_lags : int, default 4
            Number of lags for ARCH-LM test
            
        Returns
        -------
        dict
            Dictionary of test results
        """
        logger.info("Running comprehensive heteroskedasticity testing")
        
        results = {}
        
        # ARCH-LM test
        try:
            results['arch_lm'] = self.test_arch_lm(residuals, arch_lags)
            logger.debug("ARCH-LM test completed: %s", results['arch_lm'].interpretation)
        except Exception as e:
            logger.error("ARCH-LM test failed: %s", e)
        
        # Breusch-Pagan test
        try:
            results['breusch_pagan'] = self.test_breusch_pagan(residuals)
            logger.debug("Breusch-Pagan test completed: %s", results['breusch_pagan'].interpretation)
        except Exception as e:
            logger.error("Breusch-Pagan test failed: %s", e)
        
        # White's test (if fitted values available)
        if fitted_values is not None:
            try:
                results['white'] = self.test_white(residuals, fitted_values)
                logger.debug("White's test completed: %s", results['white'].interpretation)
            except Exception as e:
                logger.error("White's test failed: %s", e)
        
        # Summary
        hetero_detected = any(result.is_heteroskedastic for result in results.values())
        logger.info("Heteroskedasticity testing summary: %s detected across %d tests",
                   "Heteroskedasticity" if hetero_detected else "No heteroskedasticity",
                   len(results))
        
        return results


class RobustInferenceManager:
    """Manages robust inference for time series models with heteroskedasticity."""
    
    def __init__(self, config_manager: Optional = None):
        """Initialize the robust inference manager.
        
        Parameters
        ----------
        config_manager : ConfigurationManager, optional
            Configuration manager for robust inference settings
        """
        self.config_manager = config_manager
        
        # Load configuration
        self.robust_config = self._load_robust_config()
    
    def _load_robust_config(self) -> Dict[str, Any]:
        """Load robust inference configuration."""
        default_config = {
            'enabled': True,
            'cov_type': 'robust',
            'cov_kwds': {},
            'default_maxlags': 4,
            'kernel': 'bartlett',
            'use_correction': True
        }
        
        if self.config_manager and CONFIG_AVAILABLE:
            try:
                config = self.config_manager.get('model.robust_errors', {})
                default_config.update(config)
                logger.debug("Loaded robust inference configuration from config manager")
            except Exception as e:
                logger.warning("Failed to load robust config, using defaults: %s", e)
        
        return default_config
    
    def fit_sarimax_with_robust_errors(self, 
                                     endog: pd.Series,
                                     exog: Optional[pd.DataFrame] = None,
                                     order: Tuple[int, int, int] = (1, 1, 1),
                                     seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 4),
                                     **fit_kwargs) -> Any:
        """Fit SARIMAX model with robust standard errors.
        
        Parameters
        ----------
        endog : pd.Series
            Endogenous time series
        exog : pd.DataFrame, optional
            Exogenous variables
        order : tuple
            SARIMAX order (p, d, q)
        seasonal_order : tuple
            Seasonal order (P, D, Q, s)
        **fit_kwargs
            Additional fitting arguments
            
        Returns
        -------
        SARIMAX results object with robust standard errors
        """
        logger.info("Fitting SARIMAX model with robust standard errors")
        
        # Create model
        model = SARIMAX(
            endog,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
            simple_differencing=False
        )
        
        # Set up robust fitting parameters
        robust_fit_kwargs = fit_kwargs.copy()
        
        if self.robust_config['enabled']:
            robust_fit_kwargs['cov_type'] = self.robust_config['cov_type']
            
            # Add covariance keywords
            cov_kwds = self.robust_config['cov_kwds'].copy()
            if 'maxlags' not in cov_kwds and self.robust_config['cov_type'] in ['hac', 'newey-west']:
                cov_kwds['maxlags'] = self.robust_config['default_maxlags']
            if 'kernel' not in cov_kwds and self.robust_config['cov_type'] in ['hac', 'newey-west']:
                cov_kwds['kernel'] = self.robust_config['kernel']
            if 'use_correction' not in cov_kwds:
                cov_kwds['use_correction'] = self.robust_config['use_correction']
            
            if cov_kwds:
                robust_fit_kwargs['cov_kwds'] = cov_kwds
            
            logger.debug("Using robust covariance: %s with options %s",
                        self.robust_config['cov_type'], cov_kwds)
        
        # Fit model
        try:
            fitted_model = model.fit(disp=False, **robust_fit_kwargs)
            
            # Verify robust standard errors were applied
            if hasattr(fitted_model, 'cov_type'):
                logger.info("Model fitted with %s covariance", fitted_model.cov_type)
            else:
                logger.warning("Could not verify robust covariance application")
            
            return fitted_model
            
        except Exception as e:
            logger.error("Robust SARIMAX fitting failed: %s", e)
            
            # Fallback to non-robust fitting
            logger.warning("Falling back to non-robust standard errors")
            fallback_kwargs = {k: v for k, v in fit_kwargs.items() 
                             if k not in ['cov_type', 'cov_kwds']}
            return model.fit(disp=False, **fallback_kwargs)
    
    def compute_hac_covariance(self, model_results, maxlags: Optional[int] = None) -> np.ndarray:
        """Compute HAC (Heteroskedasticity and Autocorrelation Consistent) covariance matrix.
        
        Parameters
        ----------
        model_results : statsmodels results object
            Fitted model results
        maxlags : int, optional
            Maximum lags for HAC estimation
            
        Returns
        -------
        np.ndarray
            HAC covariance matrix
        """
        if maxlags is None:
            maxlags = self.robust_config['default_maxlags']
        
        try:
            # Compute HAC covariance matrix
            hac_cov = cov_hac(model_results, nlags=maxlags, kernel=self.robust_config['kernel'])
            logger.debug("HAC covariance matrix computed with %d lags", maxlags)
            return hac_cov
            
        except Exception as e:
            logger.error("HAC covariance computation failed: %s", e)
            return model_results.cov_params()
    
    def assess_heteroskedasticity_and_recommend(self, 
                                              model_results,
                                              arch_lags: int = 4) -> Dict[str, Any]:
        """Assess heteroskedasticity and recommend appropriate inference approach.
        
        Parameters
        ----------
        model_results : statsmodels results object
            Fitted model results
        arch_lags : int, default 4
            Number of lags for ARCH-LM test
            
        Returns
        -------
        dict
            Assessment results and recommendations
        """
        logger.info("Assessing heteroskedasticity and generating recommendations")
        
        # Extract residuals and fitted values
        residuals = model_results.resid
        fitted_values = model_results.fittedvalues
        
        # Run heteroskedasticity tests
        tester = HeteroskedasticityTester()
        hetero_results = tester.comprehensive_heteroskedasticity_test(
            residuals, fitted_values, arch_lags
        )
        
        # Determine if heteroskedasticity is present
        hetero_detected = any(result.is_heteroskedastic for result in hetero_results.values())
        
        # Generate recommendations
        recommendations = []
        if hetero_detected:
            recommendations.extend([
                "Heteroskedasticity detected - use robust standard errors",
                "Consider HAC/Newey-West standard errors for inference",
                "Evaluate GARCH modeling for residual variance"
            ])
            
            # Check if current model uses robust errors
            if hasattr(model_results, 'cov_type') and model_results.cov_type != 'nonrobust':
                recommendations.append(f"Current model uses {model_results.cov_type} covariance - appropriate choice")
            else:
                recommendations.append("Current model uses non-robust standard errors - consider refitting with robust option")
        else:
            recommendations.append("No significant heteroskedasticity detected - standard inference is appropriate")
        
        return {
            'heteroskedasticity_detected': hetero_detected,
            'test_results': hetero_results,
            'recommendations': recommendations,
            'suggested_robust_options': {
                'cov_type': 'hac' if hetero_detected else 'nonrobust',
                'maxlags': arch_lags,
                'kernel': 'bartlett'
            }
        }


def test_heteroskedasticity(residuals: pd.Series,
                          fitted_values: Optional[pd.Series] = None,
                          arch_lags: int = 4,
                          significance_level: float = 0.05) -> Dict[str, HeteroskedasticityResult]:
    """Convenience function for heteroskedasticity testing.
    
    Parameters
    ----------
    residuals : pd.Series
        Model residuals
    fitted_values : pd.Series, optional
        Model fitted values
    arch_lags : int, default 4
        Number of lags for ARCH-LM test
    significance_level : float, default 0.05
        Significance level for tests
        
    Returns
    -------
    dict
        Dictionary of heteroskedasticity test results
    """
    tester = HeteroskedasticityTester(significance_level)
    return tester.comprehensive_heteroskedasticity_test(residuals, fitted_values, arch_lags)


def apply_robust_inference(endog: pd.Series,
                          exog: Optional[pd.DataFrame] = None,
                          order: Tuple[int, int, int] = (1, 1, 1),
                          seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 4),
                          config_manager: Optional = None) -> Tuple[Any, Dict[str, Any]]:
    """Apply robust inference to SARIMAX model fitting.
    
    Parameters
    ----------
    endog : pd.Series
        Endogenous time series
    exog : pd.DataFrame, optional
        Exogenous variables
    order : tuple
        SARIMAX order
    seasonal_order : tuple
        Seasonal SARIMAX order
    config_manager : optional
        Configuration manager
        
    Returns
    -------
    tuple
        (fitted_model, assessment_results)
    """
    manager = RobustInferenceManager(config_manager)
    
    # Fit model with robust errors
    fitted_model = manager.fit_sarimax_with_robust_errors(
        endog, exog, order, seasonal_order
    )
    
    # Assess heteroskedasticity
    assessment = manager.assess_heteroskedasticity_and_recommend(fitted_model)
    
    return fitted_model, assessment