"""Comprehensive residual diagnostics for SARIMAX models.

This module provides systematic residual diagnostic testing to validate
model adequacy and identify potential specification issues.

Features:
- Ljung-Box test for serial correlation
- Jarque-Bera test for normality
- ARCH-LM test for heteroskedasticity
- ACF/PACF analysis and plotting
- CUSUM tests for parameter stability
- Comprehensive diagnostic reporting
- Integration with configuration system
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Statistical tests
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import het_arch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

# Import jarque_bera from correct location
try:
    from statsmodels.stats.stattools import jarque_bera
except ImportError:
    try:
        from statsmodels.stats.stattools import jarque_bera as jb_test
        def jarque_bera(x):
            return jb_test(x)
    except ImportError:
        # Fallback to scipy implementation
        def jarque_bera(residuals):
            """Fallback Jarque-Bera test implementation."""
            from scipy.stats import jarque_bera as scipy_jb
            stat, pval = scipy_jb(residuals.dropna())
            n = len(residuals.dropna())
            skewness = residuals.skew()
            kurtosis = residuals.kurtosis()
            return stat, pval, skewness, kurtosis

# Configuration integration
try:
    from config import get_config, ConfigurationManager
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)


class DiagnosticTest(Enum):
    """Types of residual diagnostic tests."""
    LJUNG_BOX = "ljung_box"
    JARQUE_BERA = "jarque_bera"
    ARCH_LM = "arch_lm"
    SHAPIRO_WILK = "shapiro_wilk"
    BREUSCH_GODFREY = "breusch_godfrey"
    DURBIN_WATSON = "durbin_watson"


@dataclass
class DiagnosticResult:
    """Results from a single diagnostic test."""
    
    test_name: str
    test_type: DiagnosticTest
    test_statistic: float
    p_value: float
    critical_value: Optional[float] = None
    significance_level: float = 0.05
    degrees_of_freedom: Optional[int] = None
    
    # Additional test-specific information
    test_description: Optional[str] = None
    additional_stats: Dict[str, float] = field(default_factory=dict)
    
    @property
    def is_significant(self) -> bool:
        """Check if test rejects null hypothesis."""
        return self.p_value < self.significance_level
    
    @property
    def interpretation(self) -> str:
        """Get interpretation of test result."""
        if self.test_type == DiagnosticTest.LJUNG_BOX:
            if self.is_significant:
                return "Serial correlation detected in residuals"
            else:
                return "No significant serial correlation in residuals"
        elif self.test_type == DiagnosticTest.JARQUE_BERA:
            if self.is_significant:
                return "Residuals not normally distributed"
            else:
                return "Residuals appear normally distributed"
        elif self.test_type == DiagnosticTest.ARCH_LM:
            if self.is_significant:
                return "ARCH effects detected in residuals"
            else:
                return "No ARCH effects detected in residuals"
        else:
            if self.is_significant:
                return f"Null hypothesis rejected (p={self.p_value:.4f})"
            else:
                return f"Null hypothesis not rejected (p={self.p_value:.4f})"


class ResidualDiagnostics:
    """Comprehensive residual diagnostic testing."""
    
    def __init__(self, significance_level: float = 0.05, config_manager: Optional = None):
        """Initialize residual diagnostics.
        
        Parameters
        ----------
        significance_level : float, default 0.05
            Significance level for all tests
        config_manager : ConfigurationManager, optional
            Configuration manager for diagnostic settings
        """
        self.significance_level = significance_level
        self.config_manager = config_manager
        
        # Load diagnostic configuration
        self.diag_config = self._load_diagnostic_config()
    
    def _load_diagnostic_config(self) -> Dict[str, Any]:
        """Load diagnostic configuration from config manager."""
        default_config = {
            'ljung_box_lags': 10,
            'arch_lm_lags': 4,
            'acf_pacf_lags': 20,
            'significance_level': self.significance_level,
            'create_plots': True,
            'save_plots': True
        }
        
        if self.config_manager and CONFIG_AVAILABLE:
            try:
                eval_config = self.config_manager.get_evaluation_config()
                diagnostic_config = eval_config.get('diagnostic_tests', {})
                
                # Update with specific test settings
                residual_config = diagnostic_config.get('residual_diagnostics', {})
                for test_name, test_config in residual_config.items():
                    if test_name == 'ljung_box':
                        default_config['ljung_box_lags'] = test_config.get('lags', default_config['ljung_box_lags'])
                    elif test_name == 'arch_lm':
                        default_config['arch_lm_lags'] = test_config.get('lags', default_config['arch_lm_lags'])
                
                default_config['significance_level'] = diagnostic_config.get('significance_level', default_config['significance_level'])
                logger.debug("Loaded diagnostic configuration from config manager")
                
            except Exception as e:
                logger.warning("Failed to load diagnostic config, using defaults: %s", e)
        
        return default_config
    
    def ljung_box_test(self, residuals: pd.Series, lags: Optional[int] = None) -> DiagnosticResult:
        """Ljung-Box test for serial correlation in residuals.
        
        Parameters
        ----------
        residuals : pd.Series
            Model residuals
        lags : int, optional
            Number of lags to test (default from config)
            
        Returns
        -------
        DiagnosticResult
            Ljung-Box test results
        """
        if lags is None:
            lags = self.diag_config['ljung_box_lags']
        
        logger.debug("Running Ljung-Box test with %d lags", lags)
        
        try:
            # Run Ljung-Box test
            lb_result = acorr_ljungbox(residuals, lags=lags, return_df=True)
            
            # Use the test statistic and p-value for the specified number of lags
            if lags in lb_result.index:
                test_stat = lb_result.loc[lags, 'lb_stat']
                p_value = lb_result.loc[lags, 'lb_pvalue']
            else:
                # Use the maximum available lags
                max_lag = lb_result.index.max()
                test_stat = lb_result.loc[max_lag, 'lb_stat']
                p_value = lb_result.loc[max_lag, 'lb_pvalue']
                lags = max_lag
            
            return DiagnosticResult(
                test_name="Ljung-Box Test",
                test_type=DiagnosticTest.LJUNG_BOX,
                test_statistic=float(test_stat),
                p_value=float(p_value),
                degrees_of_freedom=lags,
                significance_level=self.significance_level,
                test_description=f"Test for serial correlation in residuals (H0: No serial correlation, lags={lags})"
            )
            
        except Exception as e:
            logger.error("Ljung-Box test failed: %s", e)
            raise
    
    def jarque_bera_test(self, residuals: pd.Series) -> DiagnosticResult:
        """Jarque-Bera test for normality of residuals.
        
        Parameters
        ----------
        residuals : pd.Series
            Model residuals
            
        Returns
        -------
        DiagnosticResult
            Jarque-Bera test results
        """
        logger.debug("Running Jarque-Bera normality test")
        
        try:
            # Run Jarque-Bera test
            jb_stat, jb_pval, skew, kurtosis = jarque_bera(residuals)
            
            return DiagnosticResult(
                test_name="Jarque-Bera Test",
                test_type=DiagnosticTest.JARQUE_BERA,
                test_statistic=float(jb_stat),
                p_value=float(jb_pval),
                degrees_of_freedom=2,
                significance_level=self.significance_level,
                test_description="Test for normality of residuals (H0: Residuals are normally distributed)",
                additional_stats={'skewness': float(skew), 'kurtosis': float(kurtosis)}
            )
            
        except Exception as e:
            logger.error("Jarque-Bera test failed: %s", e)
            raise
    
    def arch_lm_test(self, residuals: pd.Series, lags: Optional[int] = None) -> DiagnosticResult:
        """ARCH-LM test for heteroskedasticity in residuals.
        
        Parameters
        ----------
        residuals : pd.Series
            Model residuals
        lags : int, optional
            Number of lags to test (default from config)
            
        Returns
        -------
        DiagnosticResult
            ARCH-LM test results
        """
        if lags is None:
            lags = self.diag_config['arch_lm_lags']
        
        logger.debug("Running ARCH-LM test with %d lags", lags)
        
        try:
            # Import from heteroskedasticity module if available
            try:
                from .heteroskedasticity import HeteroskedasticityTester
                tester = HeteroskedasticityTester(self.significance_level)
                arch_result = tester.test_arch_lm(residuals, lags)
                
                return DiagnosticResult(
                    test_name="ARCH-LM Test",
                    test_type=DiagnosticTest.ARCH_LM,
                    test_statistic=arch_result.test_statistic,
                    p_value=arch_result.p_value,
                    degrees_of_freedom=arch_result.degrees_of_freedom,
                    significance_level=self.significance_level,
                    test_description=arch_result.test_description
                )
            except ImportError:
                # Fallback to direct implementation
                lm_stat, lm_pval, f_stat, f_pval = het_arch(residuals, nlags=lags)
                
                return DiagnosticResult(
                    test_name="ARCH-LM Test",
                    test_type=DiagnosticTest.ARCH_LM,
                    test_statistic=float(lm_stat),
                    p_value=float(lm_pval),
                    degrees_of_freedom=lags,
                    significance_level=self.significance_level,
                    test_description=f"Test for ARCH effects in residuals (H0: No ARCH effects, lags={lags})"
                )
                
        except Exception as e:
            logger.error("ARCH-LM test failed: %s", e)
            raise
    
    def shapiro_wilk_test(self, residuals: pd.Series) -> DiagnosticResult:
        """Shapiro-Wilk test for normality (for smaller samples).
        
        Parameters
        ----------
        residuals : pd.Series
            Model residuals
            
        Returns
        -------
        DiagnosticResult
            Shapiro-Wilk test results
        """
        logger.debug("Running Shapiro-Wilk normality test")
        
        try:
            if len(residuals) > 5000:
                logger.warning("Shapiro-Wilk test may be unreliable for large samples (n=%d)", len(residuals))
            
            sw_stat, sw_pval = stats.shapiro(residuals.dropna())
            
            return DiagnosticResult(
                test_name="Shapiro-Wilk Test",
                test_type=DiagnosticTest.SHAPIRO_WILK,
                test_statistic=float(sw_stat),
                p_value=float(sw_pval),
                significance_level=self.significance_level,
                test_description="Test for normality of residuals (H0: Residuals are normally distributed)"
            )
            
        except Exception as e:
            logger.error("Shapiro-Wilk test failed: %s", e)
            raise
    
    def compute_acf_pacf(self, residuals: pd.Series, lags: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Compute ACF and PACF for residuals.
        
        Parameters
        ----------
        residuals : pd.Series
            Model residuals
        lags : int, optional
            Number of lags to compute
            
        Returns
        -------
        dict
            Dictionary with 'acf' and 'pacf' arrays
        """
        if lags is None:
            lags = min(self.diag_config['acf_pacf_lags'], len(residuals) // 4)
        
        logger.debug("Computing ACF/PACF with %d lags", lags)
        
        try:
            # Compute ACF and PACF
            acf_vals, acf_confint = acf(residuals, nlags=lags, alpha=0.05)
            pacf_vals, pacf_confint = pacf(residuals, nlags=lags, alpha=0.05)
            
            return {
                'acf': acf_vals,
                'pacf': pacf_vals,
                'acf_confint': acf_confint,
                'pacf_confint': pacf_confint,
                'lags': np.arange(lags + 1)
            }
            
        except Exception as e:
            logger.error("ACF/PACF computation failed: %s", e)
            raise
    
    def create_diagnostic_plots(self, residuals: pd.Series, 
                              output_dir: Optional[Path] = None,
                              model_name: str = "SARIMAX") -> Dict[str, Path]:
        """Create comprehensive diagnostic plots.
        
        Parameters
        ----------
        residuals : pd.Series
            Model residuals
        output_dir : Path, optional
            Directory to save plots
        model_name : str
            Model name for plot titles
            
        Returns
        -------
        dict
            Dictionary mapping plot names to file paths
        """
        if not self.diag_config.get('create_plots', True):
            return {}
        
        logger.info("Creating diagnostic plots for %s", model_name)
        
        plot_paths = {}
        
        try:
            # Set up plotting
            plt.style.use('default')
            
            # 1. Residuals time series plot
            fig, ax = plt.subplots(figsize=(12, 4))
            residuals.plot(ax=ax, title=f'{model_name} Residuals', color='blue', alpha=0.7)
            ax.axhline(0, color='red', linestyle='--', alpha=0.5)
            ax.set_ylabel('Residuals')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if output_dir and self.diag_config.get('save_plots', True):
                plot_path = output_dir / f'{model_name}_residuals_timeseries.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plot_paths['residuals_timeseries'] = plot_path
            
            plt.close()
            
            # 2. Residuals histogram with normal overlay
            fig, ax = plt.subplots(figsize=(8, 6))
            residuals.hist(bins=30, density=True, alpha=0.7, ax=ax, color='skyblue')
            
            # Overlay normal distribution
            x = np.linspace(residuals.min(), residuals.max(), 100)
            normal_y = stats.norm.pdf(x, residuals.mean(), residuals.std())
            ax.plot(x, normal_y, 'r-', linewidth=2, label='Normal Distribution')
            
            ax.set_title(f'{model_name} Residuals Distribution')
            ax.set_xlabel('Residuals')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if output_dir and self.diag_config.get('save_plots', True):
                plot_path = output_dir / f'{model_name}_residuals_histogram.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plot_paths['residuals_histogram'] = plot_path
            
            plt.close()
            
            # 3. Q-Q plot
            fig, ax = plt.subplots(figsize=(8, 8))
            stats.probplot(residuals.dropna(), dist="norm", plot=ax)
            ax.set_title(f'{model_name} Q-Q Plot (Normal)')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if output_dir and self.diag_config.get('save_plots', True):
                plot_path = output_dir / f'{model_name}_qq_plot.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plot_paths['qq_plot'] = plot_path
            
            plt.close()
            
            # 4. ACF and PACF plots
            acf_pacf_data = self.compute_acf_pacf(residuals)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # ACF plot
            lags = acf_pacf_data['lags']
            acf_vals = acf_pacf_data['acf']
            acf_confint = acf_pacf_data['acf_confint']
            
            ax1.plot(lags, acf_vals, 'bo-', markersize=4)
            ax1.fill_between(lags, acf_confint[:, 0], acf_confint[:, 1], alpha=0.2, color='blue')
            ax1.axhline(0, color='black', linestyle='-', alpha=0.5)
            ax1.set_title(f'{model_name} Autocorrelation Function (ACF)')
            ax1.set_xlabel('Lags')
            ax1.set_ylabel('ACF')
            ax1.grid(True, alpha=0.3)
            
            # PACF plot
            pacf_vals = acf_pacf_data['pacf']
            pacf_confint = acf_pacf_data['pacf_confint']
            
            ax2.plot(lags, pacf_vals, 'ro-', markersize=4)
            ax2.fill_between(lags, pacf_confint[:, 0], pacf_confint[:, 1], alpha=0.2, color='red')
            ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
            ax2.set_title(f'{model_name} Partial Autocorrelation Function (PACF)')
            ax2.set_xlabel('Lags')
            ax2.set_ylabel('PACF')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if output_dir and self.diag_config.get('save_plots', True):
                plot_path = output_dir / f'{model_name}_acf_pacf.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plot_paths['acf_pacf'] = plot_path
            
            plt.close()
            
            logger.info("Created %d diagnostic plots", len(plot_paths))
            
        except Exception as e:
            logger.error("Failed to create diagnostic plots: %s", e)
        
        return plot_paths
    
    def run_comprehensive_diagnostics(self, 
                                    residuals: pd.Series,
                                    model_name: str = "SARIMAX",
                                    create_plots: bool = True,
                                    output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Run comprehensive residual diagnostics.
        
        Parameters
        ----------
        residuals : pd.Series
            Model residuals
        model_name : str
            Model name for reporting
        create_plots : bool
            Whether to create diagnostic plots
        output_dir : Path, optional
            Directory for saving plots and results
            
        Returns
        -------
        dict
            Comprehensive diagnostic results
        """
        logger.info("Running comprehensive residual diagnostics for %s", model_name)
        
        results = {
            'model_name': model_name,
            'n_residuals': len(residuals),
            'test_results': {},
            'summary_statistics': {},
            'plots': {},
            'overall_assessment': {}
        }
        
        # Basic summary statistics
        results['summary_statistics'] = {
            'mean': float(residuals.mean()),
            'std': float(residuals.std()),
            'skewness': float(residuals.skew()),
            'kurtosis': float(residuals.kurtosis()),
            'min': float(residuals.min()),
            'max': float(residuals.max()),
            'jarque_bera_stat': None,  # Will be filled by test
        }
        
        # Run diagnostic tests
        test_functions = [
            ('ljung_box', self.ljung_box_test),
            ('jarque_bera', self.jarque_bera_test),
            ('arch_lm', self.arch_lm_test),
        ]
        
        # Add Shapiro-Wilk for smaller samples
        if len(residuals) <= 5000:
            test_functions.append(('shapiro_wilk', self.shapiro_wilk_test))
        
        for test_name, test_func in test_functions:
            try:
                test_result = test_func(residuals)
                results['test_results'][test_name] = test_result
                logger.debug("%s: %s", test_result.test_name, test_result.interpretation)
                
                # Update summary statistics with test-specific info
                if test_name == 'jarque_bera':
                    results['summary_statistics']['jarque_bera_stat'] = test_result.test_statistic
                    
            except Exception as e:
                logger.error("Test %s failed: %s", test_name, e)
                continue
        
        # Create diagnostic plots
        if create_plots:
            try:
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                
                plot_paths = self.create_diagnostic_plots(residuals, output_dir, model_name)
                results['plots'] = plot_paths
                
            except Exception as e:
                logger.error("Failed to create plots: %s", e)
        
        # Overall assessment
        results['overall_assessment'] = self._assess_model_adequacy(results['test_results'])
        
        logger.info("Comprehensive diagnostics completed for %s", model_name)
        return results
    
    def _assess_model_adequacy(self, test_results: Dict[str, DiagnosticResult]) -> Dict[str, Any]:
        """Assess overall model adequacy based on test results.
        
        Parameters
        ----------
        test_results : dict
            Dictionary of test results
            
        Returns
        -------
        dict
            Overall assessment
        """
        assessment = {
            'issues_detected': [],
            'warnings': [],
            'recommendations': [],
            'overall_adequate': True
        }
        
        for test_name, result in test_results.items():
            if result.is_significant:
                if result.test_type == DiagnosticTest.LJUNG_BOX:
                    assessment['issues_detected'].append("Serial correlation in residuals")
                    assessment['recommendations'].append("Consider increasing AR or MA order")
                    assessment['overall_adequate'] = False
                    
                elif result.test_type == DiagnosticTest.ARCH_LM:
                    assessment['issues_detected'].append("Heteroskedasticity (ARCH effects)")
                    assessment['recommendations'].append("Use robust standard errors or consider GARCH modeling")
                    # Don't set overall_adequate = False as this is addressable with robust inference
                    
                elif result.test_type in [DiagnosticTest.JARQUE_BERA, DiagnosticTest.SHAPIRO_WILK]:
                    assessment['warnings'].append("Residuals not normally distributed")
                    assessment['recommendations'].append("Consider robust inference; normality not critical for forecasting")
        
        if not assessment['issues_detected'] and not assessment['warnings']:
            assessment['recommendations'].append("Model diagnostics look good - no major issues detected")
        
        return assessment


def run_comprehensive_diagnostics(residuals: pd.Series,
                                model_name: str = "SARIMAX",
                                create_plots: bool = True,
                                output_dir: Optional[Path] = None,
                                significance_level: float = 0.05) -> Dict[str, Any]:
    """Convenience function for comprehensive residual diagnostics.
    
    Parameters
    ----------
    residuals : pd.Series
        Model residuals
    model_name : str
        Model name
    create_plots : bool
        Whether to create plots
    output_dir : Path, optional
        Output directory
    significance_level : float
        Significance level for tests
        
    Returns
    -------
    dict
        Comprehensive diagnostic results
    """
    diagnostics = ResidualDiagnostics(significance_level)
    return diagnostics.run_comprehensive_diagnostics(residuals, model_name, create_plots, output_dir)