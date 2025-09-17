# gdp_forecaster_src/diagnostics_utils.py

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union
import logging

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> None:
    """
    Create directory if it doesn't exist, including all parent directories.
    
    Parameters
    ----------
    path : Path
        Directory path to create
    """
    path.mkdir(parents=True, exist_ok=True)


def save_residual_diagnostics(residuals: Union[pd.Series, np.ndarray], 
                             out_dir: Path, 
                             fname_prefix: str = "Residuals") -> None:
    """
    Save comprehensive residual diagnostics including ACF/PACF, Ljung-Box, ARCH LM, and CUSUM tests.
    
    This function generates a complete set of residual diagnostic plots and test results
    to assess model adequacy and identify potential issues with the fitted model.

    Parameters
    ----------
    residuals : Union[pd.Series, np.ndarray]
        Residual vector from a fitted model
    out_dir : Path
        Output directory where diagnostic artifacts will be written
    fname_prefix : str, default="Residuals"
        Prefix for output filenames to distinguish different models

    Returns
    -------
    None
        
    Notes
    -----
    Creates the following files:
    - {prefix}_ACF_PACF.png: Combined ACF and PACF plots
    - {prefix}_LjungBox.csv: Ljung-Box test results for multiple lags
    - {prefix}_ARCH_LM.csv: ARCH LM test results for heteroskedasticity
    - {prefix}_CUSUM.csv: CUSUM stability test data
    - {prefix}_CUSUM.png: CUSUM stability plot with reference bounds
    """
    ensure_dir(out_dir)
    
    try:
        resid = pd.Series(residuals).dropna()
    except Exception:
        resid = pd.Series(np.asarray(residuals)).dropna()

    if resid.empty:
        logger.warning("Residual diagnostics skipped: empty residual series.")
        return

    # Generate ACF/PACF panel
    _save_acf_pacf_plots(resid, out_dir, fname_prefix)
    
    # Generate Ljung-Box test results
    _save_ljungbox_tests(resid, out_dir, fname_prefix)
    
    # Generate ARCH LM test results
    _save_arch_lm_tests(resid, out_dir, fname_prefix)
    
    # Generate CUSUM stability analysis
    _save_cusum_analysis(resid, out_dir, fname_prefix)


def _save_acf_pacf_plots(resid: pd.Series, out_dir: Path, fname_prefix: str) -> None:
    """
    Create and save combined ACF and PACF plots for residual analysis.
    
    These plots help identify remaining autocorrelation patterns in residuals
    that might indicate model misspecification.
    
    Parameters
    ----------
    resid : pd.Series
        Cleaned residual series
    out_dir : Path
        Output directory
    fname_prefix : str
        Filename prefix
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    try:
        lags = int(min(24, max(10, len(resid) // 4)))
    except Exception:
        lags = 24
        
    try:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), dpi=150)
        plot_acf(resid, ax=axes[0], lags=lags, zero=False)
        axes[0].set_title("Residual ACF")
        plot_pacf(resid, ax=axes[1], lags=lags, zero=False, method="ywmle")
        axes[1].set_title("Residual PACF")
        fig.tight_layout()
        fig.savefig(out_dir / f"{fname_prefix}_ACF_PACF.png", dpi=300)
        plt.close(fig)
        logger.debug("ACF/PACF plots saved successfully")
    except Exception as e:
        logger.debug("Failed to render ACF/PACF diagnostics: %s", e)


def _save_ljungbox_tests(resid: pd.Series, out_dir: Path, fname_prefix: str) -> None:
    """
    Perform and save Ljung-Box portmanteau tests for autocorrelation.
    
    The Ljung-Box test checks for autocorrelation in residuals across multiple lags,
    helping to identify if the model has captured all temporal dependencies.
    
    Parameters
    ----------
    resid : pd.Series
        Cleaned residual series
    out_dir : Path
        Output directory
    fname_prefix : str
        Filename prefix
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    try:
        max_lag = int(min(24, max(1, len(resid) - 1)))
        df_lb = acorr_ljungbox(resid, lags=np.arange(1, max_lag + 1), return_df=True)
        df_lb.to_csv(out_dir / f"{fname_prefix}_LjungBox.csv", index=True)
        logger.debug("Ljung-Box tests saved successfully")
    except Exception as e:
        logger.debug("Ljung-Box diagnostics skipped: %s", e)


def _save_arch_lm_tests(resid: pd.Series, out_dir: Path, fname_prefix: str) -> None:
    """
    Perform and save ARCH LM tests for heteroskedasticity in residuals.
    
    The ARCH LM test checks for autoregressive conditional heteroskedasticity,
    which can indicate time-varying volatility in the residuals.
    
    Parameters
    ----------
    resid : pd.Series
        Cleaned residual series
    out_dir : Path
        Output directory
    fname_prefix : str
        Filename prefix
    """
    from statsmodels.stats.diagnostic import het_arch
    
    try:
        nlags = int(min(12, max(2, len(resid) // 10)))
        lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(resid, nlags=nlags)
        pd.DataFrame({
            "lm_stat": [lm_stat], 
            "lm_pvalue": [lm_pvalue], 
            "f_stat": [f_stat], 
            "f_pvalue": [f_pvalue]
        }).to_csv(out_dir / f"{fname_prefix}_ARCH_LM.csv", index=False)
        logger.debug("ARCH LM tests saved successfully")
    except Exception as e:
        logger.debug("ARCH LM diagnostics skipped: %s", e)


def _save_cusum_analysis(resid: pd.Series, out_dir: Path, fname_prefix: str) -> None:
    """
    Perform and save CUSUM stability analysis for structural breaks.
    
    CUSUM (Cumulative Sum) analysis helps detect structural breaks or instability
    in the model parameters over time.
    
    Parameters
    ----------
    resid : pd.Series
        Cleaned residual series
    out_dir : Path
        Output directory
    fname_prefix : str
        Filename prefix
    """
    try:
        mu = float(np.mean(resid))
        sd = float(np.std(resid, ddof=1))
        
        if not np.isfinite(sd) or sd <= 1e-12:
            raise ValueError("Degenerate residual std for CUSUM.")
        
        # Standardize residuals
        z = (resid - mu) / sd
        cusum = np.cumsum(z)
        t = np.arange(1, len(cusum) + 1)
        
        # Approximate 95% reference bounds for CUSUM (heuristic constant for visualization)
        bound = 1.36 * np.sqrt(t)
        
        # Save CUSUM data
        df_cusum = pd.DataFrame({
            "t": t, 
            "cusum": cusum, 
            "upper_95": bound, 
            "lower_95": -bound
        })
        df_cusum.to_csv(out_dir / f"{fname_prefix}_CUSUM.csv", index=False)
        
        # Create CUSUM plot
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        ax.plot(t, cusum, label="CUSUM", color="tab:purple")
        ax.plot(t, bound, "--", color="gray", linewidth=1, label="±95% bounds")
        ax.plot(t, -bound, "--", color="gray", linewidth=1)
        ax.set_title("CUSUM stability (standardized residuals)")
        ax.set_xlabel("t")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{fname_prefix}_CUSUM.png", dpi=300)
        plt.close(fig)
        
        logger.debug("CUSUM stability analysis saved successfully")
    except Exception as e:
        logger.debug("CUSUM stability diagnostics skipped: %s", e)


def run_comprehensive_diagnostics(residuals: Union[pd.Series, np.ndarray],
                                 model_name: str,
                                 out_dir: Path) -> dict:
    """
    Run a comprehensive set of residual diagnostics and return summary statistics.
    
    This function performs multiple diagnostic tests and returns a summary
    dictionary that can be used for automated model evaluation.
    
    Parameters
    ----------
    residuals : Union[pd.Series, np.ndarray]
        Model residuals to analyze
    model_name : str
        Name of the model for file naming
    out_dir : Path
        Output directory for diagnostic files
        
    Returns
    -------
    dict
        Summary dictionary containing test statistics and p-values
        
    Notes
    -----
    Returns test results that can be used for programmatic model evaluation
    and selection based on diagnostic criteria.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.stats.stattools import jarque_bera
    
    ensure_dir(out_dir)
    
    try:
        resid = pd.Series(residuals).dropna()
    except Exception:
        resid = pd.Series(np.asarray(residuals)).dropna()
    
    if resid.empty:
        logger.warning("Cannot run diagnostics on empty residual series")
        return {}
    
    results = {
        "model": model_name,
        "n_residuals": len(resid),
        "residual_mean": float(np.mean(resid)),
        "residual_std": float(np.std(resid, ddof=1))
    }
    
    # Ljung-Box test for autocorrelation
    try:
        max_lag = min(10, len(resid) // 4)
        lb_result = acorr_ljungbox(resid, lags=max_lag, return_df=False)
        results["ljungbox_stat"] = float(lb_result[0][-1])  # Last lag statistic
        results["ljungbox_pvalue"] = float(lb_result[1][-1])  # Last lag p-value
    except Exception as e:
        logger.debug("Ljung-Box test failed: %s", e)
        results["ljungbox_stat"] = float("nan")
        results["ljungbox_pvalue"] = float("nan")
    
    # ARCH LM test for heteroskedasticity
    try:
        nlags = min(5, len(resid) // 10)
        lm_stat, lm_pvalue, _, _ = het_arch(resid, nlags=nlags)
        results["arch_lm_stat"] = float(lm_stat)
        results["arch_lm_pvalue"] = float(lm_pvalue)
    except Exception as e:
        logger.debug("ARCH LM test failed: %s", e)
        results["arch_lm_stat"] = float("nan")
        results["arch_lm_pvalue"] = float("nan")
    
    # Jarque-Bera test for normality
    try:
        jb_stat, jb_pvalue = jarque_bera(resid)
        results["jarque_bera_stat"] = float(jb_stat)
        results["jarque_bera_pvalue"] = float(jb_pvalue)
    except Exception as e:
        logger.debug("Jarque-Bera test failed: %s", e)
        results["jarque_bera_stat"] = float("nan")
        results["jarque_bera_pvalue"] = float("nan")
    
    # Save diagnostic plots
    save_residual_diagnostics(resid, out_dir, model_name)
    
    return results


def interpret_diagnostic_results(results: dict, alpha: float = 0.05) -> dict:
    """
    Interpret diagnostic test results and provide recommendations.
    
    This function translates statistical test results into actionable insights
    about model adequacy and potential improvements.
    
    Parameters
    ----------
    results : dict
        Results from run_comprehensive_diagnostics
    alpha : float, default=0.05
        Significance level for test interpretation
        
    Returns
    -------
    dict
        Interpretation summary with recommendations
    """
    interpretation = {
        "model": results.get("model", "unknown"),
        "overall_adequacy": "unknown"
    }
    
    issues = []
    
    # Check autocorrelation
    lb_pval = results.get("ljungbox_pvalue", float("nan"))
    if np.isfinite(lb_pval):
        if lb_pval < alpha:
            issues.append("significant autocorrelation in residuals")
            interpretation["autocorrelation_issue"] = True
        else:
            interpretation["autocorrelation_issue"] = False
    
    # Check heteroskedasticity
    arch_pval = results.get("arch_lm_pvalue", float("nan"))
    if np.isfinite(arch_pval):
        if arch_pval < alpha:
            issues.append("significant heteroskedasticity (ARCH effects)")
            interpretation["heteroskedasticity_issue"] = True
        else:
            interpretation["heteroskedasticity_issue"] = False
    
    # Check normality
    jb_pval = results.get("jarque_bera_pvalue", float("nan"))
    if np.isfinite(jb_pval):
        if jb_pval < alpha:
            issues.append("non-normal residuals")
            interpretation["normality_issue"] = True
        else:
            interpretation["normality_issue"] = False
    
    interpretation["issues"] = issues
    interpretation["overall_adequacy"] = "poor" if issues else "good"
    
    # Recommendations
    recommendations = []
    if interpretation.get("autocorrelation_issue"):
        recommendations.append("Consider increasing AR or MA order")
    if interpretation.get("heteroskedasticity_issue"):
        recommendations.append("Consider GARCH-type models for volatility")
    if interpretation.get("normality_issue"):
        recommendations.append("Consider robust estimation or non-normal distributions")
    
    interpretation["recommendations"] = recommendations
    
    return interpretation