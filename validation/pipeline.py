"""Validation pipeline for GDP-ForecasterSARIMAX data integrity.

This module provides high-level validation workflows that orchestrate
the various data integrity checks and provide comprehensive reporting.

Features:
- Comprehensive validation pipeline orchestration
- Structured validation result reporting
- Error handling and recovery strategies
- Integration with configuration system
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd
import numpy as np

from .data_integrity import ValidationPipeline, DataFingerprint, ProvenanceTracker

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    severity: ValidationSeverity
    message: str
    component: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Complete validation result with all issues and metrics."""
    is_valid: bool
    issues: List[ValidationIssue]
    metrics: Dict[str, Any]
    provenance_data: Optional[Dict[str, Any]] = None
    
    @property
    def has_errors(self) -> bool:
        """Check if result has any errors or critical issues."""
        return any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                  for issue in self.issues)
    
    @property
    def has_warnings(self) -> bool:
        """Check if result has any warnings."""
        return any(issue.severity == ValidationSeverity.WARNING for issue in self.issues)
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get all issues of a specific severity."""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def summary(self) -> str:
        """Get a summary string of the validation result."""
        total_issues = len(self.issues)
        errors = len(self.get_issues_by_severity(ValidationSeverity.ERROR))
        criticals = len(self.get_issues_by_severity(ValidationSeverity.CRITICAL))
        warnings = len(self.get_issues_by_severity(ValidationSeverity.WARNING))
        
        status = "PASS" if self.is_valid and not self.has_errors else "FAIL"
        return f"Validation {status}: {total_issues} issues ({criticals} critical, {errors} error, {warnings} warning)"


class DataValidationError(Exception):
    """Exception raised for critical data validation failures."""
    
    def __init__(self, message: str, validation_result: Optional[ValidationResult] = None):
        super().__init__(message)
        self.validation_result = validation_result


class ComprehensiveValidator:
    """Comprehensive data validation orchestrator."""
    
    def __init__(self, config_manager=None):
        """Initialize validator with optional configuration."""
        self.config_manager = config_manager
        self.validation_pipeline = ValidationPipeline()
        self.issues = []
        self.metrics = {}
    
    def validate_datasets(self, datasets: Dict[str, pd.Series], 
                         sources: Optional[Dict[str, str]] = None,
                         series_ids: Optional[Dict[str, str]] = None) -> ValidationResult:
        """Perform comprehensive validation on multiple datasets.
        
        Parameters
        ----------
        datasets : dict
            Dictionary mapping region/dataset names to pandas Series
        sources : dict, optional
            Dictionary mapping dataset names to source identifiers
        series_ids : dict, optional  
            Dictionary mapping dataset names to series identifiers
            
        Returns
        -------
        ValidationResult
            Comprehensive validation result
        """
        logger.info("Starting comprehensive validation of %d datasets", len(datasets))
        self.issues.clear()
        self.metrics.clear()
        
        # Validate basic dataset properties
        self._validate_basic_properties(datasets)
        
        # Validate data uniqueness (critical requirement)
        self._validate_uniqueness(datasets, sources, series_ids)
        
        # Validate data quality
        self._validate_data_quality(datasets)
        
        # Validate temporal properties
        self._validate_temporal_properties(datasets)
        
        # Collect provenance data
        provenance_data = self.validation_pipeline.get_provenance_data()
        
        # Determine overall validity
        is_valid = not any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                          for issue in self.issues)
        
        result = ValidationResult(
            is_valid=is_valid,
            issues=self.issues.copy(),
            metrics=self.metrics.copy(),
            provenance_data=provenance_data
        )
        
        logger.info("Validation completed: %s", result.summary())
        return result
    
    def _validate_basic_properties(self, datasets: Dict[str, pd.Series]) -> None:
        """Validate basic dataset properties."""
        logger.debug("Validating basic dataset properties")
        
        for name, data in datasets.items():
            if data is None:
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Dataset '{name}' is None",
                    component="basic_properties"
                ))
                continue
                
            if data.empty:
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Dataset '{name}' is empty",
                    component="basic_properties"
                ))
                continue
            
            # Check minimum observations
            min_obs = 30  # Default minimum
            if self.config_manager:
                try:
                    min_obs = self.config_manager.get('data_sources.validation.gdp_validation.min_observations', 30)
                except Exception:
                    pass
            
            if len(data) < min_obs:
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Dataset '{name}' has only {len(data)} observations (minimum recommended: {min_obs})",
                    component="basic_properties",
                    details={'observations': len(data), 'minimum': min_obs}
                ))
            
            # Check for datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Dataset '{name}' does not have DatetimeIndex",
                    component="basic_properties"
                ))
        
        self.metrics['dataset_count'] = len([d for d in datasets.values() if d is not None and not d.empty])
    
    def _validate_uniqueness(self, datasets: Dict[str, pd.Series],
                           sources: Optional[Dict[str, str]] = None,
                           series_ids: Optional[Dict[str, str]] = None) -> None:
        """Validate data uniqueness across datasets."""
        logger.debug("Validating data uniqueness")
        
        # Add provenance information if available
        for name, data in datasets.items():
            if data is not None and not data.empty:
                source = sources.get(name) if sources else None
                series_id = series_ids.get(name) if series_ids else None
                self.validation_pipeline.provenance_tracker.add_data(
                    key=name,
                    data=data,
                    source=source,
                    series_id=series_id,
                    region=name
                )
        
        # Validate uniqueness
        is_unique = self.validation_pipeline.validate_data_uniqueness(datasets)
        
        if not is_unique:
            _, duplicates = self.validation_pipeline.provenance_tracker.validate_uniqueness()
            for dup in duplicates:
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Data duplication detected: {dup}",
                    component="uniqueness"
                ))
        
        self.metrics['uniqueness_passed'] = is_unique
        self.metrics['duplicate_count'] = len(duplicates) if not is_unique else 0
    
    def _validate_data_quality(self, datasets: Dict[str, pd.Series]) -> None:
        """Validate data quality metrics."""
        logger.debug("Validating data quality")
        
        quality_metrics = {}
        
        for name, data in datasets.items():
            if data is None or data.empty:
                continue
                
            # Missing data analysis
            missing_count = data.isnull().sum()
            missing_pct = (missing_count / len(data)) * 100
            
            # Get missing data threshold from config
            max_missing_pct = 5  # Default
            if self.config_manager:
                try:
                    max_missing_pct = self.config_manager.get('data_sources.validation.gdp_validation.max_missing_percent', 5)
                except Exception:
                    pass
            
            if missing_pct > max_missing_pct:
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Dataset '{name}' has {missing_pct:.1f}% missing data (threshold: {max_missing_pct}%)",
                    component="data_quality",
                    details={'missing_percent': missing_pct, 'threshold': max_missing_pct}
                ))
            
            # Outlier detection (simple IQR method)
            if len(data.dropna()) > 0:
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                outlier_threshold = 1.5 * IQR
                outliers = ((data < (Q1 - outlier_threshold)) | (data > (Q3 + outlier_threshold))).sum()
                outlier_pct = (outliers / len(data)) * 100
                
                if outlier_pct > 10:  # More than 10% outliers
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Dataset '{name}' has {outlier_pct:.1f}% potential outliers",
                        component="data_quality",
                        details={'outlier_percent': outlier_pct}
                    ))
                
                quality_metrics[f'{name}_missing_pct'] = missing_pct
                quality_metrics[f'{name}_outlier_pct'] = outlier_pct
    
    def _validate_temporal_properties(self, datasets: Dict[str, pd.Series]) -> None:
        """Validate temporal properties of datasets."""
        logger.debug("Validating temporal properties")
        
        temporal_metrics = {}
        
        for name, data in datasets.items():
            if data is None or data.empty or not isinstance(data.index, pd.DatetimeIndex):
                continue
            
            # Check for temporal gaps
            expected_freq = data.index.inferred_freq
            if expected_freq:
                full_range = pd.date_range(start=data.index.min(), 
                                         end=data.index.max(), 
                                         freq=expected_freq)
                missing_dates = len(full_range) - len(data.index.intersection(full_range))
                gap_pct = (missing_dates / len(full_range)) * 100
                
                if gap_pct > 5:  # More than 5% gaps
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Dataset '{name}' has {gap_pct:.1f}% temporal gaps",
                        component="temporal_properties",
                        details={'gap_percent': gap_pct}
                    ))
                
                temporal_metrics[f'{name}_gap_pct'] = gap_pct
            
            # Check data frequency consistency
            if hasattr(data.index, 'freq') and data.index.freq:
                temporal_metrics[f'{name}_frequency'] = str(data.index.freq)
            elif expected_freq:
                temporal_metrics[f'{name}_frequency'] = expected_freq
            else:
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Dataset '{name}' has unclear frequency pattern",
                    component="temporal_properties"
                ))
            
            # Check for reasonable date range
            start_date = data.index.min()
            end_date = data.index.max()
            span_years = (end_date - start_date).days / 365.25
            
            if span_years < 5:
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Dataset '{name}' covers only {span_years:.1f} years - limited for economic modeling",
                    component="temporal_properties",
                    details={'span_years': span_years}
                ))
            
            temporal_metrics[f'{name}_start_date'] = start_date.isoformat()
            temporal_metrics[f'{name}_end_date'] = end_date.isoformat()
            temporal_metrics[f'{name}_span_years'] = span_years
        
        self.metrics.update(temporal_metrics)
    
    def validate_exog_alignment(self, endog: pd.Series, exog: pd.DataFrame) -> ValidationResult:
        """Validate alignment between endogenous and exogenous data.
        
        Parameters
        ----------
        endog : pd.Series
            Endogenous (target) variable
        exog : pd.DataFrame
            Exogenous variables
            
        Returns
        -------
        ValidationResult
            Validation result for alignment
        """
        logger.debug("Validating exogenous variable alignment")
        self.issues.clear()
        self.metrics.clear()
        
        if exog is None or exog.empty:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="No exogenous variables to validate",
                component="exog_alignment"
            ))
            return ValidationResult(is_valid=True, issues=self.issues, metrics=self.metrics)
        
        # Temporal alignment validation
        is_aligned = self.validation_pipeline.validate_temporal_alignment(endog, exog)
        
        if not is_aligned:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Exogenous variables have poor temporal alignment with endogenous variable",
                component="exog_alignment"
            ))
        
        # Calculate alignment metrics
        overlap = endog.index.intersection(exog.index)
        overlap_ratio = len(overlap) / len(endog.index)
        
        self.metrics.update({
            'alignment_passed': is_aligned,
            'overlap_ratio': overlap_ratio,
            'endog_length': len(endog),
            'exog_length': len(exog),
            'overlap_length': len(overlap)
        })
        
        if overlap_ratio < 0.5:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                message=f"Very low overlap between endog and exog: {overlap_ratio:.1%}",
                component="exog_alignment",
                details={'overlap_ratio': overlap_ratio}
            ))
        
        is_valid = not any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                          for issue in self.issues)
        
        return ValidationResult(
            is_valid=is_valid,
            issues=self.issues.copy(),
            metrics=self.metrics.copy()
        )


def run_validation_pipeline(datasets: Dict[str, pd.Series],
                          sources: Optional[Dict[str, str]] = None,
                          series_ids: Optional[Dict[str, str]] = None,
                          config_manager=None,
                          raise_on_error: bool = False) -> ValidationResult:
    """Run comprehensive data validation pipeline.
    
    This is a convenience function that creates a validator and runs
    the full validation suite.
    
    Parameters
    ----------
    datasets : dict
        Dictionary mapping region/dataset names to pandas Series
    sources : dict, optional
        Dictionary mapping dataset names to source identifiers
    series_ids : dict, optional
        Dictionary mapping dataset names to series identifiers
    config_manager : ConfigurationManager, optional
        Configuration manager for validation settings
    raise_on_error : bool, default False
        Whether to raise exception on validation errors
        
    Returns
    -------
    ValidationResult
        Comprehensive validation result
        
    Raises
    ------
    DataValidationError
        If raise_on_error=True and validation fails
    """
    logger.info("Running comprehensive validation pipeline")
    
    validator = ComprehensiveValidator(config_manager)
    result = validator.validate_datasets(datasets, sources, series_ids)
    
    # Log summary
    logger.info("Validation pipeline completed: %s", result.summary())
    
    # Log individual issues
    for issue in result.issues:
        if issue.severity == ValidationSeverity.CRITICAL:
            logger.critical("CRITICAL [%s]: %s", issue.component, issue.message)
        elif issue.severity == ValidationSeverity.ERROR:
            logger.error("ERROR [%s]: %s", issue.component, issue.message)
        elif issue.severity == ValidationSeverity.WARNING:
            logger.warning("WARNING [%s]: %s", issue.component, issue.message)
        else:
            logger.info("INFO [%s]: %s", issue.component, issue.message)
    
    # Optionally raise exception on errors
    if raise_on_error and result.has_errors:
        raise DataValidationError(f"Data validation failed: {result.summary()}", result)
    
    return result


def create_validation_report(result: ValidationResult, output_path: Optional[Path] = None) -> str:
    """Create a detailed validation report.
    
    Parameters
    ----------
    result : ValidationResult
        Validation result to report on
    output_path : Path, optional
        Path to save the report to
        
    Returns
    -------
    str
        Report content as string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("GDP-ForecasterSARIMAX Data Validation Report")
    lines.append("=" * 60)
    lines.append(f"Overall Status: {'PASS' if result.is_valid and not result.has_errors else 'FAIL'}")
    lines.append(f"Total Issues: {len(result.issues)}")
    lines.append("")
    
    # Summary by severity
    for severity in ValidationSeverity:
        issues = result.get_issues_by_severity(severity)
        if issues:
            lines.append(f"{severity.value.upper()}: {len(issues)} issues")
    
    lines.append("")
    lines.append("Validation Metrics:")
    for key, value in result.metrics.items():
        lines.append(f"  {key}: {value}")
    
    lines.append("")
    lines.append("Detailed Issues:")
    lines.append("-" * 40)
    
    for issue in result.issues:
        lines.append(f"[{issue.severity.value.upper()}] {issue.component}: {issue.message}")
        if issue.details:
            for key, value in issue.details.items():
                lines.append(f"    {key}: {value}")
        lines.append("")
    
    if result.provenance_data:
        lines.append("Provenance Information:")
        lines.append("-" * 40)
        fingerprints = result.provenance_data.get('fingerprints', {})
        for key, fp in fingerprints.items():
            lines.append(f"  {key}:")
            lines.append(f"    Hash: {fp.get('hash', 'N/A')}")
            lines.append(f"    Source: {fp.get('source', 'N/A')}")
            lines.append(f"    Date Range: {fp.get('date_range', ['N/A', 'N/A'])}")
            lines.append("")
    
    report_content = '\n'.join(lines)
    
    if output_path:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_content)
            logger.info("Validation report saved to: %s", output_path)
        except Exception as e:
            logger.error("Failed to save validation report: %s", e)
    
    return report_content