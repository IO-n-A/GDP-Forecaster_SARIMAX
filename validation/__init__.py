"""Data validation and provenance tracking for GDP-ForecasterSARIMAX.

This package provides comprehensive data validation and provenance tracking including:
- Data fingerprinting with SHA-256 hashing
- Provenance tracking (source, vintage, series_id)
- Validation pipelines to prevent data duplication
- Enhanced metrics CSV schema support
"""

from .data_integrity import (
    DataFingerprint,
    ValidationPipeline,
    ProvenanceTracker,
    validate_data_uniqueness,
    create_data_fingerprint,
    enhance_metrics_with_provenance
)

from .pipeline import (
    ValidationSeverity,
    ValidationIssue,
    ValidationResult,
    DataValidationError,
    ComprehensiveValidator,
    run_validation_pipeline,
    create_validation_report
)

__all__ = [
    # Data integrity
    'DataFingerprint',
    'ValidationPipeline', 
    'ProvenanceTracker',
    'validate_data_uniqueness',
    'create_data_fingerprint',
    'enhance_metrics_with_provenance',
    
    # Pipeline
    'ValidationSeverity',
    'ValidationIssue',
    'ValidationResult',
    'DataValidationError',
    'ComprehensiveValidator',
    'run_validation_pipeline',
    'create_validation_report'
]

# Version info
__version__ = '1.0.0'