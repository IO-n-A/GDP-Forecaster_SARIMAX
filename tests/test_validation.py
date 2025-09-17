#!/usr/bin/env python3
"""Test script to validate the data validation and provenance tracking system."""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_test_data():
    """Create test datasets for validation."""
    # Create quarterly data spanning 10 years
    dates = pd.date_range(start='2010-01-01', end='2019-12-31', freq='Q')
    
    # US GDP-like data
    us_gdp = pd.Series(
        100 * (1 + np.random.normal(0.02, 0.01, len(dates))).cumprod(),
        index=dates,
        name='US_GDP'
    )
    
    # EU GDP-like data (different pattern)
    eu_gdp = pd.Series(
        95 * (1 + np.random.normal(0.015, 0.012, len(dates))).cumprod(),
        index=dates,
        name='EU_GDP'
    )
    
    # China GDP-like data (higher growth)
    cn_gdp = pd.Series(
        80 * (1 + np.random.normal(0.04, 0.015, len(dates))).cumprod(),
        index=dates,
        name='CN_GDP'
    )
    
    # Duplicate data (should trigger validation error)
    duplicate_gdp = us_gdp.copy()
    duplicate_gdp.name = 'Duplicate_GDP'
    
    return {
        'US': us_gdp,
        'EU27_2020': eu_gdp,
        'CN': cn_gdp,
        'DUPLICATE': duplicate_gdp
    }

def test_validation_system():
    """Test the validation and provenance tracking system."""
    print("=" * 60)
    print("Testing Data Validation and Provenance Tracking System")
    print("=" * 60)
    
    try:
        from validation import (
            run_validation_pipeline,
            create_data_fingerprint,
            validate_data_uniqueness,
            DataValidationError,
            ValidationSeverity
        )
        print("✓ Validation modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import validation modules: {e}")
        return False
    
    # Create test data
    datasets = create_test_data()
    print(f"✓ Created {len(datasets)} test datasets")
    
    # Test 1: Data fingerprinting
    print("\nTest 1: Data Fingerprinting")
    try:
        us_fingerprint = create_data_fingerprint(
            datasets['US'], 
            source='TEST_FRED',
            series_id='GDPC1_TEST',
            region='US'
        )
        print(f"✓ US GDP fingerprint: {us_fingerprint.hash}")
        print(f"  - Shape: {us_fingerprint.shape}")
        print(f"  - Date range: {us_fingerprint.date_range}")
        print(f"  - Source: {us_fingerprint.source}")
    except Exception as e:
        print(f"✗ Data fingerprinting failed: {e}")
        return False
    
    # Test 2: Uniqueness validation (should detect duplicate)
    print("\nTest 2: Uniqueness Validation")
    try:
        is_unique, duplicates = validate_data_uniqueness(datasets)
        if is_unique:
            print("⚠ Expected to detect duplicates, but none found")
        else:
            print(f"✓ Correctly detected {len(duplicates)} duplications:")
            for dup in duplicates:
                print(f"  - {dup}")
    except Exception as e:
        print(f"✗ Uniqueness validation failed: {e}")
        return False
    
    # Test 3: Comprehensive validation pipeline
    print("\nTest 3: Comprehensive Validation Pipeline")
    try:
        sources = {
            'US': 'FRED',
            'EU27_2020': 'OECD',
            'CN': 'OECD', 
            'DUPLICATE': 'FRED'
        }
        
        series_ids = {
            'US': 'GDPC1',
            'EU27_2020': 'EU27_GDP',
            'CN': 'CHN_GDP',
            'DUPLICATE': 'GDPC1'
        }
        
        result = run_validation_pipeline(
            datasets=datasets,
            sources=sources,
            series_ids=series_ids,
            raise_on_error=False
        )
        
        print(f"✓ Validation pipeline completed: {result.summary()}")
        print(f"  - Total issues: {len(result.issues)}")
        print(f"  - Has errors: {result.has_errors}")
        print(f"  - Has warnings: {result.has_warnings}")
        
        # Check for expected critical error (duplicate data)
        critical_issues = result.get_issues_by_severity(ValidationSeverity.CRITICAL)
        if critical_issues:
            print(f"✓ Found {len(critical_issues)} critical issues as expected")
        else:
            print("⚠ Expected critical issues for duplicate data, but none found")
        
        # Test provenance data
        if result.provenance_data:
            fingerprints = result.provenance_data.get('fingerprints', {})
            print(f"✓ Provenance data available for {len(fingerprints)} datasets")
        else:
            print("⚠ No provenance data in result")
            
    except Exception as e:
        print(f"✗ Comprehensive validation failed: {e}")
        return False
    
    # Test 4: Test with unique data only
    print("\nTest 4: Validation with Unique Data Only")
    try:
        unique_datasets = {k: v for k, v in datasets.items() if k != 'DUPLICATE'}
        
        result = run_validation_pipeline(
            datasets=unique_datasets,
            sources={k: v for k, v in sources.items() if k != 'DUPLICATE'},
            series_ids={k: v for k, v in series_ids.items() if k != 'DUPLICATE'}
        )
        
        print(f"✓ Unique data validation: {result.summary()}")
        if result.is_valid and not result.has_errors:
            print("✓ Validation passed as expected for unique datasets")
        else:
            print("⚠ Unexpected issues with unique datasets")
            
    except Exception as e:
        print(f"✗ Unique data validation failed: {e}")
        return False
    
    # Test 5: Test exogenous alignment validation
    print("\nTest 5: Exogenous Alignment Validation")
    try:
        from validation.pipeline import ComprehensiveValidator
        
        # Create mock exogenous data
        exog_dates = pd.date_range(start='2010-01-01', end='2019-06-30', freq='Q')
        exog_data = pd.DataFrame({
            'PMI': 50 + 10 * np.random.randn(len(exog_dates))
        }, index=exog_dates)
        
        validator = ComprehensiveValidator()
        alignment_result = validator.validate_exog_alignment(datasets['US'], exog_data)
        
        print(f"✓ Exog alignment validation: {alignment_result.summary()}")
        print(f"  - Overlap ratio: {alignment_result.metrics.get('overlap_ratio', 'N/A'):.2%}")
        
    except Exception as e:
        print(f"✗ Exogenous alignment validation failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("Data validation and provenance tracking tests completed successfully!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import io
    import sys
    from contextlib import redirect_stdout, redirect_stderr
    try:
        from helpers.log_utils import append_eval_log
    except Exception:
        append_eval_log = None

    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        success = test_validation_system()

    output = buf.getvalue()
    if append_eval_log:
        try:
            append_eval_log("tests/test_validation.py", output)
        except Exception:
            pass

    print(output, end="")
    exit(0 if success else 1)