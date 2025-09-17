#!/usr/bin/env python3
"""Test script to validate the configuration management system."""

import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_configuration_system():
    """Test the configuration management system."""
    print("=" * 50)
    print("Testing GDP-ForecasterSARIMAX Configuration System")
    print("=" * 50)
    
    try:
        from config import get_config, get_api_key, ConfigurationError
        print("✓ Configuration modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import configuration modules: {e}")
        return False
    
    try:
        # Test configuration manager initialization
        config = get_config()
        print("✓ Configuration manager initialized")
        
        # Test configuration loading
        summary = config.get_configuration_summary()
        print(f"✓ Loaded configurations: {summary['loaded_configs']}")
        print(f"✓ API providers available: {summary['api_providers']}")
        
        # Test validation
        validation_errors = config.validate_configuration()
        if validation_errors:
            print("⚠ Configuration validation warnings:")
            for section, errors in validation_errors.items():
                for error in errors:
                    print(f"  - {section}: {error}")
        else:
            print("✓ Configuration validation passed")
        
        # Test model configuration access
        search_space = config.get_sarimax_search_space()
        print(f"✓ Model search space: {search_space}")
        
        fixed_params = config.get_fixed_sarimax_params()
        print(f"✓ Fixed parameters: {fixed_params}")
        
        # Test robust errors config
        robust_config = config.get_robust_errors_config()
        print(f"✓ Robust errors config: {robust_config}")
        
        # Test region-specific configuration
        for region in ['US', 'EU27_2020', 'CN']:
            region_config = config.get_region_config(region)
            if region_config:
                print(f"✓ {region} region configuration loaded with {len(region_config)} sections")
            else:
                print(f"⚠ {region} region configuration is empty")
        
        # Test evaluation metrics
        metrics = config.get_evaluation_metrics()
        print(f"✓ Evaluation metrics: {metrics}")
        
        # Test API key resolution
        print("\nAPI Key Resolution:")
        for provider in ['fred', 'eurostat', 'oecd', 'dbnomics']:
            key = get_api_key(provider)
            if key:
                print(f"✓ {provider}: Key available ({key[:8]}...)" if len(key) > 8 else f"✓ {provider}: Key available")
            else:
                print(f"- {provider}: No key (may not be required)")
        
        # Test dot notation access
        print("\nDot Notation Access:")
        test_keys = [
            'model.search_space.p_range',
            'backtesting.rolling_origin.min_train_size', 
            'evaluation.primary_metrics.mae.name',
            'data_sources.gdp_sources.US.series_column'
        ]
        
        for key in test_keys:
            value = config.get(key, 'NOT_FOUND')
            print(f"✓ {key}: {value}")
        
        print("\n" + "=" * 50)
        print("Configuration system test completed successfully!")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"✗ Configuration system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


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
        success = test_configuration_system()

    output = buf.getvalue()
    if append_eval_log:
        try:
            append_eval_log("tests/test_config.py", output)
        except Exception:
            pass

    print(output, end="")
    exit(0 if success else 1)