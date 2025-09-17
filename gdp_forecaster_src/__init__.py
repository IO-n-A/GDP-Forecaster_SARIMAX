# gdp_forecaster_src/__init__.py

"""
GDP Forecaster SARIMAX - Modular Time Series Forecasting Package

This package provides a modular implementation of SARIMAX-based GDP forecasting
capabilities, refactored from a monolithic script into focused utility modules.

Key Components
--------------
- config_utils: Configuration management and CLI override support
- data_utils: Data loading, processing, and validation
- parsing_utils: Command-line argument and configuration parsing
- transform_utils: Time series transformations (level, qoq, yoy, log_diff)
- metrics_utils: Forecast evaluation metrics and statistical tests
- plotting_utils: Visualization and charting capabilities
- diagnostics_utils: Residual analysis and model diagnostics
- forecasting_utils: SARIMAX modeling and prediction functions
- file_utils: File operations, CSV handling, and path utilities
- main: Main entry point and workflow orchestration

Usage
-----
The package can be used as a command-line tool or imported for programmatic use:

    # Command-line usage
    python -m gdp_forecaster_src.main --series-csv data/gdp_US.csv

    # Programmatic usage
    from gdp_forecaster_src import forecasting_utils, metrics_utils
"""

__version__ = "1.0.0"
__author__ = "GDP Forecaster Development Team"

# Import key functions for easy access
from .config_utils import initialize_config, get_config_value
from .data_utils import load_macro_data, load_gdp_series_csv
from .forecasting_utils import optimize_sarimax, rolling_one_step_predictions
from .metrics_utils import compute_metrics
from .plotting_utils import generate_summary_figures
from .main import main

__all__ = [
    # Core functionality
    "main",
    "initialize_config", 
    "get_config_value",
    "load_macro_data",
    "load_gdp_series_csv", 
    "optimize_sarimax",
    "rolling_one_step_predictions",
    "compute_metrics",
    "generate_summary_figures",
    # Version info
    "__version__",
    "__author__"
]