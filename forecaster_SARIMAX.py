#!/usr/bin/env python3
"""
SARIMAX modeling and evaluation on US macroeconomic quarterly data (1959–2009).

Usage
-----
This file maintains the same CLI interface as before:
    python main.py --help
    python main.py --default-run
    python main.py --series-csv data/gdp_US.csv

New Modular Structure
--------------------
The code is organized in gdp_forecaster_src/ with these modules:
- config_utils.py: Configuration management
- data_utils.py: Data loading and processing
- parsing_utils.py: CLI argument parsing
- transform_utils.py: Time series transformations
- metrics_utils.py: Evaluation metrics
- plotting_utils.py: Visualization functions
- diagnostics_utils.py: Residual diagnostics
- forecasting_utils.py: SARIMAX modeling
- file_utils.py: File operations
- main.py: Main entry point

Migration Note
--------------
The original 2000+ line monolithic file has been archived in docs/ARCHIVE/
All functionality remains the same, but the code is now more maintainable.
"""

if __name__ == "__main__":
    # Import and delegate to the modular implementation
    try:
        from gdp_forecaster_src.main import main
        main()
    except ImportError as e:
        print(f"Error: Cannot import the modules: {e}")
        print("Please ensure the gdp_forecaster_src/ directory is present and contains the modular code.")
        exit(1)
