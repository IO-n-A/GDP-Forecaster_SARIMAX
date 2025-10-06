# GDP‑ForecasterSARIMAX

## Abstract

A comprehensive, production-ready SARIMAX forecasting toolkit for quarterly GDP analysis with robust statistical methodology, configuration management, and extensive evaluation capabilities. The system provides:

- **Robust Modeling**: SARIMAX with heteroskedasticity testing, residual diagnostics, and robust standard errors
- **Configuration-Driven**: YAML-based configuration management for reproducible workflows  
- **Multi-Modal Data Fetching**: Consolidated fetchers for GDP, PMI, ESI, Industrial Production, and OECD CLI
- **Advanced Evaluation**: Enhanced metrics with stability filtering, rolling-origin cross-validation, and Diebold-Mariano testing
- **Data Validation**: SHA-256 fingerprinting and provenance tracking for data integrity
- **Production Features**: Comprehensive logging, archival protocols, and automated backtesting pipelines
- **Open Data**: Uses only public data sources (OECD, FRED, Eurostat, NBS) with no proprietary dependencies

[forecaster_SARIMAX.py](forecaster_SARIMAX.py) has been refactored into a modular package located in `gdp_forecaster_src/` for improved maintainability. The original script is now a lightweight wrapper for backward compatibility. An interactive demonstration is available in `forecaster_SARIMAX.ipynb`.

## Contents
- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview) 
- [File Structure](#file-structure)
- [Setup and Configuration](#setup-and-configuration)
- [Usage Workflows](#usage-workflows)
- [Enhanced Features](#enhanced-features)
- [Data Sources and Attribution](#data-sources-and-attribution)
- [Evaluation Methodology](#evaluation-methodology)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Key Features

### Robust Statistical Methodology
- **SARIMAX Modeling**: Grid-search over (p,q,P,Q) ∈ {0..3} with fixed d=1, D=0, s=4 (quarterly)
- **Heteroskedasticity Testing**: ARCH-LM and Breusch-Pagan tests for variance instability
- **Residual Diagnostics**: Systematic Ljung-Box, Jarque-Bera, and stability testing
- **Enhanced Model Selection**: AIC-based selection with robust diagnostic validation

### Configuration Management System
- **YAML Configuration**: Central configuration via `config/` directory with validation
- **API Key Management**: Secure key resolution from environment variables or config files
- **Model Parameters**: Configurable SARIMAX search spaces and convergence settings
- **Data Source Configuration**: Region-specific source mappings and validation rules

### Data Integration and Validation
- **Multi-Source GDP Fetching**: OECD QNA, FRED integration with configuration-driven selection
- **Exogenous Data Sources**: ISM PMI, Industrial Production, ESI, OECD CLI with regional optimization
- **Data Validation**: SHA-256 fingerprinting, uniqueness assertions, and provenance tracking
- **Quality Assurance**: Automated data integrity checks and temporal alignment validation

### Production-Ready Infrastructure
- **Comprehensive Logging**: Structured logging protocol with archiving and audit trails
- **Backtesting Framework**: Automated multi-fold cross-validation with statistical aggregation
- **Diagnostic Reports**: Systematic residual analysis with artifact generation
- **Batch Processing**: Multi-region workflows with consolidated reporting

## Architecture Overview

The system has been refactored into a modern, modular architecture centered around the **`gdp_forecaster_src`** package. This is the recommended structure for development and usage, as it promotes separation of concerns, reusability, and testability.

The legacy script-based architecture (e.g., `fetchers/`, `evaluation/`) is retained for backward compatibility but is considered deprecated.

```
├── gdp_forecaster_src/           # Main source code as a Python package
│   ├── main.py                   # CLI entry point and orchestration
│   ├── data_utils.py             # Data loading and preprocessing
│   ├── forecasting_utils.py      # Core SARIMAX modeling logic
│   └── ... (other specialized modules)
│
├── forecaster_SARIMAX.py         # Backward-compatible CLI wrapper
├── forecaster_SARIMAX.ipynb      # Interactive Jupyter Notebook for analysis
├── data/                         # (Not version controlled) Input datasets
└── tests/                        # Unit and integration tests
```

## File Structure

```
GDP-ForecasterSARIMAX/
├── README.md                     # Comprehensive documentation
├── LICENSE                       # MIT license
├── pyproject.toml                # Project configuration and tooling
├── requirements.txt              # Runtime dependencies
├── forecaster_SARIMAX.py         # Main CLI wrapper (backward compatible)
├── forecaster_SARIMAX.ipynb      # 🆕 Interactive analysis notebook
│
├── gdp_forecaster_src/           # 🆕 Refactored application package
│   ├── __init__.py
│   ├── main.py                   # Main CLI entry point
│   ├── config_utils.py
│   ├── data_utils.py
│   ├── diagnostics_utils.py
│   ├── file_utils.py
│   ├── forecasting_utils.py
│   ├── metrics_utils.py
│   ├── parsing_utils.py
│   ├── plotting_utils.py
│   └── transform_utils.py
│
├── fetchers/                     # (Legacy) Data fetching scripts
├── diagnostics/                  # (Legacy) Statistical testing scripts  
├── backtesting/                  # (Legacy) Backtesting framework
├── evaluation/                   # (Legacy) Enhanced evaluation metrics
├── validation/                   # (Legacy) Data validation & provenance
├── helpers/                      # (Legacy) Utilities and support
│
├── data/                         # (gitignored) Holds input data files
├── figures/                      # (gitignored) Holds generated plots
├── docs/                         # Additional documentation
│   └── ARCHIVE/                  # Archived legacy scripts
└── tests/                        # Comprehensive test suite
```

## Setup and Configuration

### Requirements
- Python 3.10+
- FRED API key for US Industrial Production/Manufacturers' Orders

### Installation

```bash
# Clone repository
git clone https://github.com/IO-n-A/GDP-Forecaster_SARIMAX.git
cd GDP-ForecasterSARIMAX

# Optional virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install runtime dependencies
pip install -r requirements.txt

# Install development tools (optional)
pip install -r requirements-dev.txt
pre-commit install
```

### Configuration Setup

The system uses YAML configuration files for reproducible workflows:

```bash
# Configuration files are in config/ directory
ls config/
# config_manager.py          # Central configuration management
# key_manager.py             # API key resolution  
# model_parameters.yaml      # SARIMAX parameters
# data_sources.yaml          # Data source mappings
# evaluation_metrics.yaml    # Metrics settings
# backtesting_config.yaml    # Cross-validation settings
```

### API Key Setup (Optional)

For FRED data access, configure API keys:

```bash
# Option 1: Environment variable (recommended)
export FRED_API_KEY="your_fred_api_key_here"

# Option 2: Configuration file
cp config/master_keys.yaml.example config/master_keys.yaml
# Edit config/master_keys.yaml with your API keys
```

## Usage Workflows

All commands shown from repository root. The system uses comprehensive logging and headless plotting (Matplotlib Agg backend).

### Basic GDP Forecasting

```bash
# Single region with enhanced evaluation
python forecaster_SARIMAX.py \
  --series-csv data/gdp_US.csv \
  --figures-dir figures/US \
  --metrics-csv metrics.csv \
  --log-level INFO
```

### Multi-Region Batch Processing with Enhanced Features

```bash
# Fetch GDP data using configuration-driven system
python fetchers/enhanced_gdp_fetcher.py --regions US EU27_2020 CN --validate

# Comprehensive batch evaluation with robust metrics
python forecaster_SARIMAX.py \
  --batch-run \
  --batch-countries US,EU27_2020,CN \
  --figures-dir figures \
  --metrics-csv metrics.csv \
  --log-level INFO
```

### Enhanced Exogenous Data Integration

With the new exogenous data sources:

```bash
# US with Industrial Production (long history alternative to ISM PMI)
python forecaster_SARIMAX.py \
  --series-csv data/gdp_US.csv \
  --use-exog \
  --exog-source-US industrial_production \
  --us-ip-type manufacturing \
  --figures-dir figures/US_IP \
  --metrics-csv metrics.csv

# EU with Economic Sentiment Indicator  
python forecaster_SARIMAX.py \
  --series-csv data/gdp_EU27_2020.csv \
  --use-exog \
  --exog-source-EU esi \
  --eu-geo EU27_2020 \
  --figures-dir figures/EU_ESI \
  --metrics-csv metrics.csv

# Cross-country with OECD CLI harmonized indicators
python forecaster_SARIMAX.py \
  --batch-run \
  --batch-countries US,EU27_2020,CN \
  --use-exog \
  --exog-source-US oecd_cli \
  --exog-source-EU oecd_cli \
  --exog-source-CN oecd_cli \
  --figures-dir figures/CLI \
  --metrics-csv metrics.csv
```

### Advanced Statistical Analysis

```bash
# Multi-fold cross-validation with robust standard errors
python forecaster_SARIMAX.py \
  --series-csv data/gdp_US.csv \
  --multi-fold \
  --folds 5 \
  --fold-horizon 8 \
  --dm-method fisher \
  --intervals "80,95" \
  --figures-dir figures/US_robust \
  --metrics-csv metrics.csv
```

### Configuration-Driven Workflows

```bash
# Using YAML configuration for reproducible analysis
python forecaster_SARIMAX.py \
  --series-csv data/gdp_US.csv \
  --config-driven \
  --figures-dir figures/US_config \
  --metrics-csv metrics.csv
```

## Enhanced Features

### 🆕 Robust Statistical Testing
- **Heteroskedasticity Detection**: ARCH-LM and Breusch-Pagan tests identify variance instability
- **Robust Standard Errors**: HAC/Newey-West covariance correction for reliable inference
- **Residual Diagnostics**: Systematic Ljung-Box, Jarque-Bera, and stability testing
- **Enhanced Model Selection**: AIC-based selection with robust diagnostic validation

### 🆕 Advanced Evaluation Metrics
- **Stable Metrics Priority**: MAE, RMSE, MASE, and Theil's U as primary evaluation criteria
- **MAPE/sMAPE Deprecation**: Known instability issues addressed with appropriate warnings
- **Robust Secondary Metrics**: Median AE and Directional Accuracy for robustness
- **Performance Classification**: Excellent/Good/Acceptable/Poor threshold-based assessment

### 🆕 Configuration Management
- **Central Configuration**: YAML-based system with validation and error handling
- **API Key Security**: Environment variable precedence with configuration file fallback  
- **Model Parameters**: Configurable SARIMAX search spaces and convergence settings
- **Data Source Mapping**: Region-specific source configuration with validation

### 🆕 Enhanced Data Sources

**US Economic Indicators:**
- ISM PMI via DB.NOMICS (existing, short history)
- **Industrial Production** via FRED (new, long history)
- **Manufacturers' New Orders** via FRED (new, long history)
- **OECD CLI** for harmonized cross-country analysis (new)

**EU Economic Indicators:**
- Economic Sentiment Indicator via Eurostat (existing)
- Industry Confidence via Eurostat (existing) 
- **OECD CLI** for harmonized analysis (new)

**China Economic Indicators:**
- NBS PMI via official sources/DB.NOMICS (existing)
- **OECD CLI** for harmonized analysis (new)

### 🆕 Data Validation and Provenance
- **SHA-256 Fingerprinting**: Ensures data uniqueness and prevents duplication issues
- **Temporal Alignment**: Validates proper time series alignment and frequency consistency
- **Provenance Tracking**: Records data sources, vintage, and transformation history
- **Quality Assurance**: Automated validation with comprehensive error reporting

### 🆕 Production Infrastructure
- **Structured Logging**: Comprehensive logging protocol with archival and audit trails
- **Backtesting Framework**: Multi-fold cross-validation with statistical aggregation
- **Automated Workflows**: Configuration-driven batch processing with error recovery
- **Testing Suite**: Comprehensive unit tests for all components and features

## Data Sources and Attribution

This project exclusively uses open, publicly available data sources:

### GDP Data Sources
- **Primary**: OECD Quarterly National Accounts (QNA) via SDMX-JSON API (Public Domain)
- **Alternative**: FRED (Federal Reserve Economic Data) with API key (Public Domain)
- **Coverage**: US (GDPC1), EU (EU27_2020/EA19 aggregates), China (CHN), 75+ countries

### Exogenous Economic Indicators

**United States:**
- ISM Manufacturing PMI via DB.NOMICS mirrors (2020→ coverage)
- **Industrial Production Index (INDPRO)** via FRED (long history, robust)
- **Manufacturers' New Orders (AMTMNO)** via FRED (comprehensive coverage)

**European Union:**
- Economic Sentiment Indicator (ESI) via Eurostat SDMX API (CC BY 4.0)
- Industry Confidence Indicator via Eurostat SDMX API (CC BY 4.0)
- Coverage: EA19, EU27_2020 aggregates with monthly frequency

**China:**
- National Bureau of Statistics (NBS) Manufacturing PMI (official, public)
- Available via NBS official releases or DB.NOMICS mirrors

**Cross-Country Harmonized:**
- **OECD Composite Leading Indicators (CLI)** for all major economies
- Amplitude-adjusted, normalized, and trend-restored variants
- Designed for cyclical turning point detection with consistent methodology

### Statistical and Diagnostic Tools
- **Heteroskedasticity Tests**: statsmodels implementation (ARCH-LM, Breusch-Pagan)
- **Robust Standard Errors**: HAC/Newey-West correction via statsmodels
- **Residual Diagnostics**: Ljung-Box, Jarque-Bera, ACF/PACF via statsmodels
- **Cross-Validation**: Custom implementation following academic best practices

**Data Attribution and Compliance:**
All data sources are properly attributed with source citations, vintage tracking, and license compliance. No proprietary or paywalled data sources are used in default workflows or examples.

## Evaluation Methodology

### Enhanced Metrics Framework

The evaluation system prioritizes stable, scale-independent metrics while appropriately handling unstable measures:

**Primary Stable Metrics:**
- **MAE (Mean Absolute Error)**: Scale-dependent but interpretable
- **RMSE (Root Mean Square Error)**: Penalizes large errors, widely used
- **MASE (Mean Absolute Scaled Error)**: Scale-independent, benchmarked against naive forecast
- **Theil's U1**: Scale-independent relative error measure

**Robust Secondary Metrics:**
- **Median AE**: Robust to outliers, less sensitive to extreme errors
- **Directional Accuracy**: Measures forecast direction correctness
- **Theil's U2**: Relative accuracy vs. naive random walk

**Deprecated Metrics (with warnings):**
- **MAPE/sMAPE**: Deprecated due to known instability with series near zero
- Generates appropriate warnings when computed but excluded from headline results
- Maintained for legacy compatibility with clear deprecation notices

### Statistical Testing
- **Diebold-Mariano Test**: Statistical significance of forecast improvements
- **Fisher/Stouffer Combination**: Multi-fold p-value aggregation methods
- **Confidence Intervals**: Bootstrap-based uncertainty quantification

### Backtesting Framework
- **Rolling-Origin Cross-Validation**: Expanding window with multiple train/test splits
- **Fold-Based Evaluation**: Configurable number of folds and forecast horizons
- **Performance Assessment**: Threshold-based classification (Excellent/Good/Acceptable/Poor)

## Troubleshooting

### Common Issues and Solutions

**Configuration Issues:**
```bash
# Verify configuration system
python -c "from config import get_config; print('Config system working')"

# Check API key resolution
python -c "from config import get_api_key; print(get_api_key('fred') or 'No FRED key')"
```

**Data Fetching Issues:**
```bash
# Test enhanced GDP fetcher
python -c "from fetchers import get_gdp_fetcher; f=get_gdp_fetcher(); print('GDP fetcher ready')"

# Verify data sources
python -c "from fetchers import fetch_us_indicators, fetch_eu_indicators; print('Indicators ready')"
```

**Evaluation System Issues:**
```bash
# Test enhanced metrics
python -c "from evaluation import EnhancedMetricsCalculator; print('Enhanced metrics ready')"
```

**Import Errors:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- For development features: `pip install -r requirements-dev.txt`
- Check Python version: Requires Python 3.10+

**Data Quality Issues:**
- The system includes comprehensive validation with SHA-256 fingerprinting
- Check logs for data integrity warnings and validation errors
- Use `--log-level DEBUG` for detailed diagnostics

**Performance Issues:**
- Use `--aic-cache` for repeated runs with same model search space
- Configure smaller search ranges with `--p-range`, `--q-range` options
- Enable parallel processing where available

### Getting Help
- Check the comprehensive logging output for diagnostic information
- Review configuration files in `config/` directory for settings
- Examine test files in `tests/` directory for usage examples
- All diagnostic plots and metrics are saved for post-analysis

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgements

**Data Providers:**
- OECD: Quarterly National Accounts and Composite Leading Indicators
- Federal Reserve Economic Data (FRED): US economic time series
- Eurostat: European economic statistics and business confidence surveys
- National Bureau of Statistics of China: Official PMI data
- DB.NOMICS: Data aggregation and API services

**Statistical Methods:**
- Based on established econometric practices and academic literature
- Implements robust statistical testing following best practices
- Uses statsmodels for core statistical computations

**Open Source Philosophy:**
This project is committed to open science and reproducible research using only publicly available data sources and transparent methodologies.