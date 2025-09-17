"""Enhanced GDP data fetcher with configuration integration and validation.

This module provides an enhanced GDP fetching system that:
- Uses the configuration system to determine data sources
- Ensures distinct data sources for US (FRED), EU (OECD/Eurostat), China (OECD)
- Integrates with data validation and provenance tracking
- Prevents data duplication issues through validation
- Provides comprehensive error handling and fallback mechanisms

Data Sources by Region:
- US: FRED Real GDP (Series GDPC1) - Bureau of Economic Analysis
- EU27_2020: OECD Quarterly National Accounts - Eurostat aggregation
- CN: OECD Quarterly National Accounts - National Bureau of Statistics

Features:
- Configuration-driven data source selection
- Data validation and uniqueness checking
- Provenance tracking with source attribution
- Robust error handling with fallback mechanisms
- Integration with API key management
"""

import logging
from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Configuration and validation imports
try:
    from config import get_config, get_api_key, ConfigurationError
    from validation import (
        run_validation_pipeline, 
        create_data_fingerprint,
        enhance_metrics_with_provenance,
        ValidationResult
    )
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).warning("Enhanced features not available: %s", e)
    ENHANCED_FEATURES_AVAILABLE = False

# Existing fetcher imports
from .gdp_fetchers import fred_observations, FredSeries, RECOMMENDED_FRED_SERIES
from .oecd_fetchers import fetch_oecd_qna

logger = logging.getLogger(__name__)


class EnhancedGDPFetcher:
    """Enhanced GDP fetcher with configuration and validation integration."""
    
    def __init__(self):
        """Initialize the enhanced GDP fetcher."""
        self.config_manager = None
        self.validation_results = {}
        self.provenance_data = {}
        
        # Initialize configuration if available
        if ENHANCED_FEATURES_AVAILABLE:
            try:
                from config import get_config
                self.config_manager = get_config()
                logger.info("Configuration system initialized for GDP fetching")
            except Exception as e:
                logger.warning("Failed to initialize configuration: %s", e)
    
    def fetch_all_regions(self, regions: Optional[List[str]] = None, 
                         validate_data: bool = True) -> Tuple[Dict[str, pd.Series], ValidationResult]:
        """Fetch GDP data for all specified regions with validation.
        
        Parameters
        ----------
        regions : list, optional
            List of region codes to fetch. Defaults to ['US', 'EU27_2020', 'CN']
        validate_data : bool, default True
            Whether to run data validation and uniqueness checks
            
        Returns
        -------
        tuple
            (datasets_dict, validation_result)
        """
        if regions is None:
            regions = ['US', 'EU27_2020', 'CN']
        
        logger.info("Fetching GDP data for regions: %s", regions)
        datasets = {}
        sources = {}
        series_ids = {}
        
        # Fetch data for each region
        for region in regions:
            try:
                data, source_info = self.fetch_region_gdp(region)
                if data is not None and not data.empty:
                    datasets[region] = data
                    sources[region] = source_info['source']
                    series_ids[region] = source_info['series_id']
                    logger.info("Successfully fetched %d observations for %s from %s", 
                              len(data), region, source_info['source'])
                else:
                    logger.error("No data retrieved for region %s", region)
            except Exception as e:
                logger.error("Failed to fetch data for region %s: %s", region, e)
                continue
        
        # Validate data if requested
        validation_result = None
        if validate_data and ENHANCED_FEATURES_AVAILABLE and datasets:
            try:
                validation_result = run_validation_pipeline(
                    datasets=datasets,
                    sources=sources,
                    series_ids=series_ids,
                    config_manager=self.config_manager,
                    raise_on_error=False
                )
                
                if validation_result.has_errors:
                    logger.error("Data validation failed: %s", validation_result.summary())
                    # Log specific issues
                    for issue in validation_result.issues:
                        if issue.severity.value in ['error', 'critical']:
                            logger.error("Validation issue: %s", issue.message)
                else:
                    logger.info("Data validation passed: %s", validation_result.summary())
                
                # Store provenance data
                self.provenance_data = validation_result.provenance_data or {}
                
            except Exception as e:
                logger.error("Data validation failed with exception: %s", e)
        
        return datasets, validation_result
    
    def fetch_region_gdp(self, region: str) -> Tuple[pd.Series, Dict[str, str]]:
        """Fetch GDP data for a specific region using configured data source.
        
        Parameters
        ----------
        region : str
            Region code (US, EU27_2020, CN)
            
        Returns
        -------
        tuple
            (data_series, source_info_dict)
        """
        logger.debug("Fetching GDP data for region: %s", region)
        
        # Get region configuration
        region_config = self._get_region_config(region)
        gdp_config = region_config.get('gdp_source', {})
        
        # Determine data source and parameters
        source_type = gdp_config.get('source_type', 'api')
        
        if source_type == 'csv':
            # Load from CSV file
            return self._fetch_from_csv(region, gdp_config)
        else:
            # Fetch from API based on region
            if region == 'US':
                return self._fetch_us_gdp(gdp_config)
            elif region == 'EU27_2020':
                return self._fetch_eu_gdp(gdp_config)
            elif region == 'CN':
                return self._fetch_china_gdp(gdp_config)
            else:
                raise ValueError(f"Unsupported region: {region}")
    
    def _get_region_config(self, region: str) -> Dict[str, Any]:
        """Get configuration for a specific region."""
        if self.config_manager:
            try:
                return self.config_manager.get_region_config(region)
            except Exception as e:
                logger.warning("Failed to get region config for %s: %s", region, e)
        
        # Fallback configuration
        fallback_configs = {
            'US': {
                'gdp_source': {
                    'source_type': 'api',
                    'provider': 'FRED',
                    'series_id': 'GDPC1'
                }
            },
            'EU27_2020': {
                'gdp_source': {
                    'source_type': 'api', 
                    'provider': 'OECD',
                    'location_code': 'EU27_2020'
                }
            },
            'CN': {
                'gdp_source': {
                    'source_type': 'api',
                    'provider': 'OECD',
                    'location_code': 'CHN'
                }
            }
        }
        
        return fallback_configs.get(region, {})
    
    def _fetch_us_gdp(self, config: Dict[str, Any]) -> Tuple[pd.Series, Dict[str, str]]:
        """Fetch US GDP data from FRED."""
        series_id = config.get('series_id', 'GDPC1')
        
        # Get API key
        api_key = None
        if ENHANCED_FEATURES_AVAILABLE:
            api_key = get_api_key('fred')
        
        if not api_key:
            # Try environment variable or config fallback
            import os
            api_key = os.environ.get('FRED_API_KEY')
            if not api_key and self.config_manager:
                try:
                    api_key = self.config_manager.get_api_key('fred')
                except:
                    pass
        
        if not api_key:
            raise ValueError("FRED API key not available. Set FRED_API_KEY environment variable.")
        
        logger.info("Fetching US GDP from FRED (series: %s)", series_id)
        
        # Fetch data using existing FRED fetcher
        df = fred_observations(series_id, api_key=api_key)
        
        if df.empty:
            raise ValueError(f"No data returned from FRED for series {series_id}")
        
        # Convert to series with proper naming
        data_series = pd.Series(
            df['value'].values,
            index=pd.to_datetime(df['date']),
            name='gdp'
        )
        
        # Remove any missing values
        data_series = data_series.dropna()
        
        source_info = {
            'source': 'FRED',
            'series_id': series_id,
            'provider': 'Federal Reserve Bank of St. Louis',
            'attribution': 'U.S. Bureau of Economic Analysis'
        }
        
        logger.info("Retrieved %d US GDP observations from FRED", len(data_series))
        return data_series, source_info
    
    def _fetch_eu_gdp(self, config: Dict[str, Any]) -> Tuple[pd.Series, Dict[str, str]]:
        """Fetch EU GDP data from OECD."""
        location_code = config.get('location_code', 'EU27_2020')
        
        logger.info("Fetching EU GDP from OECD (location: %s)", location_code)
        
        # Fetch data using existing OECD fetcher
        try:
            df = fetch_oecd_qna(location=location_code)
        except Exception as e:
            # Try fallback to EA19 if EU27_2020 fails
            if location_code == 'EU27_2020':
                logger.warning("EU27_2020 failed, trying EA19 fallback: %s", e)
                df = fetch_oecd_qna(location='EA19')
                location_code = 'EA19'
            else:
                raise
        
        if df.empty:
            raise ValueError(f"No data returned from OECD for location {location_code}")
        
        # Convert to series
        data_series = pd.Series(
            df['gdp'].values,
            index=df['date'],
            name='gdp'
        )
        
        # Remove any missing values
        data_series = data_series.dropna()
        
        source_info = {
            'source': 'OECD',
            'series_id': f'QNA.{location_code}.B1_GA.CLV_I10.Q',
            'provider': 'Organisation for Economic Co-operation and Development',
            'attribution': 'Eurostat' if location_code.startswith('EU') else 'Eurostat/ECB'
        }
        
        logger.info("Retrieved %d EU GDP observations from OECD", len(data_series))
        return data_series, source_info
    
    def _fetch_china_gdp(self, config: Dict[str, Any]) -> Tuple[pd.Series, Dict[str, str]]:
        """Fetch China GDP data from OECD."""
        location_code = config.get('location_code', 'CHN')
        
        logger.info("Fetching China GDP from OECD (location: %s)", location_code)
        
        # Fetch data using existing OECD fetcher
        df = fetch_oecd_qna(location=location_code)
        
        if df.empty:
            raise ValueError(f"No data returned from OECD for location {location_code}")
        
        # Convert to series
        data_series = pd.Series(
            df['gdp'].values,
            index=df['date'],
            name='gdp'
        )
        
        # Remove any missing values
        data_series = data_series.dropna()
        
        source_info = {
            'source': 'OECD',
            'series_id': f'QNA.{location_code}.B1_GA.CLV_I10.Q',
            'provider': 'Organisation for Economic Co-operation and Development',
            'attribution': 'National Bureau of Statistics of China'
        }
        
        logger.info("Retrieved %d China GDP observations from OECD", len(data_series))
        return data_series, source_info
    
    def _fetch_from_csv(self, region: str, config: Dict[str, Any]) -> Tuple[pd.Series, Dict[str, str]]:
        """Fetch GDP data from CSV file."""
        file_path = Path(config.get('file_path', ''))
        
        if not file_path.exists():
            raise FileNotFoundError(f"GDP data file not found: {file_path}")
        
        logger.info("Loading %s GDP data from CSV: %s", region, file_path)
        
        # Load CSV
        df = pd.read_csv(file_path)
        
        # Get column mappings from config
        series_column = config.get('series_column', 'gdp')
        date_columns = config.get('date_columns', ['date'])
        
        # Handle date parsing
        if len(date_columns) == 1:
            # Single date column
            df['date'] = pd.to_datetime(df[date_columns[0]])
        else:
            # Multiple date columns (e.g., year, quarter)
            if 'year' in date_columns and 'quarter' in date_columns:
                df['date'] = pd.to_datetime(df['year'].astype(str) + 'Q' + df['quarter'].astype(str))
            else:
                raise ValueError(f"Unsupported date column configuration: {date_columns}")
        
        # Create series
        data_series = pd.Series(
            df[series_column].values,
            index=df['date'],
            name='gdp'
        )
        
        # Remove any missing values
        data_series = data_series.dropna()
        
        source_info = {
            'source': 'CSV',
            'series_id': f'{region}_{series_column}',
            'provider': 'Local File',
            'attribution': config.get('description', f'{region} GDP data from CSV')
        }
        
        logger.info("Loaded %d %s GDP observations from CSV", len(data_series), region)
        return data_series, source_info
    
    def save_datasets(self, datasets: Dict[str, pd.Series], 
                     output_dir: Path, 
                     include_provenance: bool = True) -> Dict[str, Path]:
        """Save datasets to CSV files with optional provenance information.
        
        Parameters
        ----------
        datasets : dict
            Dictionary mapping region names to GDP series
        output_dir : Path
            Output directory for CSV files
        include_provenance : bool, default True
            Whether to include provenance information in CSV headers
            
        Returns
        -------
        dict
            Dictionary mapping region names to output file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        for region, data in datasets.items():
            output_file = output_dir / f'gdp_{region}.csv'
            
            # Create DataFrame
            df = pd.DataFrame({
                'date': data.index,
                'gdp': data.values
            })
            
            # Add provenance information as comments if requested
            if include_provenance and region in self.provenance_data.get('fingerprints', {}):
                fingerprint = self.provenance_data['fingerprints'][region]
                provenance_header = f"""# GDP Data for {region}
# Source: {fingerprint.get('source', 'Unknown')}
# Series ID: {fingerprint.get('series_id', 'Unknown')}
# Data Hash: {fingerprint.get('hash', 'Unknown')}
# Vintage: {fingerprint.get('vintage', 'Unknown')}
# Observations: {len(data)}
# Date Range: {fingerprint.get('date_range', ['Unknown', 'Unknown'])}
"""
                # Write with header
                with open(output_file, 'w') as f:
                    f.write(provenance_header)
                    df.to_csv(f, index=False)
            else:
                # Write without provenance header
                df.to_csv(output_file, index=False)
            
            saved_files[region] = output_file
            logger.info("Saved %s GDP data to: %s", region, output_file)
        
        return saved_files
    
    def get_validation_summary(self) -> str:
        """Get a summary of the last validation results."""
        if not self.validation_results:
            return "No validation results available"
        
        summary_lines = []
        for region, result in self.validation_results.items():
            if isinstance(result, dict):
                summary_lines.append(f"{region}: {result.get('status', 'unknown')}")
            else:
                summary_lines.append(f"{region}: {str(result)}")
        
        return "\n".join(summary_lines)


def fetch_distinct_gdp_data(regions: List[str] = None, 
                           output_dir: Optional[Path] = None,
                           validate: bool = True) -> Tuple[Dict[str, pd.Series], Optional[ValidationResult]]:
    """Convenience function to fetch distinct GDP data with validation.
    
    Parameters
    ----------
    regions : list, optional
        List of regions to fetch. Defaults to ['US', 'EU27_2020', 'CN']
    output_dir : Path, optional
        If provided, save datasets to CSV files in this directory
    validate : bool, default True
        Whether to run data validation
        
    Returns
    -------
    tuple
        (datasets, validation_result)
    """
    fetcher = EnhancedGDPFetcher()
    datasets, validation_result = fetcher.fetch_all_regions(regions, validate)
    
    if output_dir:
        fetcher.save_datasets(datasets, output_dir)
    
    return datasets, validation_result


if __name__ == "__main__":
    # Simple CLI for testing
    import sys
    
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s %(levelname)s %(name)s - %(message)s')
    
    regions = sys.argv[1:] if len(sys.argv) > 1 else ['US', 'EU27_2020', 'CN']
    
    print(f"Fetching GDP data for regions: {regions}")
    datasets, validation = fetch_distinct_gdp_data(regions, Path('data'), validate=True)
    
    print(f"\nFetched {len(datasets)} datasets:")
    for region, data in datasets.items():
        print(f"  {region}: {len(data)} observations ({data.index.min()} to {data.index.max()})")
    
    if validation:
        print(f"\nValidation: {validation.summary()}")
        if validation.issues:
            print("Issues:")
            for issue in validation.issues:
                print(f"  - [{issue.severity.value.upper()}] {issue.message}")