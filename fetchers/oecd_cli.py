"""
OECD Composite Leading Indicators (CLI) Fetcher

Fetches OECD CLI data for use as harmonized cross-country exogenous variables.
CLI is designed to track cyclical turning points and provides consistent 
monthly series across countries.

Reference: https://www.oecd.org/sdd/leading-indicators/oecd-system-of-composite-leading-indicators.htm
"""

import logging
import requests
import pandas as pd
from typing import Optional, Dict, List, Any
from datetime import datetime
import json
from io import StringIO

logger = logging.getLogger(__name__)

# OECD location codes for major economies
OECD_LOCATIONS = {
    'US': 'USA',
    'USA': 'USA', 
    'EU': 'EA19',
    'EU27_2020': 'EA19',
    'EA19': 'EA19',
    'CN': 'CHN',
    'China': 'CHN',
    'CHN': 'CHN',
    'JP': 'JPN',
    'Japan': 'JPN',
    'UK': 'GBR',
    'DE': 'DEU', 
    'Germany': 'DEU',
    'FR': 'FRA',
    'France': 'FRA',
}

# CLI subject codes
CLI_SUBJECTS = {
    'amplitude_adjusted': 'LOLITOAA_GYSA',
    'normalised': 'LOLITOAA_NCSA',
    'trend_restored': 'LOLITOAA_TRSA',
}

class OECDCLIFetcher:
    """Fetcher for OECD Composite Leading Indicators."""
    
    def __init__(self):
        """Initialize OECD CLI fetcher."""
        self.base_url = "https://stats.oecd.org/SDMX-JSON/data/MEI_CLI"
        
    def _build_url(self, location: str, subject: str = 'amplitude_adjusted',
                   measure: str = 'IDX2015', frequency: str = 'M',
                   start_period: Optional[str] = None) -> str:
        """
        Build OECD API URL for CLI data.
        
        Parameters
        ----------
        location : str
            OECD location code (e.g., 'USA', 'EA19', 'CHN')
        subject : str, default 'amplitude_adjusted'
            CLI subject type
        measure : str, default 'IDX2015'
            Measure type (index base year)
        frequency : str, default 'M'
            Data frequency (M for monthly)
        start_period : str, optional
            Start period in YYYY-MM format
            
        Returns
        -------
        str
            Complete OECD API URL
        """
        # Format: LOCATION.SUBJECT.MEASURE.FREQUENCY
        subject_code = CLI_SUBJECTS.get(subject, CLI_SUBJECTS['amplitude_adjusted'])
        
        path_parts = [location, subject_code, measure, frequency]
        path = '.'.join(path_parts)
        
        url = f"{self.base_url}/{path}/all"
        
        if start_period:
            url += f"?startPeriod={start_period}"
            
        return url
    
    def fetch_cli_series(self, location: str, subject: str = 'amplitude_adjusted',
                        start_period: Optional[str] = None) -> pd.Series:
        """
        Fetch OECD CLI series for a specific location.
        
        Parameters
        ----------
        location : str
            Location code or common name (will be mapped to OECD codes)
        subject : str, default 'amplitude_adjusted'
            CLI subject: 'amplitude_adjusted', 'normalised', 'trend_restored'
        start_period : str, optional
            Start period in YYYY-MM format (e.g., '2000-01')
            
        Returns
        -------
        pd.Series
            Monthly CLI series with datetime index
        """
        # Map common location names to OECD codes
        oecd_location = OECD_LOCATIONS.get(location.upper(), location.upper())
        
        url = self._build_url(oecd_location, subject, start_period=start_period)
        
        try:
            logger.debug("Fetching OECD CLI for %s from: %s", oecd_location, url)
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse SDMX-JSON response
            data = response.json()
            
            # Extract observations from SDMX-JSON structure
            datasets = data.get('dataSets', [])
            if not datasets:
                logger.warning("No datasets found for OECD CLI %s", oecd_location)
                return pd.Series(dtype=float)
                
            # Get the first dataset
            dataset = datasets[0]
            observations = dataset.get('observations', {})
            
            if not observations:
                logger.warning("No observations found for OECD CLI %s", oecd_location)
                return pd.Series(dtype=float)
            
            # Extract dimension information for time periods
            structure = data.get('structure', {})
            dimensions = structure.get('dimensions', {})
            observation_dims = dimensions.get('observation', [])
            
            # Find time dimension
            time_dimension = None
            for dim in observation_dims:
                if dim.get('id') == 'TIME_PERIOD':
                    time_dimension = dim
                    break
                    
            if not time_dimension:
                logger.error("Could not find TIME_PERIOD dimension for %s", oecd_location)
                return pd.Series(dtype=float)
                
            # Build time period list
            time_values = time_dimension.get('values', [])
            time_periods = [val.get('id') for val in time_values]
            
            # Extract values
            values_list = []
            dates_list = []
            
            for obs_key, obs_data in observations.items():
                # Parse observation key (typically "0:1:2:3" format)
                key_parts = obs_key.split(':')
                
                if len(key_parts) >= 1:
                    try:
                        time_index = int(key_parts[-1])  # Last part is usually time
                        if time_index < len(time_periods):
                            period = time_periods[time_index]
                            value = obs_data[0] if isinstance(obs_data, list) else obs_data
                            
                            if value is not None:
                                # Convert YYYY-MM or YYYY-QX to datetime
                                try:
                                    if '-' in period:
                                        if len(period) == 7:  # YYYY-MM
                                            date = pd.to_datetime(period + '-01')
                                        elif 'Q' in period:  # YYYY-QX
                                            date = pd.to_datetime(period)
                                        else:
                                            date = pd.to_datetime(period)
                                    else:
                                        # Assume YYYY format
                                        date = pd.to_datetime(period + '-01-01')
                                        
                                    values_list.append(float(value))
                                    dates_list.append(date)
                                except (ValueError, TypeError) as e:
                                    logger.debug("Skipping invalid date/value: %s, %s", period, value)
                                    continue
                                    
                    except (ValueError, IndexError) as e:
                        logger.debug("Skipping invalid observation key: %s", obs_key)
                        continue
            
            if not values_list:
                logger.warning("No valid observations extracted for OECD CLI %s", oecd_location)
                return pd.Series(dtype=float)
            
            # Create series
            series = pd.Series(values_list, index=dates_list, name=f'CLI_{oecd_location}')
            series = series.sort_index()
            
            # Remove duplicates (keep last)
            series = series[~series.index.duplicated(keep='last')]
            
            logger.info("Fetched %d observations for OECD CLI %s", len(series), oecd_location)
            return series
            
        except requests.exceptions.RequestException as e:
            logger.error("Failed to fetch OECD CLI for %s: %s", oecd_location, e)
            return pd.Series(dtype=float)
        except Exception as e:
            logger.error("Error processing OECD CLI for %s: %s", oecd_location, e)
            return pd.Series(dtype=float)


def fetch_oecd_cli(location: str, subject: str = 'amplitude_adjusted',
                   start_period: Optional[str] = None) -> pd.Series:
    """
    Convenience function to fetch OECD CLI series.
    
    Parameters
    ----------
    location : str
        Location code or name ('US', 'EU', 'CN', etc.)
    subject : str, default 'amplitude_adjusted'
        CLI subject type
    start_period : str, optional
        Start period in YYYY-MM format
        
    Returns
    -------
    pd.Series
        Monthly CLI series
    """
    fetcher = OECDCLIFetcher()
    return fetcher.fetch_cli_series(location, subject, start_period)


def get_available_locations() -> Dict[str, str]:
    """
    Get available location mappings for OECD CLI.
    
    Returns
    -------
    dict
        Mapping of common names to OECD location codes
    """
    return OECD_LOCATIONS.copy()


def get_available_subjects() -> Dict[str, str]:
    """
    Get available CLI subject types.
    
    Returns
    -------
    dict
        Mapping of subject names to OECD subject codes
    """
    return CLI_SUBJECTS.copy()
