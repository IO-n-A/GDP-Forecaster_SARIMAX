"""
FRED Indicators Fetcher

Fetches US economic indicators from FRED (Federal Reserve Economic Data)
for use as exogenous variables in SARIMAX models.

Key indicators:
- Industrial Production (INDPRO)
- Manufacturing Industrial Production (IPMANSICS)  
- Manufacturers' New Orders (AMTMNO)
"""

import logging
import requests
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Default FRED series for US indicators
DEFAULT_FRED_SERIES = {
    'industrial_production': 'INDPRO',
    'manufacturing_ip': 'IPMANSICS', 
    'manufacturers_orders': 'AMTMNO',
    'capacity_utilization': 'TCU',
}

class FredIndicatorsFetcher:
    """Fetcher for US economic indicators from FRED API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED indicators fetcher.
        
        Parameters
        ----------
        api_key : str, optional
            FRED API key. If None, will attempt to get from environment or config.
        """
        self.api_key = self._resolve_api_key(api_key)
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        
    def _resolve_api_key(self, explicit_key: Optional[str] = None) -> Optional[str]:
        """Resolve FRED API key from various sources."""
        if explicit_key:
            return explicit_key
            
        # Try environment variable
        import os
        api_key = os.getenv('FRED_API_KEY')
        if api_key:
            logger.debug("Using FRED API key from environment variable")
            return api_key
            
        # Try configuration system
        try:
            from config import get_api_key
            api_key = get_api_key('fred')
            if api_key:
                logger.debug("Using FRED API key from configuration")
                return api_key
        except ImportError:
            pass
        except Exception as e:
            logger.debug("Failed to get FRED API key from config: %s", e)
            
        logger.warning("No FRED API key found - some functionality may be limited")
        return None
    
    def fetch_series(self, series_id: str, start_date: Optional[str] = None, 
                    end_date: Optional[str] = None) -> pd.Series:
        """
        Fetch a FRED time series.
        
        Parameters
        ----------
        series_id : str
            FRED series ID (e.g., 'INDPRO')
        start_date : str, optional
            Start date in YYYY-MM-DD format
        end_date : str, optional  
            End date in YYYY-MM-DD format
            
        Returns
        -------
        pd.Series
            Time series data with datetime index
        """
        if not self.api_key:
            raise ValueError("FRED API key is required for data fetching")
            
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'sort_order': 'asc',
        }
        
        if start_date:
            params['observation_start'] = start_date
        if end_date:
            params['observation_end'] = end_date
            
        try:
            logger.debug("Fetching FRED series: %s", series_id)
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            observations = data.get('observations', [])
            
            if not observations:
                logger.warning("No observations found for series %s", series_id)
                return pd.Series(dtype=float)
                
            # Convert to DataFrame then Series
            df = pd.DataFrame(observations)
            
            # Clean and convert data
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Filter out missing values (marked as '.')
            df = df.dropna(subset=['value'])
            
            if df.empty:
                logger.warning("No valid data points for series %s", series_id)
                return pd.Series(dtype=float)
                
            # Create series with datetime index
            series = pd.Series(df['value'].values, index=df['date'], name=series_id)
            series = series.sort_index()
            
            logger.info("Fetched %d observations for FRED series %s", len(series), series_id)
            return series
            
        except requests.exceptions.RequestException as e:
            logger.error("Failed to fetch FRED series %s: %s", series_id, e)
            raise
        except Exception as e:
            logger.error("Error processing FRED series %s: %s", series_id, e)
            raise
    
    def fetch_industrial_production(self, start_date: Optional[str] = None,
                                  end_date: Optional[str] = None,
                                  series_type: str = 'total') -> pd.Series:
        """
        Fetch US Industrial Production index.
        
        Parameters
        ----------
        start_date : str, optional
            Start date in YYYY-MM-DD format
        end_date : str, optional
            End date in YYYY-MM-DD format  
        series_type : str, default 'total'
            Type of IP series: 'total', 'manufacturing', 'capacity_util'
            
        Returns
        -------
        pd.Series
            Monthly industrial production series
        """
        series_map = {
            'total': 'INDPRO',
            'manufacturing': 'IPMANSICS',  
            'capacity_util': 'TCU',
        }
        
        if series_type not in series_map:
            raise ValueError(f"Unknown series_type: {series_type}. "
                           f"Valid options: {list(series_map.keys())}")
        
        series_id = series_map[series_type]
        return self.fetch_series(series_id, start_date, end_date)
    
    def fetch_manufacturers_orders(self, start_date: Optional[str] = None,
                                 end_date: Optional[str] = None) -> pd.Series:
        """
        Fetch Manufacturers' New Orders.
        
        Parameters
        ----------
        start_date : str, optional
            Start date in YYYY-MM-DD format
        end_date : str, optional
            End date in YYYY-MM-DD format
            
        Returns
        -------
        pd.Series
            Monthly manufacturers' new orders series
        """
        return self.fetch_series('AMTMNO', start_date, end_date)


def fetch_us_industrial_production(api_key: Optional[str] = None,
                                 start_date: Optional[str] = None,
                                 series_type: str = 'total') -> pd.Series:
    """
    Convenience function to fetch US Industrial Production.
    
    Parameters
    ----------
    api_key : str, optional
        FRED API key
    start_date : str, optional
        Start date in YYYY-MM-DD format
    series_type : str, default 'total'
        Type of series: 'total', 'manufacturing', 'capacity_util'
        
    Returns
    -------
    pd.Series
        Monthly industrial production series
    """
    fetcher = FredIndicatorsFetcher(api_key)
    return fetcher.fetch_industrial_production(start_date=start_date, series_type=series_type)


def fetch_us_manufacturers_orders(api_key: Optional[str] = None,
                                start_date: Optional[str] = None) -> pd.Series:
    """
    Convenience function to fetch US Manufacturers' New Orders.
    
    Parameters
    ----------
    api_key : str, optional
        FRED API key
    start_date : str, optional
        Start date in YYYY-MM-DD format
        
    Returns
    -------
    pd.Series
        Monthly manufacturers' new orders series
    """
    fetcher = FredIndicatorsFetcher(api_key)
    return fetcher.fetch_manufacturers_orders(start_date=start_date)