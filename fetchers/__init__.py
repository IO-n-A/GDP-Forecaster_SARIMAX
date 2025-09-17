"""
Data Fetchers for GDP-ForecasterSARIMAX

This module provides consolidated data fetching capabilities for:
- GDP data from multiple sources (OECD, FRED)
- Exogenous indicators (PMI, ESI, Industrial Production, CLI)
- Configuration-driven fetching with validation

Main Components:
- EnhancedGDPFetcher: Modern configuration-driven GDP fetching
- Regional exogenous data fetchers (US, EU, China)
- Cross-country harmonized indicators (OECD CLI)
- Utility functions for data normalization and validation
"""

# Core GDP fetching
from .enhanced_gdp_fetcher import (
    EnhancedGDPFetcher,
    fetch_distinct_gdp_data,
)

# OECD data fetching
from .oecd_fetchers import (
    fetch_oecd_qna,
    save_csv,
)

# FRED utilities
from .gdp_fetchers import (
    fred_search,
    fred_observations,
    recommended_series,
    RECOMMENDED_FRED_SERIES,
)

# NEW: FRED indicators for US
from .fred_indicators import (
    FredIndicatorsFetcher,
    fetch_us_industrial_production,
    fetch_us_manufacturers_orders,
)

# NEW: OECD CLI for cross-country indicators
from .oecd_cli import (
    OECDCLIFetcher,
    fetch_oecd_cli,
    get_available_locations,
    get_available_subjects,
)

# Exogenous data fetchers
from .dbnomics_pmi import (
    fetch_ism_pmi_csv,
    ism_pmi_monthly,
    normalize_ism_pmi_df,
)

from .eurostat_bcs import (
    fetch_esi_sdmx,
    fetch_industry_conf_sdmx,
    to_monthly_series,
)

from .nbs_pmi import (
    normalize_nbs_pmi_df,
    load_nbs_pmi_csv,
    fetch_dbnomics_nbs,
)

# Legacy CLI support (maintained for backward compatibility)
from .fetch_gdp import (
    normalize_regions,
    normalize_oecd_regions,
    file_tag_for_region,
    label_for_region,
    fetch_region_series,
)

__all__ = [
    # Core GDP fetching
    'EnhancedGDPFetcher',
    'fetch_distinct_gdp_data',
    
    # OECD utilities  
    'fetch_oecd_qna',
    'save_csv',
    
    # FRED utilities
    'fred_search',
    'fred_observations',
    'recommended_series',
    'RECOMMENDED_FRED_SERIES',
    
    # NEW: FRED indicators
    'FredIndicatorsFetcher',
    'fetch_us_industrial_production',
    'fetch_us_manufacturers_orders',
    
    # NEW: OECD CLI
    'OECDCLIFetcher',
    'fetch_oecd_cli',
    'get_available_locations',
    'get_available_subjects',
    
    # US indicators
    'fetch_ism_pmi_csv',
    'ism_pmi_monthly', 
    'normalize_ism_pmi_df',
    
    # EU indicators
    'fetch_esi_sdmx',
    'fetch_industry_conf_sdmx',
    'to_monthly_series',
    
    # China indicators
    'normalize_nbs_pmi_df',
    'load_nbs_pmi_csv',
    'fetch_dbnomics_nbs',
    
    # Legacy/CLI utilities
    'normalize_regions',
    'normalize_oecd_regions',
    'file_tag_for_region',
    'label_for_region',
    'fetch_region_series',
]


# Convenience factory functions for region-specific fetching
def get_gdp_fetcher():
    """Get the main GDP fetcher instance."""
    return EnhancedGDPFetcher()


def fetch_us_indicators(indicator_type='ism_pmi'):
    """
    Fetch US exogenous indicators.
    
    Parameters
    ----------
    indicator_type : str, default 'ism_pmi'
        'ism_pmi', 'industrial_production', 'manufacturers_orders'
        
    Returns
    -------
    pd.Series
        Monthly indicator series
    """
    if indicator_type == 'ism_pmi':
        return ism_pmi_monthly()
    elif indicator_type == 'industrial_production':
        return fetch_us_industrial_production()
    elif indicator_type == 'manufacturers_orders':
        return fetch_us_manufacturers_orders()
    else:
        raise ValueError(f"Unknown US indicator type: {indicator_type}")


def fetch_eu_indicators(indicator='esi'):
    """
    Fetch EU exogenous indicators.
    
    Parameters
    ----------
    indicator : str
        'esi' for Economic Sentiment Indicator or 'industry' for Industry Confidence
        
    Returns
    -------
    pd.Series
        Monthly indicator series
    """
    if indicator == 'esi':
        df = fetch_esi_sdmx()
        return to_monthly_series(df, out_name='ESI')
    elif indicator == 'industry':
        df = fetch_industry_conf_sdmx()
        return to_monthly_series(df, out_name='INDUSTRY_CONF')
    else:
        raise ValueError(f"Unknown EU indicator: {indicator}")


def fetch_china_indicators(url=None):
    """
    Fetch China exogenous indicators (NBS PMI).
    
    Parameters
    ----------
    url : str, optional
        Custom URL for NBS PMI data
        
    Returns
    -------
    pd.Series
        Monthly PMI series
    """
    if url:
        return fetch_dbnomics_nbs(url)
    else:
        # Default NBS PMI URL could be configured
        raise NotImplementedError("Default China PMI URL not yet configured")


def fetch_harmonized_indicators(location, start_period=None):
    """
    Fetch OECD CLI harmonized indicators for any supported location.
    
    Parameters
    ----------
    location : str
        Location code or name ('US', 'EU', 'CN', etc.)
    start_period : str, optional
        Start period in YYYY-MM format
        
    Returns
    -------
    pd.Series
        Monthly CLI series
    """
    return fetch_oecd_cli(location, start_period=start_period)