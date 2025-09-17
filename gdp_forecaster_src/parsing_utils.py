# gdp_forecaster_src/parsing_utils.py

import argparse
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


def parse_range_arg(s: Optional[str], default: str = "0-3", config_key: Optional[str] = None, 
                   args: Optional[argparse.Namespace] = None) -> List[int]:
    """
    Parse a CLI range argument like '0-3' or '0,1,2,3' into a list of integers.
    
    This function handles different input formats for specifying parameter ranges:
    - Range format: "0-3" becomes [0, 1, 2, 3]
    - List format: "0,1,2,3" becomes [0, 1, 2, 3]
    - Mixed scenarios with configuration system fallback
    
    Parameters
    ----------
    s : str, optional
        CLI range argument string to parse
    default : str, default="0-3"
        Default range if no CLI arg or config value provided
    config_key : str, optional
        Configuration key path for fallback value
    args : argparse.Namespace, optional
        CLI arguments for precedence checking
        
    Returns
    -------
    List[int]
        Parsed range as sorted list of unique integers
        
    Examples
    --------
    >>> parse_range_arg("0-3")
    [0, 1, 2, 3]
    >>> parse_range_arg("0,2,4")
    [0, 2, 4]
    >>> parse_range_arg("1-2")
    [1, 2]
    """
    from .config_utils import get_config_value
    
    # Use configuration-aware helper for getting the value
    if config_key:
        range_value = get_config_value(config_key, default, args, None)
        # If we got a list from config, return it directly
        if isinstance(range_value, list):
            return sorted(set(int(x) for x in range_value))
        # Otherwise treat as string to parse
        txt = str(range_value).strip() if range_value is not None else default
    else:
        txt = (s or default).strip()
    
    out: List[int] = []
    
    # Handle range format (e.g., "0-3")
    if "-" in txt and "," not in txt:
        try:
            a, b = txt.split("-", 1)
            lo = int(a.strip())
            hi = int(b.strip())
            out = list(range(lo, hi + 1))
        except Exception:
            pass
    # Handle comma-separated list format (e.g., "0,1,2,3")
    else:
        try:
            out = [int(x.strip()) for x in txt.split(",") if x.strip() != ""]
        except Exception:
            out = []
    
    # Fallback to default if parsing failed
    if not out:
        out = [0, 1, 2, 3]
    
    return sorted(set(out))


def parse_intervals_arg(s: Optional[str], default: str = "80,95") -> List[int]:
    """
    Parse a CLI intervals argument like '80,95' into sorted unique integer coverage levels.
    
    This function converts comma-separated confidence interval specifications into
    a list of coverage percentages, ensuring they are valid (between 1 and 99).
    
    Parameters
    ----------
    s : str, optional
        CLI intervals argument (e.g., "80,95" or "90")
    default : str, default="80,95"
        Default intervals if parsing fails
        
    Returns
    -------
    List[int]
        Sorted list of unique coverage levels as integers between 1 and 99
        
    Examples
    --------
    >>> parse_intervals_arg("80,95")
    [80, 95]
    >>> parse_intervals_arg("90")
    [90]
    >>> parse_intervals_arg("70,80,90,95")
    [70, 80, 90, 95]
    """
    txt = (s or default).strip()
    try:
        vals = sorted({int(x.strip()) for x in txt.split(",") if x.strip() != ""})
        # Filter to valid percentage range
        vals = [v for v in vals if 1 <= v < 100]
        return vals or [80, 95]  # fallback to default if no valid values
    except Exception:
        return [80, 95]  # fallback to default on any parsing error


def parse_batch_countries(country_string: str, default: str = "US,EU27_2020,CN") -> List[str]:
    """
    Parse a comma-separated string of country codes into a list.
    
    This function handles the parsing of batch country specifications for
    multi-country analysis runs.
    
    Parameters
    ----------
    country_string : str
        Comma-separated string of country codes
    default : str, default="US,EU27_2020,CN"
        Default countries if input is empty
        
    Returns
    -------
    List[str]
        List of cleaned country codes
        
    Examples
    --------
    >>> parse_batch_countries("US,EU,CN")
    ['US', 'EU', 'CN']
    >>> parse_batch_countries(" US , EU27_2020 , CN ")
    ['US', 'EU27_2020', 'CN']
    """
    countries_str = country_string or default
    return [c.strip() for c in countries_str.split(",") if c.strip()]


def validate_target_transform(transform: str) -> str:
    """
    Validate and normalize target transformation specification.
    
    This function ensures that the target transformation is one of the supported
    types and provides helpful error messages for invalid inputs.
    
    Parameters
    ----------
    transform : str
        Target transformation type to validate
        
    Returns
    -------
    str
        Validated transformation type
        
    Raises
    ------
    ValueError
        If the transformation type is not supported
        
    Examples
    --------
    >>> validate_target_transform("level")
    'level'
    >>> validate_target_transform("qoq")
    'qoq'
    """
    valid_transforms = ["level", "qoq", "yoy", "log_diff"]
    if transform not in valid_transforms:
        raise ValueError(f"Invalid target transform '{transform}'. Must be one of: {valid_transforms}")
    return transform


def validate_log_level(log_level: str) -> str:
    """
    Validate and normalize logging level specification.
    
    Parameters
    ----------
    log_level : str
        Logging level to validate
        
    Returns
    -------
    str
        Validated logging level
        
    Raises
    ------
    ValueError
        If the logging level is not supported
    """
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    level_upper = log_level.upper()
    if level_upper not in valid_levels:
        raise ValueError(f"Invalid log level '{log_level}'. Must be one of: {valid_levels}")
    return level_upper