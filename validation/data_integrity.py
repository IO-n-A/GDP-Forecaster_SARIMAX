"""Data integrity and provenance tracking for GDP-ForecasterSARIMAX.

This module implements the data fingerprinting system and provenance tracking
to ensure data uniqueness and maintain complete data lineage.

Features:
- SHA-256 data fingerprinting
- Data provenance tracking (source, vintage, series_id)
- Validation pipelines to prevent identical datasets
- Enhanced metrics CSV schema with provenance columns
- Temporal alignment validation
"""

import hashlib
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DataFingerprint:
    """Data fingerprint with SHA-256 hash and metadata."""
    
    hash: str                           # SHA-256 hash (truncated to 16 chars)
    full_hash: str                      # Full SHA-256 hash
    shape: Tuple[int, ...]              # Data shape
    date_range: Tuple[str, str]         # (min_date, max_date) as ISO strings
    source: Optional[str] = None        # Data source (e.g., "FRED", "OECD")
    vintage: Optional[str] = None       # Data collection timestamp
    series_id: Optional[str] = None     # Original series identifier
    region: Optional[str] = None        # Geographic region
    frequency: Optional[str] = None     # Data frequency
    
    @classmethod
    def from_series(cls, data: pd.Series, source: Optional[str] = None, 
                   series_id: Optional[str] = None, region: Optional[str] = None) -> 'DataFingerprint':
        """Create a DataFingerprint from a pandas Series.
        
        Parameters
        ----------
        data : pd.Series
            Time series data with DatetimeIndex
        source : str, optional
            Data source identifier
        series_id : str, optional
            Original series identifier
        region : str, optional
            Geographic region code
            
        Returns
        -------
        DataFingerprint
            Data fingerprint object
        """
        if data.empty:
            raise ValueError("Cannot create fingerprint from empty series")
        
        # Compute hash
        full_hash = cls._compute_hash(data)
        hash_short = full_hash[:16]
        
        # Get date range
        if hasattr(data.index, 'min') and hasattr(data.index, 'max'):
            try:
                date_range = (
                    data.index.min().isoformat(),
                    data.index.max().isoformat()
                )
            except Exception:
                date_range = ("unknown", "unknown")
        else:
            date_range = ("unknown", "unknown")
        
        # Infer frequency
        frequency = None
        if hasattr(data.index, 'freq') and data.index.freq:
            frequency = str(data.index.freq)
        elif hasattr(data.index, 'inferred_freq') and data.index.inferred_freq:
            frequency = data.index.inferred_freq
        
        return cls(
            hash=hash_short,
            full_hash=full_hash,
            shape=data.shape,
            date_range=date_range,
            source=source,
            vintage=datetime.now().isoformat(),
            series_id=series_id,
            region=region,
            frequency=frequency
        )
    
    @staticmethod
    def _compute_hash(data: pd.Series) -> str:
        """Compute SHA-256 hash of data values and index.
        
        Parameters
        ----------
        data : pd.Series
            Input data series
            
        Returns
        -------
        str
            SHA-256 hash as hexadecimal string
        """
        try:
            # Create content string from values and index
            values_bytes = data.values.tobytes()
            
            # Handle index - convert to string representation for hashing
            if hasattr(data.index, 'values'):
                index_str = str(data.index.values).encode('utf-8')
            else:
                index_str = str(list(data.index)).encode('utf-8')
            
            # Combine values and index for hashing
            content = values_bytes + index_str
            
            return hashlib.sha256(content).hexdigest()
            
        except Exception as e:
            logger.warning("Error computing hash, using fallback method: %s", e)
            # Fallback: hash string representation
            content_str = f"{data.values.tolist()}{list(data.index)}"
            return hashlib.sha256(content_str.encode('utf-8')).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass 
class ProvenanceTracker:
    """Tracks data provenance and lineage information."""
    
    fingerprints: Dict[str, DataFingerprint]
    config_hash: Optional[str] = None
    eval_timestamp: Optional[str] = None
    
    def __post_init__(self):
        """Initialize with current evaluation timestamp."""
        if self.eval_timestamp is None:
            self.eval_timestamp = datetime.now().isoformat()
    
    def add_data(self, key: str, data: pd.Series, source: Optional[str] = None,
                series_id: Optional[str] = None, region: Optional[str] = None) -> DataFingerprint:
        """Add data series and create fingerprint.
        
        Parameters
        ----------
        key : str
            Identifier for the data series
        data : pd.Series
            Time series data
        source : str, optional
            Data source identifier
        series_id : str, optional
            Original series identifier  
        region : str, optional
            Geographic region
            
        Returns
        -------
        DataFingerprint
            Created fingerprint
        """
        fingerprint = DataFingerprint.from_series(data, source, series_id, region)
        self.fingerprints[key] = fingerprint
        logger.debug("Added data fingerprint for %s: %s", key, fingerprint.hash)
        return fingerprint
    
    def get_fingerprint(self, key: str) -> Optional[DataFingerprint]:
        """Get fingerprint for a data key."""
        return self.fingerprints.get(key)
    
    def validate_uniqueness(self) -> Tuple[bool, List[str]]:
        """Validate that all tracked data series are unique.
        
        Returns
        -------
        tuple
            (is_valid, list_of_duplicate_keys)
        """
        hash_to_keys = {}
        duplicates = []
        
        for key, fingerprint in self.fingerprints.items():
            hash_val = fingerprint.full_hash
            if hash_val in hash_to_keys:
                duplicates.append(f"{key} duplicates {hash_to_keys[hash_val]}")
                logger.warning("Data duplication detected: %s has same hash as %s", 
                             key, hash_to_keys[hash_val])
            else:
                hash_to_keys[hash_val] = key
        
        return len(duplicates) == 0, duplicates
    
    def set_config_hash(self, config_dict: Dict[str, Any]) -> str:
        """Set configuration hash for reproducibility.
        
        Parameters
        ----------
        config_dict : dict
            Configuration dictionary to hash
            
        Returns
        -------
        str
            Configuration hash
        """
        config_str = str(sorted(config_dict.items()))
        self.config_hash = hashlib.sha256(config_str.encode('utf-8')).hexdigest()[:16]
        return self.config_hash


class ValidationPipeline:
    """Data validation pipeline to ensure data integrity."""
    
    def __init__(self):
        self.provenance_tracker = ProvenanceTracker({})
    
    def validate_data_uniqueness(self, datasets: Dict[str, pd.Series]) -> bool:
        """Prevent identical datasets from being used for different regions.
        
        Parameters
        ----------
        datasets : dict
            Dictionary mapping region names to data series
            
        Returns
        -------
        bool
            True if all datasets are unique, False otherwise
        """
        logger.info("Validating data uniqueness across %d datasets", len(datasets))
        
        # Clear existing fingerprints
        self.provenance_tracker.fingerprints.clear()
        
        # Create fingerprints for all datasets
        for region, data in datasets.items():
            if data is not None and not data.empty:
                self.provenance_tracker.add_data(
                    key=region,
                    data=data,
                    region=region
                )
        
        # Validate uniqueness
        is_valid, duplicates = self.provenance_tracker.validate_uniqueness()
        
        if not is_valid:
            logger.error("Data uniqueness validation failed:")
            for dup in duplicates:
                logger.error("  - %s", dup)
            return False
        
        logger.info("Data uniqueness validation passed - all %d datasets are unique", len(datasets))
        return True
    
    def validate_temporal_alignment(self, endog: pd.Series, exog: pd.DataFrame) -> bool:
        """Ensure no look-ahead bias in exogenous variables.
        
        Parameters
        ----------
        endog : pd.Series
            Endogenous (target) series
        exog : pd.DataFrame
            Exogenous variables
            
        Returns
        -------
        bool
            True if temporal alignment is valid
        """
        if exog is None or exog.empty:
            return True
            
        # Check that exogenous data doesn't extend beyond endogenous data
        endog_end = endog.index.max()
        exog_end = exog.index.max()
        
        if exog_end > endog_end:
            logger.warning("Exogenous data extends beyond endogenous data: %s > %s",
                         exog_end, endog_end)
            # This might be acceptable for forecasting, so just warn
        
        # Check for sufficient overlap
        overlap = endog.index.intersection(exog.index)
        overlap_ratio = len(overlap) / len(endog.index)
        
        if overlap_ratio < 0.8:
            logger.warning("Limited temporal overlap between endog and exog: %.1f%%", 
                         overlap_ratio * 100)
            return False
        
        logger.debug("Temporal alignment validation passed: %.1f%% overlap", 
                    overlap_ratio * 100)
        return True
    
    def get_provenance_data(self) -> Dict[str, Any]:
        """Get provenance data for metrics enhancement.
        
        Returns
        -------
        dict
            Provenance data dictionary
        """
        return {
            'fingerprints': {k: v.to_dict() for k, v in self.provenance_tracker.fingerprints.items()},
            'config_hash': self.provenance_tracker.config_hash,
            'eval_timestamp': self.provenance_tracker.eval_timestamp
        }


# Convenience functions
def create_data_fingerprint(data: pd.Series, source: Optional[str] = None,
                          series_id: Optional[str] = None, region: Optional[str] = None) -> DataFingerprint:
    """Create a data fingerprint for a series.
    
    Parameters
    ----------
    data : pd.Series
        Time series data
    source : str, optional
        Data source identifier
    series_id : str, optional
        Original series identifier
    region : str, optional
        Geographic region
        
    Returns
    -------
    DataFingerprint
        Data fingerprint
    """
    return DataFingerprint.from_series(data, source, series_id, region)


def validate_data_uniqueness(datasets: Dict[str, pd.Series]) -> Tuple[bool, List[str]]:
    """Validate uniqueness across multiple datasets.
    
    Parameters
    ----------
    datasets : dict
        Dictionary mapping names to data series
        
    Returns
    -------
    tuple
        (is_valid, list_of_issues)
    """
    pipeline = ValidationPipeline()
    is_valid = pipeline.validate_data_uniqueness(datasets)
    
    if is_valid:
        return True, []
    else:
        _, duplicates = pipeline.provenance_tracker.validate_uniqueness()
        return False, duplicates


def enhance_metrics_with_provenance(metrics_row: Dict[str, Any], 
                                   provenance_data: Dict[str, Any],
                                   region: str) -> Dict[str, Any]:
    """Enhance metrics row with provenance information.
    
    Parameters
    ----------
    metrics_row : dict
        Existing metrics row
    provenance_data : dict
        Provenance tracking data
    region : str
        Region identifier
        
    Returns
    -------
    dict
        Enhanced metrics row with provenance columns
    """
    enhanced_row = metrics_row.copy()
    
    # Add provenance columns
    fingerprints = provenance_data.get('fingerprints', {})
    region_fingerprint = fingerprints.get(region, {})
    
    enhanced_row.update({
        'data_hash': region_fingerprint.get('hash', ''),
        'source': region_fingerprint.get('source', ''),
        'vintage': region_fingerprint.get('vintage', ''),
        'series_id': region_fingerprint.get('series_id', ''),
        'config_hash': provenance_data.get('config_hash', ''),
        'eval_timestamp': provenance_data.get('eval_timestamp', '')
    })
    
    return enhanced_row


# Enhanced metrics CSV schema definition
ENHANCED_METRICS_SCHEMA = {
    # Existing columns
    'region': str,
    'model': str,
    'metric_name': str,
    'value': float,
    'fold': int,
    
    # New provenance columns
    'data_hash': str,           # Data fingerprint
    'source': str,              # e.g., "FRED", "OECD"
    'vintage': str,             # ISO timestamp
    'series_id': str,           # Original series identifier
    'config_hash': str,         # Configuration fingerprint
    'eval_timestamp': str       # Evaluation run timestamp
}