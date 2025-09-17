# gdp_forecaster_src/file_utils.py

import csv
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> None:
    """
    Create directory if it doesn't exist, including all parent directories.
    
    This is a fundamental utility for ensuring output directories exist before
    writing files, preventing common file operation errors.
    
    Parameters
    ----------
    path : Path
        Directory path to create
        
    Notes
    -----
    Uses mkdir with parents=True and exist_ok=True for safe operation.
    No error is raised if the directory already exists.
    """
    path.mkdir(parents=True, exist_ok=True)


def append_metrics_csv_row(csv_path: Optional[Path], 
                          row: Dict[str, Any], 
                          header: List[str]) -> None:
    """
    Append a single metrics row to CSV, creating header on first write.
    
    This function manages metrics collection across multiple runs by appending
    results to a shared CSV file, creating the file with headers if needed.
    
    Parameters
    ----------
    csv_path : Optional[Path]
        Path to metrics CSV file (None to skip writing)
    row : Dict[str, Any]
        Dictionary containing metric values to write
    header : List[str]
        List of column names for the CSV
        
    Notes
    -----
    - Creates parent directories if they don't exist
    - Writes header row only if file doesn't exist
    - Handles encoding and newline parameters for cross-platform compatibility
    """
    if csv_path is None:
        return
        
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        exists = csv_path.exists()
        
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
            
    except Exception as e:
        logger.error("Failed to append metrics to %s: %s", csv_path, e)


def get_eval_md_path(args: Optional[object], base_dir: Path) -> Path:
    """
    Get the evaluation markdown file path with optional override.
    
    This function determines where to write evaluation markdown output,
    supporting command-line overrides while providing sensible defaults.
    
    Parameters
    ----------
    args : Optional[object]
        Arguments object that may contain eval_md attribute
    base_dir : Path
        Base directory for default path construction
        
    Returns
    -------
    Path
        Path to evaluation markdown file
        
    Notes
    -----
    Default path is {base_dir}/analysis/eval_SARIMAX.md
    Creates parent directories as needed.
    """
    try:
        override = getattr(args, "eval_md", None) if args is not None else None
    except Exception:
        override = None
        
    p = Path(override) if override else (base_dir / "analysis" / "eval_SARIMAX.md")
    ensure_dir(p.parent)
    return p


def append_eval_md(eval_md_path: Path, title: str, body: str) -> None:
    """
    Append a section to evaluation markdown file with timestamp.
    
    This function adds timestamped sections to a markdown evaluation log,
    useful for tracking analysis results over time.
    
    Parameters
    ----------
    eval_md_path : Path
        Path to markdown file
    title : str
        Section title
    body : str
        Section content
        
    Notes
    -----
    Adds ISO timestamp and formats as markdown section with level 2 heading.
    Handles file operations gracefully with error logging.
    """
    try:
        ts = datetime.utcnow().isoformat()
        with eval_md_path.open("a", encoding="utf-8") as f:
            f.write(f"\n\n## {title}  \n")
            f.write(f"_timestamp: {ts}_\n\n")
            f.write(body.strip() + "\n")
    except Exception as e:
        logger.debug("Failed to append to eval markdown %s: %s", eval_md_path, e)


def md_table_from_df(df: pd.DataFrame, 
                    max_rows: int = 10, 
                    columns: Optional[List[str]] = None) -> str:
    """
    Convert a DataFrame to markdown table format.
    
    This function generates markdown-formatted tables from pandas DataFrames,
    useful for including tabular results in markdown reports.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to convert
    max_rows : int, default=10
        Maximum number of rows to include
    columns : Optional[List[str]]
        Specific columns to include (None for all)
        
    Returns
    -------
    str
        Markdown table string, or empty string if conversion fails
        
    Notes
    -----
    - Limits output to max_rows for readability
    - Handles missing columns gracefully
    - Returns empty string on any conversion error
    """
    try:
        if columns is not None:
            keep = [c for c in columns if c in df.columns]
            if keep:
                df = df.loc[:, keep]
                
        df_disp = df.head(max_rows).copy()
        cols = list(df_disp.columns)
        
        if not cols:
            return ""
            
        # Create markdown table
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        separator = "| " + " | ".join("---" for _ in cols) + " |"
        
        rows = []
        for _, row in df_disp.iterrows():
            vals = [str(row[c]) for c in cols]
            rows.append("| " + " | ".join(vals) + " |")
            
        return "\n".join([header, separator] + rows)
        
    except Exception as e:
        logger.debug("Failed to render markdown table: %s", e)
        return ""


def validate_metrics_df(df: pd.DataFrame) -> bool:
    """
    Validate metrics DataFrame schema and warn about data quality issues.
    
    This function checks the structure and content of metrics DataFrames to
    ensure they contain expected columns and identify potential data issues.
    
    Parameters
    ----------
    df : pd.DataFrame
        Metrics DataFrame to validate
        
    Returns
    -------
    bool
        True if schema looks OK (even with duplicates), False if key fields missing
        
    Notes
    -----
    - Checks for required columns
    - Warns about duplicate (country, model) combinations
    - Warns about duplicate forecast hashes
    - Logs recommendations for optional columns
    """
    required = [
        "country", "model", "ME", "MAE", "RMSE", "MAPE", "sMAPE",
        "median_APE", "MASE", "TheilU1", "TheilU2", "DM_t", "DM_p"
    ]
    
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning("metrics.csv missing required columns: %s", missing)
        return False

    # Check for duplicate (country, model) combinations
    try:
        dups = (
            df.assign(_ord_=np.arange(len(df)))
            .groupby(["country", "model"], as_index=False)
            .agg(n=("model", "size"))
        )
        n_dup = int((dups["n"] > 1).sum())
        if n_dup > 0:
            logger.warning(
                "metrics.csv has %d (country, model) groups with multiple rows; "
                "latest row will be used in summaries.", n_dup
            )
    except Exception as e:
        logger.debug("metrics.csv duplicate check failed: %s", e)

    # Check for duplicate forecast hashes
    try:
        if {"hash_SARIMA", "hash_naive"}.issubset(set(df.columns)):
            dups_hash = (
                df.assign(_ord_=np.arange(len(df)))
                .groupby(["country", "model", "hash_SARIMA", "hash_naive"], as_index=False)
                .agg(n=("model", "size"))
            )
            n_dup_hash = int((dups_hash["n"] > 1).sum())
            if n_dup_hash > 0:
                logger.warning(
                    "metrics.csv has %d duplicate groups with identical "
                    "(country, model, hash_SARIMA, hash_naive).", n_dup_hash
                )
    except Exception as e:
        logger.debug("metrics.csv duplicate-by-hash check failed: %s", e)

    # Check for optional columns
    for opt in ["hit_80", "hit_95"]:
        if opt not in df.columns:
            logger.info(
                "metrics.csv does not include optional column '%s' (hit rates); "
                "consider adding.", opt
            )

    return True


def safe_read_csv(csv_path: Path, **kwargs) -> Optional[pd.DataFrame]:
    """
    Safely read CSV file with comprehensive error handling.
    
    This function provides robust CSV reading with detailed error logging
    and graceful failure handling.
    
    Parameters
    ----------
    csv_path : Path
        Path to CSV file
    **kwargs
        Additional arguments passed to pd.read_csv
        
    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame if successful, None if any error occurs
        
    Notes
    -----
    - Checks file existence and size
    - Provides detailed error messages for different failure modes
    - Returns None rather than raising exceptions
    """
    try:
        if not csv_path.exists():
            logger.warning("CSV file not found: %s", csv_path)
            return None
            
        if csv_path.stat().st_size == 0:
            logger.warning("CSV file is empty: %s", csv_path)
            return None
            
        df = pd.read_csv(csv_path, **kwargs)
        
        if df.empty:
            logger.warning("CSV file contains no data: %s", csv_path)
            return None
            
        return df
        
    except pd.errors.EmptyDataError:
        logger.warning("CSV file contains no data: %s", csv_path)
        return None
    except pd.errors.ParserError as e:
        logger.error("Failed to parse CSV file %s: %s", csv_path, e)
        return None
    except Exception as e:
        logger.error("Failed to read CSV file %s: %s", csv_path, e)
        return None


def backup_file(file_path: Path, max_backups: int = 5) -> Optional[Path]:
    """
    Create a backup of an existing file before overwriting.
    
    This function creates numbered backups of files to prevent data loss
    when overwriting existing results.
    
    Parameters
    ----------
    file_path : Path
        Path to file to backup
    max_backups : int, default=5
        Maximum number of backup files to retain
        
    Returns
    -------
    Optional[Path]
        Path to created backup file, or None if backup failed
        
    Notes
    -----
    - Creates backups with .bak.N suffix (N = 1, 2, ...)
    - Rotates backups to keep only max_backups files
    - Returns None if original file doesn't exist
    """
    if not file_path.exists():
        return None
        
    try:
        # Find next backup number
        backup_num = 1
        while True:
            backup_path = file_path.with_suffix(f"{file_path.suffix}.bak.{backup_num}")
            if not backup_path.exists():
                break
            backup_num += 1
            
        # Create backup
        backup_path.write_bytes(file_path.read_bytes())
        
        # Clean up old backups
        cleanup_old_backups(file_path, max_backups)
        
        logger.debug("Created backup: %s", backup_path)
        return backup_path
        
    except Exception as e:
        logger.warning("Failed to create backup of %s: %s", file_path, e)
        return None


def cleanup_old_backups(file_path: Path, max_backups: int) -> None:
    """
    Remove old backup files beyond the specified limit.
    
    Parameters
    ----------
    file_path : Path
        Original file path (backups are .bak.N versions)
    max_backups : int
        Maximum number of backups to retain
    """
    try:
        backup_pattern = f"{file_path.name}.bak.*"
        backup_files = list(file_path.parent.glob(backup_pattern))
        
        if len(backup_files) <= max_backups:
            return
            
        # Sort by modification time (oldest first)
        backup_files.sort(key=lambda p: p.stat().st_mtime)
        
        # Remove oldest files
        for backup_file in backup_files[:-max_backups]:
            backup_file.unlink()
            logger.debug("Removed old backup: %s", backup_file)
            
    except Exception as e:
        logger.debug("Failed to cleanup old backups: %s", e)


def resolve_path(path_str: str, base_dir: Path) -> Path:
    """
    Resolve a path string relative to a base directory if not absolute.
    
    This utility function handles path resolution consistently across the
    application, supporting both absolute and relative paths.
    
    Parameters
    ----------
    path_str : str
        Path string to resolve
    base_dir : Path
        Base directory for relative path resolution
        
    Returns
    -------
    Path
        Resolved absolute path
        
    Examples
    --------
    >>> resolve_path("data/file.csv", Path("/project"))
    Path("/project/data/file.csv")
    >>> resolve_path("/absolute/path.csv", Path("/project"))
    Path("/absolute/path.csv")
    """
    path = Path(path_str)
    return path if path.is_absolute() else (base_dir / path)


def get_file_hash(file_path: Path, algorithm: str = "sha256") -> Optional[str]:
    """
    Calculate hash of file contents for integrity checking.
    
    This function computes file hashes for verifying data integrity
    and detecting changes in input files.
    
    Parameters
    ----------
    file_path : Path
        Path to file to hash
    algorithm : str, default="sha256"
        Hash algorithm to use
        
    Returns
    -------
    Optional[str]
        Hex digest of file hash, or None if error
        
    Notes
    -----
    Reads file in chunks to handle large files efficiently.
    """
    import hashlib
    
    if not file_path.exists():
        return None
        
    try:
        hasher = hashlib.new(algorithm)
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.debug("Failed to compute hash for %s: %s", file_path, e)
        return None