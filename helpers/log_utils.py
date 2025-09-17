"""
Logging Utilities for GDP-ForecasterSARIMAX Project

This module provides utilities for the logging protocol as specified in analysis/ToDo.md:
- Generate unique IDs and timestamps
- Format log entries according to the specified template
- Handle archiving of backlog files

Template Format:
Date: <YYYY-MM-DD HH:MM:SS.ffffff>, Author: IO-n_A ID: <Generated_or_Sourced_ID>
Checkpoint: <Category> - <Brief, Action-Oriented Title>
Description: <A detailed (300 words), past-tense description of the action taken, including the "why". 
Specify which files were modified (with paths and line numbers if applicable) and the outcome.>
--------------
--------------
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class LogProtocolManager:
    """
    Manages the logging protocol for the GDP-ForecasterSARIMAX project
    following the specification in analysis/ToDo.md
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the logging protocol manager
        
        Parameters
        ----------
        project_root : Path, optional
            Root directory of the project. If None, will auto-detect.
        """
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
            
        self.log_dir = self.project_root / "log"
        self.archive_dir = self.log_dir / "ARCHIVE"
        self.backlog_file = self.log_dir / "backlog.md"
        
        # Ensure directories exist
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_unique_id(self, category: str = "LOG", task_id: Optional[str] = None) -> str:
        """
        Generate a unique ID for log entries
        
        Parameters
        ----------
        category : str
            Category of the task (e.g., "LOG", "IMPL", "DIAG", "VALID")
        task_id : str, optional
            Optional specific task identifier
            
        Returns
        -------
        str
            Unique ID in format: LOG-YYYYMMDD-HHMMSS-CATEGORY-NN
        """
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d-%H%M%S")
        
        if task_id:
            return f"LOG-{timestamp}-{category}-{task_id}"
        else:
            # Generate simple incremental ID
            counter = getattr(self, '_id_counter', 0) + 1
            setattr(self, '_id_counter', counter)
            return f"LOG-{timestamp}-{category}-{counter:02d}"
    
    def generate_timestamp(self) -> str:
        """
        Generate timestamp in the required format
        
        Returns
        -------
        str
            Timestamp in format: YYYY-MM-DD HH:MM:SS.ffffff
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    
    def format_log_entry(
        self, 
        checkpoint_category: str, 
        checkpoint_title: str, 
        description: str,
        custom_id: Optional[str] = None
    ) -> str:
        """
        Format a log entry according to the template specification
        
        Parameters
        ----------
        checkpoint_category : str
            Category of the checkpoint (e.g., "Implementation", "Validation", "Configuration")
        checkpoint_title : str
            Brief, action-oriented title
        description : str
            Detailed past-tense description (should be ~300 words)
        custom_id : str, optional
            Custom ID, if None will generate one
            
        Returns
        -------
        str
            Formatted log entry
        """
        timestamp = self.generate_timestamp()
        if custom_id is None:
            log_id = self.generate_unique_id(category=checkpoint_category[:4].upper())
        else:
            log_id = custom_id
            
        entry = f"""Date: {timestamp}, Author: IO-n_A ID: {log_id}
Checkpoint: {checkpoint_category} - {checkpoint_title}
Description: {description}
--------------
--------------"""
        
        return entry
    
    def archive_backlog(self) -> str:
        """
        Create a timestamped archive copy of the current backlog
        
        Returns
        -------
        str
            Path to the archived file
        """
        if not self.backlog_file.exists():
            logger.warning("Backlog file does not exist: %s", self.backlog_file)
            return ""
            
        # Generate archive filename with timestamp
        now = datetime.now()
        archive_name = now.strftime("%y%m%d-%H:%M_backlog.md")
        archive_path = self.archive_dir / archive_name
        
        try:
            # Copy current backlog to archive
            import shutil
            shutil.copy2(self.backlog_file, archive_path)
            logger.info("Archived backlog to: %s", archive_path)
            return str(archive_path)
        except Exception as e:
            logger.error("Failed to archive backlog: %s", e)
            return ""
    
    def add_log_entry(
        self, 
        checkpoint_category: str, 
        checkpoint_title: str, 
        description: str,
        custom_id: Optional[str] = None,
        archive_first: bool = True
    ) -> bool:
        """
        Add a new log entry to the backlog following the protocol
        
        Parameters
        ----------
        checkpoint_category : str
            Category of the checkpoint
        checkpoint_title : str
            Brief, action-oriented title
        description : str
            Detailed description
        custom_id : str, optional
            Custom ID for the entry
        archive_first : bool
            Whether to archive current backlog before adding entry
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            # Step 1: Archive current backlog if requested
            if archive_first:
                self.archive_backlog()
            
            # Step 2: Format new entry
            new_entry = self.format_log_entry(
                checkpoint_category, 
                checkpoint_title, 
                description, 
                custom_id
            )
            
            # Step 3: Prepend to backlog file
            if self.backlog_file.exists():
                # Read existing content
                existing_content = self.backlog_file.read_text(encoding='utf-8')
                # Prepend new entry
                updated_content = new_entry + "\n" + existing_content
            else:
                updated_content = new_entry
            
            # Write updated content
            self.backlog_file.write_text(updated_content, encoding='utf-8')
            
            logger.info("Added log entry: %s - %s", checkpoint_category, checkpoint_title)
            return True
            
        except Exception as e:
            logger.error("Failed to add log entry: %s", e)
            return False
    
    def validate_log_format(self, entry: str) -> Dict[str, Any]:
        """
        Validate that a log entry follows the required format
        
        Parameters
        ----------
        entry : str
            Log entry to validate
            
        Returns
        -------
        Dict[str, Any]
            Validation results with 'valid' boolean and 'issues' list
        """
        issues = []
        
        lines = entry.strip().split('\n')
        
        # Check required components
        if not any(line.startswith('Date:') for line in lines):
            issues.append("Missing 'Date:' field")
        
        if not any(line.startswith('Checkpoint:') for line in lines):
            issues.append("Missing 'Checkpoint:' field")
            
        if not any(line.startswith('Description:') for line in lines):
            issues.append("Missing 'Description:' field")
            
        # Check for separators
        separator_count = entry.count('--------------')
        if separator_count < 2:
            issues.append("Missing separator lines (need exactly 2)")
        
        # Check author format
        if 'Author: IO-n_A' not in entry:
            issues.append("Missing or incorrect 'Author: IO-n_A' format")
            
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }


def get_log_manager(project_root: Optional[Path] = None) -> LogProtocolManager:
    """
    Get a singleton instance of the log protocol manager
    
    Parameters
    ----------
    project_root : Path, optional
        Root directory of the project
        
    Returns
    -------
    LogProtocolManager
        The log protocol manager instance
    """
    if not hasattr(get_log_manager, '_instance'):
        get_log_manager._instance = LogProtocolManager(project_root)
    return get_log_manager._instance


# Convenience functions for common operations
def log_checkpoint(
    category: str, 
    title: str, 
    description: str,
    custom_id: Optional[str] = None
) -> bool:
    """
    Add a checkpoint log entry
    
    Parameters
    ----------
    category : str
        Category of the checkpoint
    title : str
        Brief title
    description : str
        Detailed description
    custom_id : str, optional
        Custom ID
        
    Returns
    -------
    bool
        Success status
    """
    manager = get_log_manager()
    return manager.add_log_entry(category, title, description, custom_id)


def archive_current_backlog() -> str:
    """
    Archive the current backlog
    
    Returns
    -------
    str
        Path to archived file
    """
    manager = get_log_manager()
    return manager.archive_backlog()
# --- Evaluation log append helper ---

from typing import Optional
from pathlib import Path
from datetime import datetime
import logging

def append_eval_log(origin: str, content: str, eval_file: Optional[Path] = None) -> None:
    """
    Append a timestamped section to analysis/eval_SARIMAX.md.

    Parameters
    ----------
    origin : str
        Logical origin of the content (e.g., 'tests/test_backtesting.py')
    content : str
        Console transcript to append
    eval_file : Path, optional
        Explicit path to the evaluation log; defaults to analysis/eval_SARIMAX.md
    """
    try:
        project_root = Path(__file__).parent.parent
        if eval_file is None:
            eval_file = project_root / "analysis" / "eval_SARIMAX.md"
        eval_file.parent.mkdir(parents=True, exist_ok=True)

        # Timestamp in UTC with Z suffix
        ts = datetime.utcnow().isoformat() + "Z"

        header = f"## {ts} — {origin}\n\n"
        block = "```text\n" + (content.rstrip() if content is not None else "") + "\n```\n\n---\n"

        with open(eval_file, "a", encoding="utf-8") as f:
            f.write(header)
            f.write(block)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Failed to append eval log: %s", e)