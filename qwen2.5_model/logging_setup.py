#!/usr/bin/env python3
"""
Logging setup utility for training and inference scripts.

Provides consistent logging configuration across all scripts.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    script_name: str,
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    console_level: int = logging.INFO
) -> logging.Logger:
    """
    Set up logging for a script.
    
    Args:
        script_name: Name of the script (e.g., 'generate_attestation_dataset')
        log_dir: Directory for log files (default: ./logs)
        level: Logging level for file handler
        console_level: Logging level for console handler
        
    Returns:
        Configured logger instance
    """
    # Determine log directory
    if log_dir is None:
        script_path = Path(__file__).parent
        log_dir = script_path / "logs"
    else:
        log_dir = Path(log_dir)
    
    # Create logs directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)  # Capture all levels, filter at handler
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # File handler - log everything with timestamps
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{script_name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - less verbose
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Log the log file location
    logger.info(f"Logging to: {log_file}")
    
    return logger


def log_exception(logger: logging.Logger, exc: Exception, context: str = ""):
    """
    Log an exception with full traceback.
    
    Args:
        logger: Logger instance
        exc: Exception object
        context: Additional context message
    """
    import traceback
    if context:
        logger.error(f"{context}: {exc}")
    else:
        logger.error(f"Exception: {exc}")
    logger.debug(f"Traceback:\n{traceback.format_exc()}")


# Convenience function for creating a logger with default settings
def get_logger(script_name: str) -> logging.Logger:
    """Get a logger with default settings."""
    return setup_logging(script_name)

