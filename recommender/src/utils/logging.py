"""
Logging configuration for the EU Legal Recommender system.

This module provides a standardized logging setup for all components of the system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

from src.config import LOGS_DIR


def setup_logger(
    name: str,
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    console_output: bool = True,
    file_output: bool = True,
) -> logging.Logger:
    """
    Set up a logger with standardized formatting.
    
    Args:
        name: Name of the logger
        log_file: Path to the log file (if None, uses name.log in LOGS_DIR)
        level: Logging level (default: INFO)
        console_output: Whether to output logs to console
        file_output: Whether to output logs to file
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console handler if requested
    if console_output and not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if requested
    if file_output and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        if log_file is None:
            log_file = LOGS_DIR / f"{name}.log"
        else:
            log_file = Path(log_file)
            
        # Ensure parent directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name, creating it if it doesn't exist.
    
    Args:
        name: Name of the logger
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If the logger doesn't have handlers, set it up
    if not logger.handlers:
        return setup_logger(name)
    
    return logger
