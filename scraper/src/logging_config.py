"""
Logging Configuration Module for EUR-Lex Web Scraper

This module provides a flexible and comprehensive logging configuration 
system using the Loguru library. It supports dynamic configuration 
of logging parameters, including console and file-based logging.

Key Features:
- Configurable logging levels
- Console and file logging support
- Automatic log file rotation and retention
- Detailed error diagnostics
- Configuration via YAML file

Logging Capabilities:
- Configurable log format
- Log rotation and retention management
- Backtrace and diagnostic information
- Centralized logging configuration
"""

import sys
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger


def setup_logging(config_path: Optional[Path] = None) -> None:
    """
    Configure logging for the EUR-Lex web scraper application.

    Reads logging configuration from a YAML file and sets up 
    logging handlers for console and file-based logging. Provides 
    flexible and detailed logging capabilities.

    Args:
        config_path (Optional[Path], optional): 
            Path to the logging configuration file. 
            Defaults to 'config/config.yaml' in the project root.

    Behavior:
        - Removes default logger
        - Configures console logging to stderr
        - Creates log directory if it doesn't exist
        - Sets up file logging with rotation and retention
        - Enables backtrace and diagnostic logging

    Configuration Options (from YAML):
        - format: Log message format string
        - level: Logging level (e.g., INFO, DEBUG, ERROR)
        - rotation: Log file rotation strategy
        - retention: Log file retention period

    Notes:
        - Uses Loguru for advanced logging features
        - Supports dynamic configuration via external YAML file
        - Provides comprehensive logging for debugging and monitoring
    """
    # Load configuration
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)["logging"]
    
    # Remove default logger
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=config["format"],
        level=config["level"],
        backtrace=True,
        diagnose=True,
    )
    
    # Add file handler
    log_path = Path(__file__).parent.parent / "logs" / "eurlex_scraper.log"
    log_path.parent.mkdir(exist_ok=True)
    
    logger.add(
        str(log_path),
        rotation=config["rotation"],
        retention=config["retention"],
        format=config["format"],
        level=config["level"],
        backtrace=True,
        diagnose=True,
    )
    
    logger.info("Logging configured successfully")
