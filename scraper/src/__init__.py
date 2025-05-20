"""
EUR-Lex Document Scraper - Core Module

This package contains the core functionality for scraping legislative documents from EUR-Lex.
"""

from .scraper import EURLexScraper
from .parsers import DocumentParser
from .storage import StorageManager
from .document_tracker import DocumentTracker
from .config_manager import ConfigManager
from .exceptions import (
    ScrapingError,
    ParseError,
    StorageError,
    ValidationError,
    InvalidDateError,
    ConfigurationError
)

__all__ = [
    'EURLexScraper',
    'DocumentParser',
    'StorageManager',
    'DocumentTracker',
    'ConfigManager',
    'ScrapingError',
    'ParseError',
    'StorageError',
    'ValidationError',
    'InvalidDateError',
    'ConfigurationError'
]
