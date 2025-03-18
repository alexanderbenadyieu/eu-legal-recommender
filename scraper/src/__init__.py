"""
EUR-Lex Document Scraper - Core Module

This package contains the core functionality for scraping legislative documents from EUR-Lex.
"""

from .scraper import EURLexScraper
from .parsers import DocumentParser
from .storage import DocumentStorage
from .document_tracker import DocumentTracker
from .config_manager import ConfigManager
from .exceptions import (
    ScraperError,
    ParserError,
    StorageError,
    ValidationError,
    InvalidDateError,
    ConfigError
)

__all__ = [
    'EURLexScraper',
    'DocumentParser',
    'DocumentStorage',
    'DocumentTracker',
    'ConfigManager',
    'ScraperError',
    'ParserError',
    'StorageError',
    'ValidationError',
    'InvalidDateError',
    'ConfigError'
]
