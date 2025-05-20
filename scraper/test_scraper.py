#!/usr/bin/env python3
"""
Test script for the EUR-Lex scraper.
This script runs the scraper for a specific date to verify functionality.
"""
import sys
import os
from datetime import datetime
from pathlib import Path
from loguru import logger

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent))

# Import necessary modules
from src.main import scrape_date_range, validate_date_range
from src.logging_config import setup_logging

def main():
    """Run a test scraping for a specific date."""
    # Setup logging
    setup_logging()
    
    # Define the date to scrape (must be on or after October 2nd, 2023)
    test_date = datetime(2024, 1, 15)
    
    logger.info(f"Starting test scraping for date: {test_date.date()}")
    
    try:
        # Validate the date
        validate_date_range(test_date, test_date)
        
        # Run the scraper for the specified date
        scrape_date_range(test_date, test_date)
        
        logger.success("Test scraping completed successfully")
        
    except Exception as e:
        logger.error(f"Test scraping failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
