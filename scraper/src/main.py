"""
EUR-Lex Document Scraper - Command-Line Interface

A comprehensive command-line tool for systematically scraping 
legislative documents from the EUR-Lex repository across specified 
date ranges.

Key Features:
- Date range-based document scraping
- Robust date validation
- Configurable scraping parameters
- Comprehensive logging
- Error handling

Scraping Workflow:
- Validate input date range
- Initialize scraper
- Iterate through dates
- Collect and store documents
- Handle potential errors

Technologies:
- Argparse for CLI argument parsing
- Loguru for advanced logging
- Custom scraping infrastructure
"""

from datetime import datetime, timedelta
from typing import Optional
import argparse
from loguru import logger

from scraper import EURLexScraper
from logging_config import setup_logging
from exceptions import InvalidDateError

def validate_date_range(start_date: datetime, end_date: datetime) -> None:
    """
    Validate the date range for EUR-Lex document scraping.

    Performs comprehensive validation of the input date range, 
    ensuring compliance with scraping constraints and logical requirements.

    Validation Checks:
    - Minimum allowed scraping date (October 2nd, 2023)
    - Chronological order of dates
    - Website-specific scraping limitations

    Args:
        start_date (datetime): Start date for document scraping
        end_date (datetime): End date for document scraping

    Raises:
        InvalidDateError: If start date is before the minimum allowed date
        ValueError: If end date is chronologically before start date

    Notes:
        - Enforces website-specific scraping constraints
        - Provides informative error messages
        - Prevents invalid date range configurations
    """
    min_allowed_date = datetime(2023, 10, 2)
    if start_date < min_allowed_date:
        raise InvalidDateError(
            f"Cannot scrape dates before October 2nd, 2023 due to website structure changes. "
            f"Provided start date: {start_date.date()}"
        )
    if end_date < start_date:
        raise ValueError(f"End date ({end_date.date()}) cannot be before start date ({start_date.date()})")

def scrape_date_range(
    start_date: datetime,
    end_date: datetime,
    scraper: Optional[EURLexScraper] = None
) -> None:
    """
    Systematically scrape EUR-Lex documents across a specified date range.

    Orchestrates the complete document scraping process for a given 
    chronological range, handling initialization, iteration, and error management.

    Workflow:
    1. Validate input date range
    2. Initialize scraper (if not provided)
    3. Iterate through dates
    4. Scrape journal documents
    5. Handle and log potential errors

    Args:
        start_date (datetime): Start date for scraping (inclusive)
        end_date (datetime): End date for scraping (inclusive)
        scraper (Optional[EURLexScraper], optional): 
            Pre-configured EURLexScraper instance. 
            Creates a new instance if not provided.

    Notes:
        - Supports flexible scraper configuration
        - Provides comprehensive error handling
        - Logs scraping activities and potential issues
        - Iterates chronologically through dates
    """
    try:
        validate_date_range(start_date, end_date)
    except InvalidDateError as e:
        logger.error(str(e))
        return
    except ValueError as e:
        logger.error(str(e))
        return

    # Initialize scraper if not provided
    if scraper is None:
        scraper = EURLexScraper()
    
    current_date = start_date
    while current_date <= end_date:
        try:
            logger.info(f"Processing date: {current_date.date()}")
            stored_paths = scraper.scrape_journal(current_date)
            
            if stored_paths:
                logger.success(f"Successfully stored {len(stored_paths)} documents for {current_date.date()}")
            else:
                logger.warning(f"No documents stored for {current_date.date()} "
                             "(either no documents published or already scraped)")
                
        except Exception as e:
            logger.error(f"Error processing date {current_date.date()}: {str(e)}")
        
        finally:
            current_date += timedelta(days=1)

def main() -> None:
    """
    Main entry point for the EUR-Lex document scraper CLI.

    Manages command-line argument parsing, logging configuration, 
    and initiation of the document scraping process.

    CLI Argument Handling:
    - Parse start and end dates
    - Configure logging
    - Initialize scraping workflow

    Workflow:
    1. Set up argument parser
    2. Configure logging
    3. Parse command-line arguments
    4. Validate and convert date inputs
    5. Initiate date range scraping

    Notes:
        - Provides user-friendly CLI for document scraping
        - Supports flexible date range specification
        - Configures logging for comprehensive tracking
        - Handles potential CLI argument errors
    """
    parser = argparse.ArgumentParser(description="EUR-Lex document scraper")
    parser.add_argument(
        "--start-date",
        type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
        help="Start date in YYYY-MM-DD format",
        required=True
    )
    parser.add_argument(
        "--end-date",
        type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
        help="End date in YYYY-MM-DD format",
        required=True
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger.info(f"Starting scraping from {args.start_date.date()} to {args.end_date.date()}")
    scrape_date_range(args.start_date, args.end_date)
    logger.success("Scraping completed")

if __name__ == "__main__":
    main()
