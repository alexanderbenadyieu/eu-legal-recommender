"""Test script for EUR-Lex scraper."""
from datetime import datetime

from loguru import logger

from config_manager import ConfigManager
from logging_config import setup_logging
from scraper import EURLexScraper


def test_scraper():
    """Test the scraper with January 17th, 2025."""
    # Setup logging
    setup_logging()
    logger.info("Testing scraper...")
    
    try:
        # Initialize scraper
        scraper = EURLexScraper()
        
        # Test date (January 17th, 2025)
        test_date = datetime(2025, 1, 16)
        
        logger.info(f"Scraping journal for date: {test_date.date()}")
        stored_paths = scraper.scrape_journal(test_date)
        
        if stored_paths:
            logger.success(f"Successfully scraped {len(stored_paths)} documents")
            for path in stored_paths:
                logger.info(f"Stored document at: {path}")
        else:
            logger.warning("No documents found/scraped")
        
        logger.success("Scraper test completed successfully")
        
    except Exception as e:
        logger.error(f"Scraper test failed: {str(e)}")
        raise


if __name__ == "__main__":
    test_scraper()
