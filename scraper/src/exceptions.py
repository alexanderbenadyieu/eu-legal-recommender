class EURLexScraperError(Exception):
    """Base exception for EUR-Lex scraper."""
    pass


class ConfigurationError(EURLexScraperError):
    """Raised when there's an error in configuration."""
    pass


class ScrapingError(EURLexScraperError):
    """Raised when there's an error during scraping."""
    pass


class ValidationError(EURLexScraperError):
    """Raised when document validation fails."""
    pass


class StorageError(EURLexScraperError):
    """Raised when there's an error storing data."""
    pass


class RateLimitError(ScrapingError):
    """Raised when rate limit is exceeded."""
    pass


class ParseError(ScrapingError):
    """Raised when parsing HTML content fails."""
    pass


class NetworkError(ScrapingError):
    """Raised when network-related errors occur."""
    pass


class InvalidDateError(EURLexScraperError):
    """Exception raised when trying to scrape documents before October 2nd, 2023.
    
    The EUR-Lex website changed its structure on October 2nd, 2023, making older
    documents inaccessible through the current scraping method.
    """
    pass
