# EUR-Lex Web Scraper

A web scraper for extracting legislative documents from EUR-Lex, focusing on the Official Journal L series (Legislation). The scraper captures comprehensive metadata, including document identifiers, directory codes, and full-text content.

## Features

- **Automated Document Scraping**
  - Supports date range-based scraping
  - Handles multiple document types
  - Automatic retry mechanism for failed requests
  - Comprehensive error handling and recovery
  
- **Rich Metadata Extraction**
  - Document identifiers and CELEX numbers
  - Multiple directory codes and descriptions
  - Publication dates and legal dates
  - Authors and responsible bodies
  - EuroVoc descriptors and subject matters

- **Efficient Storage**
  - Organized hierarchical storage structure (year/month/day)
  - JSON format for easy data processing
  - Automatic directory creation

## System Architecture

### Core Components

1. **Scraper Core (`scraper.py`)**
   - Manages the scraping workflow
   - Implements rate limiting and retry logic
   - Coordinates between parsers, storage, and metrics

2. **Parsers (`parsers.py`)**
   - `MetadataParser`: Extracts structured metadata from document pages
   - `DocumentParser`: Processes document content and formats
   - Handles complex HTML structures using BeautifulSoup
   - Implements robust error handling for malformed content

3. **Storage Manager (`storage.py`)**
   - Implements hierarchical storage (year/month/day)
   - Handles file operations and path management
   - Ensures atomic writes to prevent data corruption

4. **Document Tracker (`document_tracker.py`)**
   - Manages document deduplication
   - Tracks already processed documents
   - Prevents re-scraping of existing documents

5. **Deduplicator (`deduplicator.py`)**
   - Advanced document deduplication logic
   - Compares document metadata to identify unique entries
   - Prevents storage of duplicate or redundant documents

6. **Validation (`validation.py`)**
   - Validates document metadata against a predefined schema
   - Flexible identifier validation
     - Identifier is now an optional field
     - Allows documents without a standard identifier format
   - Ensures data integrity and consistency

7. **Metrics Collection (`metrics.py`)**
   - Prometheus integration for monitoring
   - Tracks key performance indicators:
     - Document processing rates
     - Success/failure counts
     - Processing times
     - Request statistics
   - Supports both file-based and HTTP exports

8. **Configuration Management (`config_manager.py`)**
   - YAML-based configuration
   - Environment variable support
   - Runtime configuration validation

### Data Structures

#### Document Content
```python
class DocumentContent:
    full_text: str            # Complete document text
    html_url: str             # Source HTML URL
    pdf_url: str              # PDF version URL
    metadata: DocumentMetadata # Associated metadata
```

#### Document Metadata
```python
@dataclass
class DocumentMetadata:
    celex_number: str          # Unique document identifier
    title: str                 # Document title
    identifier: str            # Document reference number
    eli_uri: str              # European Legislation Identifier
    adoption_date: datetime    # Date of adoption
    effect_date: datetime      # Date when document takes effect
    end_validity_date: datetime # End of validity date
    authors: List[str]         # Document authors/institutions
    form: str                  # Document form/type
    eurovoc_descriptors: List[str]    # EuroVoc classification
    subject_matters: List[str]        # Subject categories
    directory_codes: List[str]        # Classification codes
    directory_descriptions: List[str]  # Code descriptions
`````

## Quick Start

### Prerequisites

- Python 3.9+
- `pip install -r requirements.txt`

### Running the Scraper

```bash
# Basic usage
python src/main.py

# Specify date range
python src/main.py --start-date 2023-10-01 --end-date 2024-01-31

# Configuration options in config/config.yaml
```

### Development

- Create virtual environment: `python -m venv venv`
- Activate: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Run tests: `python -m pytest src/test/`

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd eurlex-scraper
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
eurlex-scraper/
├── src/                # Source code
│   ├── __init__.py
│   ├── main.py        # Main entry point
│   ├── scraper.py     # Core scraping logic
│   ├── parsers.py     # HTML and metadata parsers
│   ├── storage.py     # Data storage handlers
│   ├── document_tracker.py # Document tracking
│   ├── deduplicator.py # Document deduplication
│   ├── validation.py  # Metadata validation
│   └── metrics.py     # Metrics collection
├── config/            # Configuration files
│   └── config.yaml    # Main configuration
├── data/             # Data storage
│   └── YYYY/MM/DD/   # Hierarchical document storage
├── logs/            # Log files
└── tests/           # Unit and integration tests
```
### Command Line Arguments

- `--start-date`: Start date for scraping (YYYY-MM-DD)
- `--end-date`: End date for scraping (YYYY-MM-DD)
- `--config`: Path to custom config file (optional)
- `--log-level`: Set logging level (optional)

## Output Format

Documents are stored as JSON files with the following structure:

```json
{
  "metadata": {
    "celex_number": "32025R0040",
    "title": "Document Title",
    "identifier": "Document Identifier",
    "directory_codes": ["13.30.99.00", "15.10.30.30"],
    "directory_descriptions": [
      "Description 1",
      "Description 2"
    ],
    "dates": {
      "Date of document": "YYYY-MM-DD",
      "Date of effect": "YYYY-MM-DD"
    }
  }
}
```

## Error Handling

The scraper implements several error handling mechanisms:

1. **Request Retries**
   - Exponential backoff with configurable delays
   - Maximum retry attempts limit
   - Specific handling for different HTTP status codes

2. **Rate Limiting**
   - Token bucket algorithm implementation
   - Configurable rate limits
   - Automatic rate adjustment based on server response

3. **Recovery Mechanisms**
   - Checkpoint system for interrupted sessions
   - Transaction-like storage operations
   - Automatic cleanup of partial downloads

4. **Logging and Monitoring**
   - Structured logging with rotation
   - Prometheus metrics for monitoring
   - Alert conditions for critical failures

## Limitations

### Date Range Restriction
The scraper only works for documents published on or after October 2nd, 2023. This limitation exists because the EUR-Lex website underwent structural changes on this date
- Documents before this date use a different URL format and page structure
- Attempting to scrape earlier dates will result in an `InvalidDateError`

```bash
$ python src/main.py --start-date 2023-09-15 --end-date 2023-09-15
ERROR    Cannot scrape dates before October 2nd, 2023 due to website structure changes. Provided start date: 2023-09-15
```

### Other Limitations
- Only scrapes the Official Journal L series, not adapted to C series
- Limited to documents in English - it can be easily adapted to other languages with URL handling
- Some document types may have incomplete metadata
