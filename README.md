# EUR-Lex Web Scraper

A robust web scraper for extracting legislative documents from EUR-Lex, focusing on the Official Journal L series (Legislation). The scraper captures comprehensive metadata, including document identifiers, directory codes, and full-text content.

## Features

- **Automated Document Scraping**
  - Scrapes legislative documents from EUR-Lex's Official Journal L series
  - Supports date range-based scraping
  - Handles multiple document types and formats
  
- **Rich Metadata Extraction**
  - Document identifiers and CELEX numbers
  - Multiple directory codes and descriptions
  - Publication dates and legal dates
  - Authors and responsible bodies
  - EuroVoc descriptors and subject matters
  
- **Robust Architecture**
  - Automatic retry mechanism for failed requests
  - Rate limiting to respect server constraints
  - Comprehensive error handling and recovery
  - Detailed logging with rotation
  - Prometheus metrics collection

- **Efficient Storage**
  - Organized hierarchical storage structure (year/month/day)
  - JSON format for easy data processing
  - Deduplication of existing documents
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
```

#### Document Content
```python
class DocumentContent:
    full_text: str            # Complete document text
    html_url: str             # Source HTML URL
    pdf_url: str              # PDF version URL
    metadata: DocumentMetadata # Associated metadata
```

## Recent Updates

- Enhanced CELEX number validation to support more document formats
- Improved metadata schema flexibility
- Added comprehensive `.gitignore` for better version control

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

## Usage

### Basic Usage

Run the scraper for a specific date range:
```bash
python src/main.py --start-date 2025-01-01 --end-date 2025-01-31
```

### Command Line Arguments

- `--start-date`: Start date for scraping (YYYY-MM-DD)
- `--end-date`: End date for scraping (YYYY-MM-DD)
- `--config`: Path to custom config file (optional)
- `--log-level`: Set logging level (optional)

## Configuration

The scraper can be configured through `config/config.yaml`:

```yaml
scraping:
  base_url: 'https://eur-lex.europa.eu'
  request_timeout: 30
  language: 'EN'
  rate_limit: 1.0  # Requests per second
  max_retries: 3
  retry_delay: 5.0

storage:
  base_dir: 'data'
  file_format: 'json'
  backup_enabled: true
  compression: false

metrics:
  enabled: true
  directory: 'metrics'
  export_port: 9090
  collection_interval: 60

logging:
  level: 'INFO'
  rotation: '1 day'
  retention: '30 days'
  compression: 'gz'
```

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

## Performance Considerations

1. **Memory Management**
   - Streaming response handling for large documents
   - Efficient string processing for text content
   - Garbage collection optimization

2. **Storage Efficiency**
   - Compression options for stored documents
   - Deduplication of repeated content
   - Efficient indexing structure

3. **Concurrency**
   - Configurable concurrent downloads
   - Connection pooling
   - Resource usage limits

## Limitations

### Date Range Restriction
The scraper only works for documents published on or after October 2nd, 2023. This limitation exists because:
- The EUR-Lex website underwent structural changes on this date
- Documents before this date use a different URL format and page structure
- Attempting to scrape earlier dates will result in an `InvalidDateError`

Example error when trying to scrape older documents:
```bash
$ python src/main.py --start-date 2023-09-15 --end-date 2023-09-15
ERROR    Cannot scrape dates before October 2nd, 2023 due to website structure changes. Provided start date: 2023-09-15
```

### Other Limitations
- Only scrapes the Official Journal L series
- Limited to documents in English
- Some document types may have incomplete metadata
