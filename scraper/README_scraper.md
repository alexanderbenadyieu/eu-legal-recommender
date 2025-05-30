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
  - Full HTML content preservation
  - Raw and processed text content

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
     - Preserves complete HTML structure
     - Extracts clean text content
     - Maintains document formatting
   - Handles complex HTML structures using BeautifulSoup
   - Implements robust error handling for malformed content

3. **Storage Manager (`storage.py`)**
   - Implements hierarchical storage (year/month/day)
   - Handles file operations and path management
   - Ensures atomic writes to prevent data corruption

4. **Document Tracker (`document_tracker.py`)**
   - Manages document tracking and deduplication
   - Tracks already processed documents
   - Prevents re-scraping of existing documents
   - Identifies and resolves duplicate documents
   - Supports post-processing cleanup of duplicates

5. **Validation (`validation.py`)**
   - Validates document metadata against a predefined schema
   - Flexible identifier validation
     - Identifier is now an optional field
     - Allows documents without a standard identifier format
   - Ensures data integrity and consistency

6. **Metrics Collection (`metrics.py`)**
   - Prometheus integration for monitoring
   - Tracks key performance indicators:
     - Document processing rates
     - Success/failure counts
     - Processing times
     - Request statistics
   - Supports both file-based and HTTP exports

7. **Configuration Management (`config_manager.py`)**
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
git clone https://github.com/alexanderbenadyieu/eu-legal-recommender
cd eu-legal-recommender/scraper
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
scraper/
├── src/                # Source code
│   ├── __init__.py
│   ├── main.py        # Main entry point
│   ├── scraper.py     # Core scraping logic
│   ├── parsers.py     # HTML and metadata parsers
│   ├── storage.py     # Data storage handlers
│   ├── document_tracker.py # Document tracking and deduplication
│   ├── deduplicate.py # Command-line interface for deduplication
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
    "celex_number": "Unique EU document identifier",
    "title": "Full official title of the document",
    "identifier": "Internal document identifier",
    "eli_uri": "European Legislation Identifier URI",
    "html_url": "URL to HTML version of the document",
    "pdf_url": "URL to PDF version of the document",
    "dates": {
      "Date of document": "Date when document was created",
      "Date of effect": "Date when document becomes active",
      "Date of end of validity": "Date when document expires"
    },
    "authors": ["List of document authors/originating bodies"],
    "responsible_body": "Primary responsible administrative body",
    "form": "Type of legal instrument",
    "eurovoc_descriptors": ["Controlled vocabulary terms describing the document"],
    "subject_matters": ["Detailed subject classification"],
    "directory_codes": ["Hierarchical classification codes"],
    "directory_descriptions": ["Descriptions of classification codes"]
  },
  "content": "Full text of the document as a string",
  "htmlcontent": "Full text of the document as a string"
}
```

## Error Handling

The scraper implements several error handling mechanisms:

1. **Request Retries**
   - Exponential backoff with configurable delays
   - Maximum retry attempts limit
   - Specific handling for different HTTP status codes

2. **Rate Limiting**
   - Configurable request timeouts
   - Automatic delay between requests
   - Adaptive retry mechanisms

3. **Recovery Mechanisms**
   - Transaction-like storage operations
   - Automatic cleanup of partial downloads
   - Robust error handling for network failures

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
