# EU Legal Document Summarization

This module provides tools for preprocessing, summarizing, and analyzing EU legal documents. It is part of the larger EU Legal Recommender system.

## Overview

The summarization module processes EU legal documents to generate concise summaries and extract keywords, which are then used by the recommender system to provide relevant document recommendations. The module implements a multi-tier summarization strategy based on document length:

- **Tier 1 (0-600 words)**: Direct abstractive summarization
- **Tier 2 (601-2,500 words)**: Two-step summarization with extractive then abstractive
- **Tier 3 (2,501-20,000 words)**: Hierarchical summarization with section-aware processing
- **Tier 4 (20,000+ words)**: Advanced hierarchical summarization with weighted extraction

## Structure

- **config/**: Configuration files for the summarization pipeline
  - `summarisation_config.yaml`: Main configuration file

- **data/**: Database and processed data
  - `processed_documents.db`: SQLite database containing processed documents
  - `data_models.sql`: SQL schema for the database

- **src/**: Source code
  - **abstractive/**: Abstractive summarization components
    - `bart_finetuner.py`: Fine-tuned BART models for abstractive summarization
  
  - **extractive/**: Extractive summarization components
    - `lexlm_wrapper.py`: Wrapper for LexLM extractive summarization

  - **postprocessing/**: Post-processing utilities
    - `deepseek_processor.py`: Processing summaries with DeepSeek models
    - `process_existing_summaries.py`: Utilities for processing existing summaries

  - **preprocessing/**: Pre-processing utilities
    - `clean_and_reprocess.py`: Cleaning and reprocessing documents
    - `html_parser.py`: Parsing HTML documents for structured content extraction
    - `process_documents.py`: Processing document text

  - **utils/**: Utility functions
    - `add_word_counts.py`: Adding word counts to documents
    - `config.py`: Centralized configuration management system
    - `database_utils.py`: Database utilities for document storage and retrieval
    - `summarisation_utils.py`: General summarization utilities
    - `text_chunking.py`: Text chunking utilities for handling large documents
    - `tier_processing.py`: Consolidated tier-specific processing utilities

  - Core files:
    - `keyword_extractor.py`: Extracting keywords from documents using KeyBERT
    - `pipeline.py`: Main pipeline implementing the multi-tier summarization strategy

- **run_summarization.py**: Command-line interface to run the summarization pipeline

## Usage

### Installation

1. Clone the repository:
```bash
git clone https://github.com/alexanderbenadyieu/eu-legal-recommender
cd eu-legal-recommender/summarization
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

### Running the Pipeline

The summarization pipeline can be run using the `run_summarization.py` script:

```bash
# Run with default settings
python run_summarization.py

# Run with specific tier
python run_summarization.py --tier 2

# Run with custom configuration file
python run_summarization.py --config path/to/custom_config.yaml

# Run with specific database type
python run_summarization.py --db-type consolidated
```

### Configuration

The summarization module uses a centralized configuration system that supports environment variables for better flexibility and portability. The configuration can be customized in several ways:

1. **Environment Variables**: Set environment variables to override default settings
   ```bash
   export SUMMARIZATION_DATA_DIR=/path/to/data
   export SUMMARIZATION_MODEL_DIR=/path/to/models
   ```

2. **Configuration File**: Provide a custom configuration file
   ```bash
   python run_summarization.py --config path/to/custom_config.yaml
   ```

3. **Default Configuration**: If no custom configuration is provided, the system will use the default configuration from `config/summarisation_config.yaml`


## Multi-Tier Summarization Approach

The EU Legal Recommender system uses a sophisticated multi-tier summarization approach that adapts to document length and complexity. This approach ensures optimal summarization quality across a wide range of document sizes, from short regulations to lengthy directives.

### Tier 1: Direct Abstractive Summarization (0-600 words)

- **Process**: Single-step abstractive summarization using a fine-tuned BART model
- **Compression Ratio**: Adaptive, based on document length
- **Use Case**: Short documents like decisions and brief regulations

### Tier 2: Two-Step Summarization (601-2,500 words)

- **Process**:
  1. Split text into chunks that fit within LexLM context (514 tokens)
  2. Extract K words where K = max(300, min(0.3 × D, 600))
  3. Generate final summary of 0.6K to 0.8K words
- **Use Case**: Medium-length documents like short directives and regulations

### Tier 3: Hierarchical Summarization (2,501-20,000 words)

- **Process**:
  1. Process each section to target length (~350 words)
  2. Apply weighted compression to reach intermediate target (600-750 words)
  3. Generate final abstractive summary (480-600 words)
- **Use Case**: Longer documents like detailed directives and regulations

### Tier 4: Advanced Hierarchical Summarization (20,000+ words)

- **Process**:
  1. Section-Based Pre-Summarization
     - Sections <750 words: Summarize to ≤350 words using BART
     - Sections 750-3000 words: Apply Tier 2 approach
     - Sections >3000 words: Apply Tier 3 approach
  2. Weighted Extraction with U-shaped importance distribution
  3. Final Abstractive Summarization (480-600 words)
- **Use Case**: Very long documents like comprehensive regulations and international agreements

## Dependencies

The summarization module relies on the following key dependencies:

- **transformers**: For BART and LexLM models
- **nltk**: For text tokenization and processing
- **keybert**: For keyword extraction
- **beautifulsoup4**: For HTML parsing
- **pyyaml**: For configuration management

See `requirements.txt` for a complete list of dependencies.
