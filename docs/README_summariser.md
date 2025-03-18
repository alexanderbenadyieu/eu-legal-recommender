# EU Legal Document Summarization

This module provides tools for preprocessing, summarizing, and analyzing EU legal documents. It is part of the larger EU Legal Recommender system.

## Overview

The summarization module processes EU legal documents to generate concise summaries and extract keywords, which are then used by the recommender system to provide relevant document recommendations.

## Structure

- **config/**: Configuration files for the summarization pipeline
  - `summarisation_config.yaml`: Main configuration file

- **data/**: Database and processed data
  - `processed_documents.db`: SQLite database containing processed documents
  - `data_models.sql`: SQL schema for the database

- **src/**: Source code
  - **abstractive/**: Abstractive summarization components
    - `bart_finetuner.py`: Fine-tuning BART models for abstractive summarization
  
  - **extractive/**: Extractive summarization components
    - `lexlm_wrapper.py`: Wrapper for LexLM extractive summarization

  - **postprocessing/**: Post-processing utilities
    - `deepseek_processor.py`: Processing summaries with DeepSeek models
    - `process_existing_summaries.py`: Utilities for processing existing summaries

  - **preprocessing/**: Pre-processing utilities
    - `clean_and_reprocess.py`: Cleaning and reprocessing documents
    - `html_parser.py`: Parsing HTML documents
    - `process_documents.py`: Processing document text

  - **testing/**: Test files
    - Various test scripts for different components

  - **utils/**: Utility functions
    - `add_word_counts.py`: Adding word counts to documents
    - `database_utils.py`: Database utilities
    - `summarisation_utils.py`: Summarization utilities
    - `text_chunking.py`: Text chunking utilities
    - `tier3_utils.py`: Utilities for tier 3 documents
    - `tier4_utils.py`: Utilities for tier 4 documents

  - Core files:
    - `keyword_extractor.py`: Extracting keywords from documents
    - `pipeline.py`: Main pipeline for document processing
    - `run_pipeline.py`: Script to run the pipeline
    - `summariser.py`: Main summarization logic

## Usage

The summarization pipeline can be run using the `run_pipeline.py` script:

```bash
python src/run_pipeline.py --config config/summarisation_config.yaml
```

## Database Structure

The module supports both legacy and consolidated database structures:

- **Legacy**: Uses `processed_documents.db` in the summarization/data directory
- **Consolidated**: Uses `eurlex.db` in the scraper/data directory

## Dependencies

See `requirements.txt` for a list of dependencies.
