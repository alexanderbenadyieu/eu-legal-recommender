# EU Legal Recommender Project Structure

This document outlines the standardized project structure for the EU Legal Recommender system.

## Directory Structure

```
recommender/
├── src/                           # Main package code
│   ├── __init__.py                # Package initialization
│   ├── config.py                  # Configuration management
│   ├── models/                    # Core model implementations
│   │   ├── __init__.py
│   │   ├── embeddings.py          # Embedding generation
│   │   ├── features.py            # Feature processing
│   │   ├── pinecone_recommender.py
│   │   ├── personalized_recommender.py
│   │   ├── ranker.py              # Result ranking
│   │   ├── similarity.py          # Similarity calculations
│   │   └── user_profile.py        # User profile management
│   ├── utils/                     # Utility functions
│   │   ├── __init__.py
│   │   ├── db_connector.py        # Database connection utilities
│   │   ├── logging.py             # Logging configuration
│   │   └── data_processing.py     # Data processing utilities
│   └── api/                       # API implementation (if applicable)
│       ├── __init__.py
│       ├── routes.py              # API routes
│       └── schemas.py             # API data schemas
├── scripts/                       # Utility scripts
│   ├── index_documents.py         # Document indexing
│   ├── recreate_embeddings.py     # Recreate embeddings
│   ├── run_benchmarks.py          # Performance benchmarking
│   └── setup_environment.py       # Environment setup
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── conftest.py                # Test fixtures
│   ├── run_tests.py               # Test runner
│   ├── test_embeddings.py         # Tests for embedding generation
│   ├── test_pinecone_recommender.py # Tests for recommender
│   ├── test_personalized_recommender.py # Tests for personalized recommender
│   └── test_user_profile.py       # Tests for user profiles
├── examples/                      # Example scripts
│   ├── README.md                  # Examples documentation
│   ├── basic/                     # Basic usage examples
│   ├── personalized/              # Personalized recommendations examples
│   └── profiles/                  # User profile examples
├── docs/                          # Documentation
│   ├── api/                       # API documentation
│   ├── guides/                    # User guides
│   └── development/               # Development documentation
├── cli.py                         # Command-line interface
├── pyproject.toml                 # Project metadata and dependencies
├── requirements.txt               # Dependencies
├── .coveragerc                    # Coverage configuration
├── STYLE_GUIDE.md                 # Coding style guide
├── README.md                      # Project overview
└── .env                           # Environment variables (gitignored)
```

## Import Structure

All imports should follow the structure defined in the STYLE_GUIDE.md:

```python
# Standard library imports
import os
import sys
from typing import Dict, List, Optional

# Third-party imports
import numpy as np
import pinecone
from sentence_transformers import SentenceTransformer

# Local imports
from src.models.embeddings import BERTEmbedder
from src.utils.db_connector import PineconeConnector
```

## Module Organization

Each module should have a clear responsibility and follow the single responsibility principle:

- `models/`: Core algorithm implementations
- `utils/`: Helper functions and utilities
- `api/`: API endpoints and schemas
- `scripts/`: Standalone scripts for maintenance tasks

## Configuration Management

Configuration should be centralized in the `config.py` file, with support for:

- Environment variables (via python-dotenv)
- Configuration files
- Default values

## Documentation

Each module, class, and function should have comprehensive docstrings following the Google style:

```python
def function_name(param1: type, param2: type) -> return_type:
    """Short description of the function.
    
    Longer description explaining the function's purpose and behavior.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: When and why this exception is raised
    """
```

## Testing

Tests should be organized by type:

- `unit/`: Tests for individual components in isolation
- `integration/`: Tests for interactions between components

Each test file should correspond to a module in the main codebase.
