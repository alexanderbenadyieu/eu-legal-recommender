# Summarization Utilities

This directory contains utility modules that support the EU legal document summarization system. These utilities handle various aspects of document processing, text manipulation, and database interactions.

## Module Overview

### Core Utilities

- **tier_utils.py**: Common utilities shared between tier3 and tier4 document processing. Contains data structures and functions for section handling, chunk processing, and extraction refinement.

- **tier3_utils.py**: Specialized utilities for processing Tier 3 documents (5,000-20,000 words). Uses common functionality from tier_utils.py.

- **tier4_utils.py**: Specialized utilities for processing Tier 4 documents (20,000-68,000 words). Uses common functionality from tier_utils.py.

### Text Processing Utilities

- **text_chunking.py**: Provides functions for chunking text in a way that respects document structure, using natural boundaries like paragraphs and sentences.

- **summarisation_utils.py**: Helper functions for text processing and summarization, including text cleaning and word counting.

### Database Utilities

- **database_utils.py**: Functions for interacting with the document database, including loading and saving documents and sections.

- **add_word_counts.py**: Script to calculate and add word counts to documents and sections in the database.

## Module Dependencies

```
tier_utils.py ◄─── tier3_utils.py
    ▲
    │
    └────── tier4_utils.py

text_chunking.py ◄─── tier_utils.py

database_utils.py
summarisation_utils.py
add_word_counts.py
```

## Common Data Structures

- **Section**: Represents a document section with ID, title, content, type, order, and word count.
- **ProcessedChunk**: Represents a processed chunk of text with its extraction weight and original section ID.

## Usage Example

```python
from summarization.src.utils.tier_utils import Section, ProcessedChunk
from summarization.src.utils.tier3_utils import process_section

# Create a section
section = Section(
    id=1,
    title="Introduction",
    content="This is the content of the introduction section...",
    section_type="introduction",
    section_order=1,
    word_count=150
)

# Process the section
processed_chunks = process_section(section, target_length=100)
```

## Maintenance Notes

When modifying these utilities:

1. Common functionality should be placed in tier_utils.py
2. Tier-specific functionality should remain in the respective tier files
3. Ensure consistent parameter naming across all modules
4. Update docstrings when modifying function signatures
