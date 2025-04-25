# EU Legal Recommender System Style Guide

This document outlines the coding style guidelines for the EU Legal Recommender System project to ensure consistency across the codebase.

## Import Style

Imports should be organized in the following order, with a blank line between each group:

1. Standard library imports
2. Third-party library imports
3. Local application imports

```python
# Standard library imports
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

# Third-party imports
import numpy as np
import pinecone
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel

# Local application imports
from src.config import EMBEDDER, PINECONE
from src.models.embeddings import BERTEmbedder
from src.utils.db_connector import get_connector
```

## Code Formatting

- Use 4 spaces for indentation (no tabs)
- Maximum line length of 100 characters
- Use snake_case for variables and function names
- Use CamelCase for class names
- Use UPPER_CASE for constants

## Documentation

- All modules, classes, and functions should have docstrings
- Use Google-style docstrings format:

```python
def function_name(param1: type, param2: type) -> return_type:
    """Short description of the function.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: When and why this exception is raised
    """
```

## Error Handling

- Use specific exception types rather than catching all exceptions
- Include meaningful error messages
- Log exceptions with appropriate log levels

```python
try:
    # Code that might raise an exception
    result = some_function()
except SpecificException as e:
    logger.error(f"Specific error occurred: {e}")
    # Handle the error appropriately
except AnotherException as e:
    logger.warning(f"Another error occurred: {e}")
    # Handle differently
```

## Logging

- Use the built-in logging module
- Use appropriate log levels:
  - DEBUG: Detailed information for debugging
  - INFO: Confirmation that things are working as expected
  - WARNING: Something unexpected happened, but the application still works
  - ERROR: The application has failed to perform some function
  - CRITICAL: The application is about to crash

```python
import logging

logger = logging.getLogger(__name__)

logger.debug("Detailed information for debugging")
logger.info("The function completed successfully")
logger.warning("This might cause a problem")
logger.error("The function failed to complete")
logger.critical("The application is about to crash")
```

## Testing

- Write unit tests for all new functionality
- Test both positive and negative cases
- Use descriptive test names that explain what is being tested

## Git Commit Messages

- Use present tense ("Add feature" not "Added feature")
- First line should be 50 characters or less
- Include a more detailed description if necessary after a blank line
- Reference issue numbers if applicable

Example:
```
Add user profile serialization methods

- Add to_dict method to convert profiles to JSON
- Add from_dict method to create profiles from JSON
- Update tests to verify serialization

Fixes #123
```
