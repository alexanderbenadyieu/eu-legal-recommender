# Docstring Templates for EU Legal Recommender

This document provides standardized templates for docstrings to be used throughout the EU Legal Recommender codebase. Following these templates ensures consistency and completeness of documentation.

## Module Docstring Template

```python
"""
[Module name/title].

[Detailed description of what the module does and its purpose within the system.]

This module [provides/implements/defines] [brief description of functionality].
"""
```

## Class Docstring Template

```python
class MyClass:
    """
    [Brief one-line description of the class].
    
    [Detailed description of what the class does, its purpose, and how it fits
    into the overall system architecture. Include any important implementation
    details or design decisions.]
    
    Attributes:
        attribute_name (type): Description of the attribute.
        
    Examples:
        >>> instance = MyClass(param1, param2)
        >>> result = instance.method()
    """
```

## Method/Function Docstring Template

```python
def my_function(param1: type, param2: type = default_value) -> return_type:
    """
    [Brief one-line description of what the function does].
    
    [Detailed description of the function's purpose, behavior, and any
    important implementation details. Include any algorithms used or
    complex logic explained.]
    
    Args:
        param1 (type): Description of the parameter.
        param2 (type, optional): Description of the parameter. Defaults to default_value.
        
    Returns:
        return_type: Description of the return value.
        
    Raises:
        ExceptionType: When and why this exception might be raised.
        
    Examples:
        >>> result = my_function('example', 42)
        >>> print(result)
    """
```

## Property Docstring Template

```python
@property
def my_property(self) -> type:
    """
    [Brief description of the property].
    
    Returns:
        type: Description of the return value.
    """
```

## Error Handling Guidelines

When documenting functions that may raise exceptions:

1. Always include a "Raises" section in the docstring
2. List each specific exception type that might be raised
3. Explain the conditions under which each exception would be raised
4. Use custom exceptions from `src.utils.exceptions` where appropriate

## Logging Guidelines

When implementing logging in a function:

1. Use the logger from `src.utils.logging` module
2. Log at appropriate levels:
   - DEBUG: Detailed information for debugging
   - INFO: Confirmation that things are working as expected
   - WARNING: Indication that something unexpected happened but the application is still working
   - ERROR: Due to a more serious problem, the application has not been able to perform a function
   - CRITICAL: A serious error indicating that the program itself may be unable to continue running

3. Include relevant context in log messages:
   - For user actions, include user identifiers
   - For data processing, include data identifiers or sizes
   - For errors, include exception details

## Example Implementation

```python
from src.utils.logging import get_logger
from src.utils.exceptions import ValidationError, DataError

logger = get_logger(__name__)

def process_document(document_id: str, options: dict = None) -> dict:
    """
    Process a legal document with the specified options.
    
    This function retrieves a document by its ID, applies the specified
    processing options, and returns the processed document data.
    
    Args:
        document_id (str): The unique identifier of the document to process.
        options (dict, optional): Processing options. Defaults to None.
        
    Returns:
        dict: The processed document data.
        
    Raises:
        ValidationError: If document_id is invalid or not found.
        DataError: If there's an error processing the document data.
    """
    logger.info(f"Processing document: {document_id}")
    
    # Validate input
    if not document_id or not document_id.strip():
        logger.error("Empty document ID provided")
        raise ValidationError("Document ID cannot be empty", field="document_id")
        
    try:
        # Processing logic here
        logger.debug(f"Applied options: {options}")
        result = {"id": document_id, "processed": True}
        logger.info(f"Successfully processed document: {document_id}")
        return result
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
        raise DataError(f"Failed to process document: {str(e)}")
```
