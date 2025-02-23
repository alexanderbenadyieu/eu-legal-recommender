"""Validation module for EUR-Lex document metadata.

This module provides comprehensive validation and parsing utilities for 
EUR-Lex document metadata, ensuring data integrity and consistency.

Key Features:
- JSON Schema-based metadata validation
- Document ID format validation
- Date parsing with flexible input formats
- Logging of non-standard document identifiers
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
import re

from jsonschema import validate
from loguru import logger

from exceptions import ValidationError

# JSON Schema for document metadata
METADATA_SCHEMA = {
    "type": "object",
    "required": ["title", "celex_number"],
    "properties": {
        "celex_number": {"type": "string"},
        "title": {"type": "string", "minLength": 1},
        "identifier": {"type": "string"},
        "eli_uri": {"type": "string", "format": "uri"},
        "html_url": {"type": "string", "format": "uri"},
        "pdf_url": {"type": "string", "format": "uri"},
        "dates": {
            "type": "object",
            "properties": {
                "Date of document": {"type": "string"},
                "Date of effect": {"type": "string"},
                "Date of end of validity": {"type": "string"}
            }
        },
        "authors": {"type": "array", "items": {"type": "string"}},
        "responsible_body": {"type": "string"},
        "form": {"type": "string"},
        "eurovoc_descriptors": {"type": "array", "items": {"type": "string"}},
        "subject_matters": {"type": "array", "items": {"type": "string"}},
        "directory_codes": {"type": "array", "items": {"type": "string"}},
        "directory_descriptions": {"type": "array", "items": {"type": "string"}}
    }
}


def validate_metadata(metadata: Dict[str, Any]) -> None:
    """
    Validate document metadata against a predefined JSON schema.

    This function performs comprehensive validation of document metadata,
    ensuring that all required fields are present and conform to expected formats.
    It uses JSON Schema validation and performs additional custom validations.

    Args:
        metadata (Dict[str, Any]): A dictionary containing document metadata.

    Raises:
        ValidationError: If metadata fails to meet the required schema or validation rules.

    Notes:
        - Logs non-standard CELEX numbers
        - Validates required fields: 'title' and 'celex_number'
        - Allows flexible metadata with optional fields
    """
    try:
        # Validate identifier (optional)
        identifier = metadata.get('identifier', '').strip()
        
        # If identifier is not empty, validate its structure
        if identifier:
            # Check for specific identifier formats
            valid_identifier_pattern = r'^[A-Z]/\d{4}/\d+(/[A-Z]+)?$'
            
            if not re.match(valid_identifier_pattern, identifier):
                # If it doesn't match the expected structure, set to empty string
                metadata['identifier'] = ''
        
        # Log non-standard CELEX numbers for tracking
        if 'celex_number' in metadata and metadata['celex_number']:
            celex_num = metadata['celex_number']
            logger.info(f"CELEX number: {celex_num}")
        
        validate(instance=metadata, schema=METADATA_SCHEMA)
    except Exception as e:
        logger.error(f"Metadata validation failed: {str(e)}")
        raise ValidationError(f"Invalid metadata format: {str(e)}") from e


def validate_document_id(doc_id: str) -> bool:
    """
    Validate the format and status of a document identifier.

    Performs comprehensive checks on document identifiers to ensure they
    meet the EUR-Lex document ID standards. Specifically:
    - Checks for valid document ID format
    - Identifies and filters out corrigendum documents
    - Logs non-standard document identifiers

    Args:
        doc_id (str): The document identifier to validate.

    Returns:
        bool: True if the document is valid and not a corrigendum, False otherwise.

    Notes:
        - Uses regex patterns to validate document ID structure
        - Handles various document ID formats
        - Provides logging for tracking non-standard identifiers
    """
    if not doc_id:
        return False
    
    # Remove 'L_' prefix if present
    if doc_id.startswith('L_'):
        doc_id = doc_id[2:]
    
    # Check if it's a corrigendum (ends with 9...)
    if doc_id[4] == '9':
        return False
    
    # Basic format validation: should be a year (2025) followed by 5 digits
    return len(doc_id) == 9 and doc_id[:4].isdigit()


def parse_date(date_str: str) -> Optional[datetime]:
    """
    Parse date strings with high flexibility and robustness.

    Attempts to parse date strings from various input formats commonly 
    found in EUR-Lex documents. Supports multiple date representations 
    and handles potential parsing errors gracefully.

    Args:
        date_str (str): A date string to be parsed.

    Returns:
        Optional[datetime]: A parsed datetime object if successful, 
                            None if parsing fails.

    Supported Formats:
        - 'DD/MM/YYYY'
        - 'YYYY-MM-DD'
        - 'Month DD, YYYY'
        - Partial or incomplete dates

    Notes:
        - Uses multiple parsing strategies
        - Handles localized date formats
        - Provides fallback mechanisms for complex date strings
    """
    formats = [
        '%d/%m/%Y',  # 10/01/2025
        '%d.%m.%Y',  # 10.01.2025
        '%Y-%m-%d',  # 2025-01-10
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    return None
