"""
Document Storage Management Module for EUR-Lex Web Scraper

A robust and flexible storage management system for organizing 
and persisting scraped EUR-Lex legislative documents. Provides 
comprehensive document storage capabilities with advanced 
directory management and metadata validation.

Key Features:
- Hierarchical document storage
- Automatic directory creation
- Metadata validation
- Flexible file naming
- Error handling and logging

Storage Strategy:
- Organized by year/month/journal/document
- JSON-based document storage
- Configurable base directory
- Supports incremental and batch document storage

Technologies:
- Python Pathlib for file and directory management
- JSON for document serialization
- Loguru for logging
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from exceptions import StorageError
from validation import validate_metadata


class StorageManager:
    """
    Advanced document storage management system for EUR-Lex web scraper.

    Manages the complete lifecycle of document storage, including 
    directory creation, file naming, and document persistence.

    Core Responsibilities:
    - Generate document storage paths
    - Ensure directory existence
    - Validate and store document metadata
    - Handle storage-related errors
    - Provide flexible storage configuration

    Attributes:
        base_dir (Path): Base directory for document storage
        
    Raises:
        StorageError: For directory creation or file storage failures

    Notes:
        - Supports hierarchical document organization
        - Provides robust error handling
        - Ensures consistent storage structure
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize the document storage management system.

        Sets up the base directory for document storage and ensures 
        its existence with comprehensive error handling.

        Args:
            base_dir (str): Base directory path for storing documents

        Notes:
            - Converts input to Path object
            - Creates base directory if it doesn't exist
            - Logs and raises errors for directory creation failures
        """
        self.base_dir = Path(base_dir)
        self._ensure_directory_exists(self.base_dir)
    
    def _ensure_directory_exists(self, directory: Path) -> None:
        """
        Create a directory if it does not already exist.

        Attempts to create the specified directory with parent directories, 
        with robust error handling and logging.

        Args:
            directory (Path): Directory path to create

        Raises:
            StorageError: If directory creation fails

        Notes:
            - Uses mkdir with parents=True for nested directory creation
            - Provides detailed error logging
            - Ensures directory availability before storage operations
        """
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {str(e)}")
            raise StorageError(f"Failed to create directory {directory}") from e
    
    def _get_document_path(self, date: datetime, journal_id: str, doc_id: str) -> Path:
        """
        Generate a hierarchical file path for a document.

        Creates a structured file path based on document metadata, 
        supporting organized and predictable document storage.

        Args:
            date (datetime): Date of the document
            journal_id (str): Identifier of the source journal
            doc_id (str): Unique document identifier

        Returns:
            Path: Complete file path for document storage

        Notes:
            - Supports year/month/journal/document hierarchy
            - Ensures consistent path generation
            - Facilitates easy document retrieval
        """
        # Create path: year/month/journal/document.json
        year_dir = self.base_dir / str(date.year)
        month_dir = year_dir / f"{date.month:02d}"
        journal_dir = month_dir / journal_id
        
        # Ensure all directories exist
        self._ensure_directory_exists(journal_dir)
        
        return journal_dir / f"{doc_id}.json"
    
    def store_document(self, 
                      document: Dict[str, Any],
                      date: datetime,
                      journal_id: str,
                      doc_id: str) -> Path:
        """
        Store a document as JSON.

        Performs comprehensive document storage, including:
        - Metadata validation
        - Path generation
        - File writing
        - Error handling

        Args:
            document (Dict[str, Any]): Document content and metadata
            date (datetime): Date of the document
            journal_id (str): Identifier of the source journal
            doc_id (str): Unique document identifier

        Returns:
            Path: Path to the stored document file

        Raises:
            StorageError: For metadata validation or file writing failures
            ValidationError: For invalid metadata

        Notes:
            - Validates metadata before storage
            - Supports JSON document storage
            - Provides comprehensive error handling
            - Logs storage activities
        """
        try:
            # Validate metadata if present
            if 'metadata' in document and document['metadata']:
                validate_metadata(document['metadata'])
            
            # Convert document to JSON-serializable format
            json_doc = {
                'metadata': document.get('metadata'),
                'content': document['full_text']
            }
            
            # Get storage path
            file_path = self._get_document_path(date, journal_id, doc_id)
            
            # Store document
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_doc, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Stored document {doc_id} at {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to store document {doc_id}: {str(e)}")
            raise StorageError(f"Failed to store document {doc_id}") from e
    
    def document_exists(self, date: datetime, journal_id: str, doc_id: str) -> bool:
        """
        Check if a document already exists.

        Verifies the existence of a document based on its metadata.

        Args:
            date (datetime): Date of the document
            journal_id (str): Identifier of the source journal
            doc_id (str): Unique document identifier

        Returns:
            bool: True if document exists

        Notes:
            - Supports flexible document existence checks
            - Provides efficient existence verification
        """
        file_path = self._get_document_path(date, journal_id, doc_id)
        return file_path.exists()
    
    def load_document(self, 
                     date: datetime,
                     journal_id: str,
                     doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a stored document.

        Retrieves a document from the storage system using its metadata.

        Args:
            date (datetime): Date of the document
            journal_id (str): Identifier of the source journal
            doc_id (str): Unique document identifier

        Returns:
            Optional[Dict[str, Any]]: Loaded document dictionary, 
                                      or None if document not found

        Notes:
            - Supports flexible document retrieval
            - Handles non-existent documents gracefully
            - Provides optional error logging
        """
        try:
            file_path = self._get_document_path(date, journal_id, doc_id)
            if not file_path.exists():
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Failed to load document {doc_id}: {str(e)}")
            return None
