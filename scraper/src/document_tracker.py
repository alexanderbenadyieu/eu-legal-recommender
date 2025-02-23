"""
Document Tracking Module for EUR-Lex Web Scraper

This module provides a robust mechanism for tracking and preventing 
duplicate document processing during web scraping. It maintains a 
set of processed document identifiers and offers methods to check 
and update the processing status.

Key Features:
- Recursive document discovery in data directories
- Efficient duplicate prevention
- Logging of document tracking activities
- Error-tolerant document loading
"""

import json
from pathlib import Path
from typing import Set
from loguru import logger


class DocumentTracker:
    """
    A comprehensive document tracking system for preventing duplicate processing.

    Manages the tracking of processed documents by maintaining a set of 
    processed CELEX numbers. Automatically discovers and loads existing 
    documents from the specified data directory.

    Attributes:
        data_dir (Path): Base directory for storing scraped documents
        processed_celex (Set[str]): Set of processed document CELEX numbers

    Notes:
        - Supports incremental scraping
        - Prevents re-processing of existing documents
        - Handles potential file reading errors gracefully
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the document tracking system.

        Sets up the document tracker by specifying the base data directory 
        and automatically loading existing processed documents.

        Args:
            data_dir (str): Path to the base directory containing scraped documents
                            This directory will be recursively searched for existing documents

        Notes:
            - Converts input path to a Path object for flexible path handling
            - Automatically calls _load_existing_documents() during initialization
            - Logs the number of existing documents loaded
        """
        self.data_dir = Path(data_dir)
        self.processed_celex: Set[str] = set()
        self._load_existing_documents()
    
    def _load_existing_documents(self):
        """
        Discover and load existing processed documents from the data directory.

        Recursively searches the data directory for JSON files, extracts 
        their CELEX numbers, and adds them to the processed documents set.

        Behavior:
            - Searches all subdirectories for .json files
            - Extracts CELEX numbers from document metadata
            - Handles potential file reading errors
            - Logs the total number of documents loaded

        Notes:
            - Uses rglob for recursive file searching
            - Supports nested directory structures
            - Provides error logging for problematic files
        """
        logger.info(f"Loading existing documents from {self.data_dir}")
        
        for json_file in self.data_dir.rglob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'metadata' in data and 'celex_number' in data['metadata']:
                        self.processed_celex.add(data['metadata']['celex_number'])
            except Exception as e:
                logger.error(f"Error processing {json_file}: {str(e)}")
        
        logger.info(f"Loaded {len(self.processed_celex)} existing documents")
    
    def is_processed(self, celex_number: str) -> bool:
        """
        Check if a document has already been processed.

        Determines whether a document with the given CELEX number 
        has been previously scraped and stored.

        Args:
            celex_number (str): The CELEX number to check for processing status

        Returns:
            bool: True if the document has been processed, False otherwise

        Notes:
            - Provides a constant-time lookup using a set
            - Case-sensitive comparison
        """
        return celex_number in self.processed_celex
    
    def mark_processed(self, celex_number: str):
        """
        Mark a document as processed by adding its CELEX number to the tracking set.

        Adds the specified CELEX number to the set of processed documents, 
        preventing future re-processing of the same document.

        Args:
            celex_number (str): The CELEX number of the document to mark as processed

        Notes:
            - Idempotent operation (calling multiple times has no additional effect)
            - Supports tracking of newly scraped documents
        """
        self.processed_celex.add(celex_number)
    
    def get_processed_count(self) -> int:
        """
        Get the number of processed documents.

        Returns:
            int: Number of processed documents
        """
        return len(self.processed_celex)
