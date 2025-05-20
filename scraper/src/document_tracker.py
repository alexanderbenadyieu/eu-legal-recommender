"""
Document Tracking and Deduplication Module for EUR-Lex Web Scraper

This module provides a comprehensive system for tracking processed documents
and preventing duplicates in the EUR-Lex web scraper. It maintains a set of
processed document identifiers and offers methods to check, update, and
deduplicate documents based on CELEX numbers.

Key Features:
- Recursive document discovery in data directories
- Efficient duplicate prevention during scraping
- Post-processing deduplication capabilities
- Intelligent duplicate resolution strategies
- Comprehensive logging of document activities
- Error-tolerant document handling
"""

import json
from pathlib import Path
from typing import Set, Dict, List, Tuple
from datetime import datetime
import shutil
import os
from loguru import logger


class DocumentTracker:
    """
    A comprehensive document tracking and deduplication system.

    Manages the tracking of processed documents by maintaining a set of 
    processed CELEX numbers. Automatically discovers and loads existing 
    documents from the specified data directory. Also provides advanced
    deduplication capabilities for post-processing cleanup.

    Attributes:
        data_dir (Path): Base directory for storing scraped documents
        processed_celex (Set[str]): Set of processed document CELEX numbers
        document_map (Dict[str, List[Path]]): Mapping of CELEX numbers to file paths (for deduplication)

    Notes:
        - Supports incremental scraping
        - Prevents re-processing of existing documents during scraping
        - Provides post-processing deduplication capabilities
        - Handles potential file reading errors gracefully
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the document tracking and deduplication system.

        Sets up the document tracker by specifying the base data directory 
        and automatically loading existing processed documents.

        Args:
            data_dir (str): Path to the base directory containing scraped documents
                            This directory will be recursively searched for existing documents

        Notes:
            - Converts input path to a Path object for flexible path handling
            - Automatically calls _load_existing_documents() during initialization
            - Logs the number of existing documents loaded
            - Prepares document mapping for deduplication functionality
        """
        self.data_dir = Path(data_dir)
        self.processed_celex: Set[str] = set()
        self.document_map: Dict[str, List[Path]] = {}  # CELEX number -> list of file paths
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
        
    def find_duplicates(self) -> Dict[str, List[Path]]:
        """
        Identify documents with duplicate CELEX numbers.

        Analyzes the document mapping to find CELEX numbers 
        associated with multiple file paths.

        Returns:
            Dict[str, List[Path]]: A dictionary where keys are CELEX numbers 
                                   and values are lists of duplicate file paths

        Notes:
            - Returns only CELEX numbers with more than one file path
            - Provides a comprehensive view of document duplicates
            - Supports further duplicate resolution strategies
        """
        # Scan documents if the document map is empty
        if not self.document_map:
            self.scan_documents()
            
        duplicates = {}
        for celex, paths in self.document_map.items():
            if len(paths) > 1:
                duplicates[celex] = paths
        
        logger.info(f"Found {len(duplicates)} documents with duplicates")
        return duplicates
    
    def scan_documents(self):
        """
        Scan all documents in the data directory and build a comprehensive 
        document mapping based on CELEX numbers.

        Behavior:
            - Recursively searches the data directory for JSON files
            - Extracts CELEX numbers from document metadata
            - Builds a mapping of CELEX numbers to file paths
            - Handles potential file reading errors

        Notes:
            - Uses rglob for recursive file searching
            - Supports nested directory structures
            - Provides error logging for problematic files
            - Populates the document_map attribute
        """
        logger.info(f"Scanning documents in {self.data_dir}")
        self.document_map.clear()  # Reset the document map
        
        for json_file in self.data_dir.rglob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'metadata' in data and 'celex_number' in data['metadata']:
                        celex = data['metadata']['celex_number']
                        if celex not in self.document_map:
                            self.document_map[celex] = []
                        self.document_map[celex].append(json_file)
            except Exception as e:
                logger.error(f"Error processing {json_file}: {str(e)}")
    
    def _parse_date_from_path(self, path: Path) -> Tuple[datetime, str]:
        """
        Extract and parse the date from a document file path.

        Attempts to parse the date from the file path using various 
        strategies, supporting different directory and filename structures.

        Args:
            path (Path): File path to extract date from

        Returns:
            Tuple[datetime, str]: Parsed datetime object and original date string

        Notes:
            - Handles various date format patterns
            - Supports nested directory date representations
            - Provides robust date parsing with multiple fallback mechanisms
        """
        try:
            # Try to extract date from directory structure (year/month/day)
            parts = path.parts
            for i in range(len(parts) - 3, len(parts)):
                if parts[i].isdigit() and len(parts[i]) == 4:  # Year
                    year = int(parts[i])
                    if i + 1 < len(parts) and parts[i+1].isdigit() and len(parts[i+1]) <= 2:  # Month
                        month = int(parts[i+1])
                        if i + 2 < len(parts) and parts[i+2].isdigit() and len(parts[i+2]) <= 2:  # Day
                            day = int(parts[i+2])
                            date_str = f"{year}-{month:02d}-{day:02d}"
                            return datetime(year, month, day), date_str
            
            # Try to extract from journal ID format (YYYYMMDD)
            for part in parts:
                if part.isdigit() and len(part) == 8:
                    year = int(part[:4])
                    month = int(part[4:6])
                    day = int(part[6:8])
                    date_str = f"{year}-{month:02d}-{day:02d}"
                    return datetime(year, month, day), date_str
            
            # Default to file modification time if date extraction fails
            mtime = path.stat().st_mtime
            dt = datetime.fromtimestamp(mtime)
            date_str = dt.strftime("%Y-%m-%d")
            return dt, date_str
            
        except Exception as e:
            logger.error(f"Error parsing date from {path}: {str(e)}")
            # Return a very old date as fallback
            return datetime(1970, 1, 1), "1970-01-01"
    
    def keep_earliest_date(self, duplicates: Dict[str, List[Path]], backup_dir: str = None):
        """
        Keep only the document from the earliest date directory for each CELEX number.

        Resolves duplicates by selecting the earliest dated document 
        and optionally backing up removed duplicates.

        Args:
            duplicates (Dict[str, List[Path]]): Dictionary of duplicate documents
            backup_dir (str, optional): Directory to backup removed duplicates

        Notes:
            - Parses dates from file paths
            - Keeps the earliest dated document
            - Optionally backs up removed duplicates
            - Provides logging for removed duplicates
        """
        if backup_dir:
            backup_path = Path(backup_dir)
            backup_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Backing up duplicates to {backup_path}")
        
        for celex, paths in duplicates.items():
            try:
                # Parse dates from paths
                dated_paths = []
                for path in paths:
                    date, date_str = self._parse_date_from_path(path)
                    dated_paths.append((date, path, date_str))
                
                if not dated_paths:
                    logger.error(f"No valid dates found for CELEX {celex}, skipping")
                    continue
                
                # Sort by date
                dated_paths.sort(key=lambda x: x[0])
                
                # Keep the first (earliest) file
                earliest_file = dated_paths[0][1]
                logger.info(f"Keeping {earliest_file} (date: {dated_paths[0][2]}) for CELEX {celex}")
                
                # Handle duplicates
                for _, duplicate, date_str in dated_paths[1:]:
                    if backup_dir:
                        # Create relative path structure in backup dir
                        rel_path = duplicate.relative_to(self.data_dir)
                        backup_file = backup_path / rel_path
                        backup_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(duplicate), str(backup_file))
                        logger.info(f"Moved duplicate {duplicate} (date: {date_str}) to {backup_file}")
                    else:
                        duplicate.unlink()
                        logger.info(f"Removed duplicate {duplicate} (date: {date_str})")
            except Exception as e:
                logger.error(f"Error processing duplicates for CELEX {celex}: {str(e)}")
                continue
    
    def cleanup_empty_directories(self):
        """
        Remove empty directories in the data directory structure.

        Walks the directory tree bottom-up and removes any empty directories.

        Notes:
            - Removes nested empty directories
            - Provides logging for removed directories
        """
        logger.info("Cleaning up empty directories...")
        empty_dirs = 0
        
        # Walk bottom-up so we can remove empty parent directories
        for dirpath, dirnames, filenames in os.walk(self.data_dir, topdown=False):
            if not dirnames and not filenames:  # Directory is empty
                try:
                    dir_to_remove = Path(dirpath)
                    # Don't remove the root data directory
                    if dir_to_remove != self.data_dir:
                        dir_to_remove.rmdir()
                        empty_dirs += 1
                        logger.info(f"Removed empty directory: {dir_to_remove}")
                except Exception as e:
                    logger.error(f"Error removing directory {dirpath}: {str(e)}")
        
        logger.info(f"Removed {empty_dirs} empty directories")
    
    def deduplicate(self, backup_dir: str = None):
        """
        Run the complete deduplication process.

        Scans documents, detects duplicates, resolves them, and cleans up empty directories.

        Args:
            backup_dir (str, optional): Directory to backup removed duplicates. Defaults to None.

        Notes:
            - Provides comprehensive logging for the deduplication process
            - Supports optional backup of removed duplicates
        """
        self.scan_documents()
        duplicates = self.find_duplicates()
        
        if not duplicates:
            logger.info("No duplicates found")
        else:
            logger.info(f"Found {len(duplicates)} documents with duplicates")
            self.keep_earliest_date(duplicates, backup_dir)
            logger.info("Deduplication complete")
        
        # Clean up empty directories after deduplication
        self.cleanup_empty_directories()
