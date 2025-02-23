"""
Document Deduplication Module for EUR-Lex Web Scraper

This module provides advanced document deduplication capabilities, 
ensuring data integrity and preventing redundant storage of documents. 
It offers sophisticated mechanisms to identify, track, and manage 
duplicate documents based on various criteria.

Key Features:
- CELEX number-based duplicate detection
- Flexible file path scanning
- Intelligent duplicate resolution
- Comprehensive logging of deduplication activities
- Configurable deduplication strategies
"""

import json
from pathlib import Path
from typing import Dict, Set, List, Tuple
from datetime import datetime
import shutil
from loguru import logger
import os


class DocumentDeduplicator:
    """
    A comprehensive document deduplication system for the EUR-Lex web scraper.

    Manages the identification and handling of duplicate documents 
    within the scraped dataset. Provides methods to scan, detect, 
    and optionally resolve document duplicates.

    Attributes:
        data_dir (Path): Base directory containing scraped documents
        document_map (Dict[str, List[Path]]): Mapping of CELEX numbers to file paths

    Notes:
        - Supports recursive document discovery
        - Handles various document metadata structures
        - Provides detailed error logging
    """

    def __init__(self, data_dir: str):
        """
        Initialize the document deduplication system.

        Sets up the deduplicator by specifying the base data directory 
        and preparing for document scanning and duplicate detection.

        Args:
            data_dir (str): Path to the base directory containing scraped documents

        Notes:
            - Converts input path to a Path object
            - Prepares an empty document mapping dictionary
            - Ready for subsequent scanning and duplicate detection
        """
        self.data_dir = Path(data_dir)
        self.document_map: Dict[str, List[Path]] = {}  # CELEX number -> list of file paths
        
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
        return {celex: paths for celex, paths in self.document_map.items() 
                if len(paths) > 1}
    
    def _parse_date_from_path(self, path: Path) -> Tuple[datetime, str]:
        """
        Extract and parse the date from a document file path.

        Attempts to parse the date from the file path using various 
        strategies, supporting different directory and filename structures.

        Args:
            path (Path): File path to extract date from

        Returns:
            Tuple[datetime, str]: A tuple containing:
                - Parsed datetime object
                - Original date string representation

        Notes:
            - Handles various date format patterns
            - Supports nested directory date representations
            - Provides robust date parsing with multiple fallback mechanisms
        """
        try:
            # Get the date directory (which is the immediate parent of the file)
            date_dir = path.parent.name
            
            # The date directory should be in YYYYMMDD format
            if len(date_dir) == 8:  # YYYYMMDD format
                return datetime.strptime(date_dir, '%Y%m%d'), date_dir
            
            raise ValueError(f"Could not parse date from directory: {date_dir}")
        except Exception as e:
            logger.error(f"Error parsing date from path {path}: {str(e)}")
            raise
    
    def keep_earliest_date(self, duplicates: Dict[str, List[Path]], backup_dir: str = None):
        """
        Keep only the document from the earliest date directory for each CELEX number.

        Resolves duplicates by selecting the earliest dated document 
        and optionally backing up removed duplicates.

        Args:
            duplicates (Dict[str, List[Path]]): Dictionary of CELEX numbers to duplicate file paths
            backup_dir (str, optional): Directory to backup removed duplicates. Defaults to None.

        Notes:
            - Sorts file paths by date
            - Keeps the earliest dated file
            - Optionally backs up removed duplicates
            - Provides logging for removed duplicates
        """
        if backup_dir:
            backup_path = Path(backup_dir)
            backup_path.mkdir(parents=True, exist_ok=True)
        
        for celex, paths in duplicates.items():
            try:
                # Sort paths by date with error handling
                dated_paths = []
                for p in paths:
                    try:
                        date, date_str = self._parse_date_from_path(p)
                        dated_paths.append((date, p, date_str))
                    except Exception as e:
                        logger.error(f"Skipping path due to date parsing error - CELEX: {celex}, Path: {p}, Error: {str(e)}")
                        continue
                
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
    
    def cleanup(self, backup_dir: str = None):
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deduplicate EUR-Lex documents based on CELEX numbers")
    parser.add_argument("--data-dir", required=True, help="Base directory containing scraped documents")
    parser.add_argument("--backup-dir", help="Optional directory to backup removed duplicates")
    
    args = parser.parse_args()
    
    deduplicator = DocumentDeduplicator(args.data_dir)
    deduplicator.cleanup(args.backup_dir)
