#!/usr/bin/env python3
"""
Document Deduplication Command-Line Interface

This script provides a command-line interface for deduplicating EUR-Lex documents
using the enhanced DocumentTracker's deduplication capabilities.
"""
import argparse
from loguru import logger
from .document_tracker import DocumentTracker
from .logging_config import setup_logging

def main():
    """
    Main entry point for the document deduplication CLI.
    
    Parses command-line arguments and initiates the deduplication process.
    """
    parser = argparse.ArgumentParser(description="Deduplicate EUR-Lex documents based on CELEX numbers")
    parser.add_argument("--data-dir", required=True, help="Base directory containing scraped documents")
    parser.add_argument("--backup-dir", help="Optional directory to backup removed duplicates")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger.info(f"Starting deduplication for {args.data_dir}")
    
    # Initialize document tracker with deduplication capabilities
    tracker = DocumentTracker(args.data_dir)
    
    # Run deduplication
    tracker.deduplicate(args.backup_dir)
    
    logger.success("Deduplication completed successfully")

if __name__ == "__main__":
    main()
