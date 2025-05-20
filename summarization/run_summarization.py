#!/usr/bin/env python3
"""
Simple wrapper script to test the summarization pipeline.

This script provides a direct way to run the summarization pipeline
without dealing with complex import structures.
"""
import os
import sys
import argparse

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(os.path.dirname(project_root)))

# Now we can import the modules we need
from database_utils import get_db_connection
from summarization.src.pipeline import SummarizationPipeline
from summarization.src.utils.config import get_config, initialize_config

def main():
    """Run the summarization pipeline with command-line arguments."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the summarization pipeline')
    parser.add_argument('--tier', type=int, choices=[1, 2, 3, 4], default=1,
                        help='Tier to process (1: <600 words, 2: 600-2500 words, 3: 2500-20000 words, 4: >20000 words)')
    parser.add_argument('--db-type', type=str, choices=['consolidated', 'legacy'], default='consolidated',
                        help="Database type to use ('consolidated' or 'legacy')")
    parser.add_argument('--config', type=str, default=None,
                        help="Path to custom configuration file (optional)")
    args = parser.parse_args()
    
    print(f"Starting summarization pipeline for tier {args.tier}")
    
    # Initialize configuration with custom path if provided
    if args.config:
        print(f"Loading custom configuration from {args.config}")
        initialize_config(args.config)
    
    # Get configuration
    config = get_config()
    
    # Initialize pipeline
    pipeline = SummarizationPipeline(db_type=args.db_type, config=config)
    
    # Process documents for specified tier
    print(f"Processing documents for tier {args.tier}")
    processed_docs = pipeline.process_documents(tier=args.tier)
    
    # Print results
    print(f"\nTier {args.tier} Processing Results:")
    print(f"Total documents processed: {len(processed_docs)}")
    
    if processed_docs:
        for doc in processed_docs:
            print(f"\nDocument: {doc['celex_number']}")
            print(f"Words: {doc.get('total_words', 0)} -> {doc.get('summary_word_count', 0)}")
            print(f"Compression ratio: {doc.get('compression_ratio', 0):.2f}")
    else:
        print("No documents were processed.")

if __name__ == "__main__":
    main()
