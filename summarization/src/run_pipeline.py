"""Run the summarization pipeline."""
import yaml
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from summarization.src.pipeline import SummarizationPipeline
from database_utils import get_db_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),  # Log to stdout
        logging.FileHandler('summarization.log')  # Also log to file
    ]
)
logger = logging.getLogger(__name__)

def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the summarization pipeline')
    parser.add_argument('--tier', type=int, choices=[1, 2, 3, 4], default=2,
                        help='Tier to process (1: <600 words, 2: 600-2500 words, 3: 2500-20000 words, 4: >20000 words)')
    parser.add_argument('--db-type', type=str, choices=['consolidated', 'legacy'], default='consolidated',
                        help="Database type to use ('consolidated' or 'legacy')")
    args = parser.parse_args()
    
    logger.info("Starting summarization pipeline")
    
    # Load config
    config_path = Path('/Users/alexanderbenady/DataThesis/eu-legal-recommender/summarization/config/summarisation_config.yaml')
    logger.info(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize pipeline with the specified database type
    logger.info(f"Initializing pipeline with {args.db_type} database")
    pipeline = SummarizationPipeline(db_type=args.db_type, config=config)
    
    # Process documents for specified tier
    logger.info(f"Starting Tier {args.tier} document processing")
    processed_docs = pipeline.process_documents(tier=args.tier)
    
    # Calculate statistics using stored values
    total_docs = len(processed_docs)
    
    # Handle both consolidated and legacy database field names
    if 'word_count' in processed_docs[0] if processed_docs else {}:
        # Consolidated database
        total_words = sum(doc['word_count'] for doc in processed_docs)
    else:
        # Legacy database
        total_words = sum(doc['total_words'] for doc in processed_docs)
        
    total_summary_words = sum(doc['summary_word_count'] for doc in processed_docs if doc['summary_word_count'])
    compression_ratios = [doc['compression_ratio'] for doc in processed_docs if doc['compression_ratio']]
    
    # Log processing statistics
    logger.info(f"\nTier {args.tier} Processing Statistics:")
    logger.info(f"Total documents processed: {total_docs}")
    if total_docs > 0:
        logger.info(f"Average document length: {total_words/total_docs:.0f} words")
        logger.info(f"Average summary length: {total_summary_words/total_docs:.0f} words")
        avg_compression = sum(compression_ratios) / len(compression_ratios) if compression_ratios else 0
        logger.info(f"Average compression ratio: {avg_compression:.2f}")
    
    # Document length distribution
    length_ranges = [
        (600, 1000), (1001, 1500), (1501, 2000), (2001, 2500)  # Tier 2
    ] if args.tier == 2 else [
        (20000, 30000), (30001, 40000), (40001, 50000), (50001, float('inf'))  # Tier 4
    ] if args.tier == 4 else [
        (2500, 5000), (5001, 10000), (10001, 15000), (15001, 20000)  # Tier 3
    ]
    length_dist = {f"{start}-{end}": 0 for start, end in length_ranges}
    
    for doc in processed_docs:
        # Get the word count from the appropriate field based on database type
        word_count = doc.get('word_count', doc.get('total_words', 0))
        
        for start, end in length_ranges:
            if start <= word_count <= end:
                length_dist[f"{start}-{end}"] += 1
                break
    
    logger.info("\nDocument Length Distribution:")
    for range_str, count in length_dist.items():
        percentage = (count / total_docs * 100) if total_docs > 0 else 0
        logger.info(f"{range_str} words: {count} docs ({percentage:.1f}%)")

if __name__ == "__main__":
    main()
