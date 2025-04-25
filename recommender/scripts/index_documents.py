#!/usr/bin/env python3
"""
Script to index EU legal documents into Pinecone using the enhanced recommender system
with legal-bert-base-uncased model and improved categorical feature processing.
"""
import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.pinecone_embeddings import PineconeEmbeddingManager
from src.models.features import FeatureProcessor
from src.models.embeddings import BERTEmbedder  # Import BERTEmbedder as it's needed by PineconeEmbeddingManager
from src.utils.db_connector import get_connector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("indexing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Index EU legal documents into Pinecone')
    parser.add_argument('--db-path', type=str, default=os.getenv('DB_PATH', 'data/eu_legal_docs.db'),
                        help='Path to the SQLite database')
    parser.add_argument('--index-name', type=str, default=os.getenv('PINECONE_INDEX', 'eu-legal-documents-legal-bert'),
                        help='Name of the Pinecone index')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size for indexing documents')
    parser.add_argument('--tiers', type=str, default='1,2,3,4',
                        help='Comma-separated list of tiers to index (1-4)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of documents to index per tier')
    parser.add_argument('--recreate-index', action='store_true',
                        help='Recreate the Pinecone index (WARNING: this will delete all existing data)')
    parser.add_argument('--dimension', type=int, default=768,
                        help='Vector dimension for the Pinecone index (768 for legal-bert-base-uncased)')
    parser.add_argument('--text-weight', type=float, default=0.7,
                        help='Weight for text similarity (0-1)')
    parser.add_argument('--categorical-weight', type=float, default=0.3,
                        help='Weight for categorical similarity (0-1)')
    return parser.parse_args()

def initialize_feature_processor(db_path: str) -> FeatureProcessor:
    """Initialize and fit the feature processor with data from the database."""
    logger.info(f"Initializing feature processor with data from {db_path}")
    
    # Connect to the database
    db_connector = get_connector('sqlite', db_path=db_path, db_type='consolidated')
    
    # Fetch a sample of documents to train the feature processor
    documents = []
    for tier in range(1, 5):  # Tiers 1-4
        tier_docs = db_connector.fetch_tier_documents(tier)
        sample_size = min(100, len(tier_docs))  # Take up to 100 documents from each tier
        documents.extend(tier_docs[:sample_size])
        logger.info(f"Fetched {sample_size} documents from tier {tier} for feature processor training")
    
    # Create and fit feature processor
    feature_processor = FeatureProcessor()
    feature_processor.fit(documents)
    logger.info(f"Fitted feature processor with {len(documents)} documents")
    
    # Log feature dimensions
    feature_dims = feature_processor.get_feature_dims()
    logger.info(f"Feature dimensions: {feature_dims}")
    
    # Close database connection
    db_connector.close()
    
    return feature_processor

def index_documents(embedding_manager: PineconeEmbeddingManager, db_connector: Any, 
                   tiers: List[int], batch_size: int, limit: Optional[int] = None) -> None:
    """Index documents from the database into Pinecone."""
    total_indexed = 0
    start_time = time.time()
    
    for tier in tiers:
        logger.info(f"Fetching and indexing documents from tier {tier}")
        
        # Fetch documents for this tier
        documents = db_connector.fetch_tier_documents(tier)
        
        if limit:
            documents = documents[:limit]
            
        logger.info(f"Found {len(documents)} documents in tier {tier}")
        
        try:
            # Generate embeddings
            embeddings_data = embedding_manager.generate_document_embeddings(documents)
            
            # Upload to Pinecone
            embedding_manager.upload_to_pinecone(embeddings_data, batch_size)
            
            total_indexed += len(documents)
            logger.info(f"Successfully indexed {len(documents)} documents from tier {tier}, total: {total_indexed}")
            
        except Exception as e:
            logger.error(f"Error indexing tier {tier} documents: {str(e)}")
            continue
            
        # Sleep briefly to avoid overwhelming the API
        time.sleep(1.0)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Indexing completed. Total documents indexed: {total_indexed}")
    logger.info(f"Elapsed time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get Pinecone API key
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    if not pinecone_api_key:
        logger.error("PINECONE_API_KEY not found in environment variables")
        sys.exit(1)
    
    # Parse tiers
    tiers = [int(tier) for tier in args.tiers.split(',')]
    logger.info(f"Will index documents from tiers: {tiers}")
    
    # Initialize feature processor (still useful for document metadata)
    feature_processor = initialize_feature_processor(args.db_path)
    
    # Initialize embedding manager
    logger.info("Initializing PineconeEmbeddingManager with legal-bert-base-uncased model")
    embedding_manager = PineconeEmbeddingManager(
        api_key=pinecone_api_key,
        environment="gcp-starter",  # Using serverless environment
        index_name=args.index_name,
        dimension=args.dimension,
        embedder_model='nlpaueb/legal-bert-base-uncased'
    )
    
    # Recreate index if requested
    if args.recreate_index:
        logger.warning("Recreating Pinecone index - this will delete all existing data!")
        embedding_manager.delete_and_recreate_index()
        logger.info(f"Created new Pinecone index '{args.index_name}' with dimension {args.dimension}")
    
    # Connect to the database
    logger.info(f"Connecting to database at {args.db_path}")
    db_connector = get_connector('sqlite', db_path=args.db_path, db_type='consolidated')
    
    # Index documents
    index_documents(
        embedding_manager=embedding_manager,
        db_connector=db_connector,
        tiers=tiers,
        batch_size=args.batch_size,
        limit=args.limit
    )
    
    # Close database connection
    db_connector.close()
    
    logger.info("Indexing process completed successfully")

if __name__ == "__main__":
    main()
