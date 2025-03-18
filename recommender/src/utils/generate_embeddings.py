#!/usr/bin/env python3
"""
Script to generate embeddings for document summaries and store them in Pinecone.
This is a convenience wrapper around the PineconeEmbeddingManager.
"""
import os
import logging
from pathlib import Path
from .pinecone_embeddings import PineconeEmbeddingManager
from .db_connector import get_connector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('embeddings_generation.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate and store document embeddings in Pinecone")
    
    # Database source arguments
    db_group = parser.add_argument_group('Database Source')
    db_group.add_argument("--db-type", type=str, choices=["sqlite", "api"], default="sqlite", 
                        help="Type of database connector to use")
    db_group.add_argument("--db-path", type=str, 
                        default=str(Path(__file__).parent.parent.parent.parent / "scraper" / "data" / "eurlex.db"),
                        help="Path to the SQLite database (for sqlite db-type)")
    db_group.add_argument("--db-structure", type=str, choices=["consolidated", "legacy"], default="consolidated",
                        help="Database structure type ('consolidated' or 'legacy')")
    db_group.add_argument("--api-url", type=str, 
                        help="URL for the API (for api db-type)")
    db_group.add_argument("--db-api-key", type=str, 
                        help="API key for the database API (for api db-type)")
    
    # Document selection arguments
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4], default=1, 
                        help="Document tier to process (1, 2, 3, or 4)")
    
    # Pinecone arguments
    pinecone_group = parser.add_argument_group('Pinecone Configuration')
    pinecone_group.add_argument("--pinecone-api-key", type=str, required=True, 
                        help="Pinecone API key")
    pinecone_group.add_argument("--index-name", type=str, default="eu-legal-documents-legal-bert", 
                        help="Pinecone index name")
    
    # Other arguments
    parser.add_argument("--batch-size", type=int, default=100, 
                        help="Batch size for Pinecone uploads")
    parser.add_argument("--model", type=str, default="nlpaueb/legal-bert-small-uncased",
                        help="Model to use for embeddings (can be a SentenceTransformer or HuggingFace model)")
    # Removed fallback model option as we're specifically using legal-bert
    
    args = parser.parse_args()
    
    # Validate arguments based on db-type
    if args.db_type == "sqlite":
        if not os.path.exists(args.db_path):
            logger.error(f"Database file not found: {args.db_path}")
            return
    elif args.db_type == "api" and not args.api_url:
        logger.error("--api-url is required when using api db-type")
        return
    
    logger.info(f"Starting embedding generation for tier {args.tier} documents")
    logger.info(f"Using database type: {args.db_type}")
    logger.info(f"Using model: {args.model}")
    
    # Initialize database connector
    if args.db_type == "sqlite":
        logger.info(f"Using SQLite database: {args.db_path} (structure: {args.db_structure})")
        db_connector = get_connector("sqlite", db_path=args.db_path, db_type=args.db_structure)
    else:  # api
        logger.info(f"Using API at: {args.api_url}")
        db_connector = get_connector("api", api_url=args.api_url, api_key=args.db_api_key)
    
    # Initialize manager
    manager = PineconeEmbeddingManager(
        api_key=args.pinecone_api_key,
        environment="gcp-starter",  # Using serverless GCP starter environment
        index_name=args.index_name,
        embedder_model=args.model
    )
    
    # Process documents
    manager.process_tier_documents(
        db_connector=db_connector,
        tier=args.tier,
        batch_size=args.batch_size
    )
    
    logger.info(f"Completed embedding generation for tier {args.tier} documents")

if __name__ == "__main__":
    main()
