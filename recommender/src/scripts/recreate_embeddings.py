#!/usr/bin/env python3
"""
Script to recreate all embeddings in Pinecone using the new approach:
1. Delete all existing embeddings
2. Regenerate embeddings for all documents (summary, keyword, and combined)
"""

import os
import sys
import logging
from dotenv import load_dotenv
from tqdm import tqdm

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.embeddings import BERTEmbedder
from utils.pinecone_embeddings import PineconeEmbeddingManager
from utils.db_connector import get_connector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Load environment variables
    load_dotenv()
    
    # Get Pinecone API key
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    if not pinecone_api_key:
        logger.error("PINECONE_API_KEY not found in environment variables")
        sys.exit(1)
    
    # Initialize embedder with the legal-bert model directly
    logger.info("Initializing BERT embedder with legal-bert model")
    embedder = BERTEmbedder(model_name='nlpaueb/legal-bert-small-uncased')
    
    # Initialize Pinecone embedding manager
    logger.info("Initializing Pinecone embedding manager")
    pinecone_environment = os.getenv('PINECONE_ENVIRONMENT', 'gcp-starter')
    embedding_manager = PineconeEmbeddingManager(
        api_key=pinecone_api_key,
        environment=pinecone_environment,
        index_name='eu-legal-docs',
        embedder_model='nlpaueb/legal-bert-small-uncased'  # Use the same model as the embedder
    )
    
    # Initialize database connector with the correct path to the database
    logger.info("Initializing database connector")
    # Use the processed_documents.db from the summarization directory
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                          'summarization', 'data', 'processed_documents.db')
    logger.info(f"Using database at: {db_path}")
    db_connector = get_connector('sqlite', db_path=db_path)
    
    # Delete all existing embeddings
    logger.info("Deleting all existing embeddings from Pinecone")
    embedding_manager.delete_all_embeddings()
    
    # Process each tier
    for tier in range(1, 5):  # Tiers 1-4
        logger.info(f"Processing tier {tier} documents")
        
        # Fetch documents for this tier
        documents = embedding_manager.fetch_tier_documents(db_connector, tier)
        
        if not documents:
            logger.warning(f"No documents found for tier {tier}")
            continue
        
        logger.info(f"Found {len(documents)} documents for tier {tier}")
        
        # Generate embeddings
        embeddings_data = embedding_manager.generate_document_embeddings(documents)
        
        # Upload to Pinecone
        embedding_manager.upload_to_pinecone(embeddings_data)
        
        logger.info(f"Completed processing tier {tier} documents")
    
    logger.info("All embeddings have been recreated successfully")

if __name__ == "__main__":
    main()
