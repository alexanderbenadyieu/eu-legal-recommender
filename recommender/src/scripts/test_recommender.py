#!/usr/bin/env python3
"""
Test script for the Pinecone recommender with the new embeddings.
"""
import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.pinecone_recommender import PineconeRecommender

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
    
    pinecone_environment = os.getenv('PINECONE_ENVIRONMENT', 'gcp-starter')
    
    # Initialize recommender
    logger.info("Initializing Pinecone recommender")
    recommender = PineconeRecommender(
        api_key=pinecone_api_key,
        index_name='eu-legal-docs',
        embedder_model='nlpaueb/legal-bert-small-uncased',
        text_weight=0.75,
        categorical_weight=0.25
    )
    
    # Test query
    test_query = "Environmental regulations for climate change and pollution control"
    logger.info(f"Testing recommender with query: '{test_query}'")
    
    # Test with different embedding types
    embedding_types = ['combined', 'summary', 'keyword']
    
    for embedding_type in embedding_types:
        logger.info(f"\nTesting with embedding type: {embedding_type}")
        
        # Get recommendations for this embedding type
        recommendations = recommender.get_recommendations(
            test_query,
            embedding_type=embedding_type,
            top_k=3  # Limit to top 3 for brevity
        )
        
        # Display results for this embedding type
        logger.info(f"Found {len(recommendations)} recommendations for {embedding_type} embedding")
        for i, rec in enumerate(recommendations[:3]):
            logger.info(f"Recommendation {i+1}:")
            logger.info(f"  Document ID: {rec['id']}")
            logger.info(f"  Title: {rec.get('metadata', {}).get('title', 'N/A')}")
            logger.info(f"  Score: {rec['score']:.4f}")
            logger.info(f"  Text Score: {rec['text_score']:.4f}")
            logger.info(f"  Categorical Score: {rec.get('categorical_score', 'N/A')}")
            if rec.get('metadata'):
                # Display title and document type first if available
                if rec['metadata'].get('title'):
                    logger.info(f"  Title: {rec['metadata']['title']}")
                if rec['metadata'].get('document_type'):
                    logger.info(f"  Document Type: {rec['metadata']['document_type']}")
                
                # Display categorical features if available
                if rec['metadata'].get('categorical_features'):
                    try:
                        cat_features = json.loads(rec['metadata']['categorical_features'])
                        logger.info(f"  Categorical Features: {json.dumps(cat_features, indent=2)}")
                    except json.JSONDecodeError:
                        logger.info(f"  Categorical Features: {rec['metadata']['categorical_features']}")
                
                # Display other metadata
                other_metadata = {k: v for k, v in rec['metadata'].items() 
                                if k not in ['title', 'document_type', 'categorical_features', 'summary']}
                if other_metadata:
                    logger.info(f"  Other Metadata: {', '.join([f'{k}: {v}' for k, v in other_metadata.items()])}")
            logger.info("---")
    
    # Test with client preferences
    logger.info("\nTesting with client preferences")
    client_preferences = {
        "subject_matter": ["environment", "climate", "emission", "pollution"]
        # No document_type or form restrictions
    }
    
    recommendations_with_prefs = recommender.get_recommendations(
        test_query, 
        query_features=client_preferences,
        top_k=3  # Limit to top 3 for brevity
    )
    
    # Display results with preferences
    logger.info(f"Found {len(recommendations_with_prefs)} recommendations with preferences")
    for i, rec in enumerate(recommendations_with_prefs[:3]):
        logger.info(f"Recommendation with preferences {i+1}:")
        logger.info(f"  Document ID: {rec['id']}")
        logger.info(f"  Score: {rec['score']:.4f}")
        logger.info(f"  Text Score: {rec['text_score']:.4f}")
        logger.info(f"  Categorical Score: {rec.get('categorical_score', 'N/A')}")
        if rec.get('metadata'):
            # Display title and document type first if available
            if rec['metadata'].get('title'):
                logger.info(f"  Title: {rec['metadata']['title']}")
            if rec['metadata'].get('document_type'):
                logger.info(f"  Document Type: {rec['metadata']['document_type']}")
            
            # Display categorical features if available
            if rec['metadata'].get('categorical_features'):
                try:
                    cat_features = json.loads(rec['metadata']['categorical_features'])
                    logger.info(f"  Categorical Features: {json.dumps(cat_features, indent=2)}")
                except json.JSONDecodeError:
                    logger.info(f"  Categorical Features: {rec['metadata']['categorical_features']}")
            
            # Display other metadata
            other_metadata = {k: v for k, v in rec['metadata'].items() 
                            if k not in ['title', 'document_type', 'categorical_features', 'summary']}
            if other_metadata:
                logger.info(f"  Other Metadata: {', '.join([f'{k}: {v}' for k, v in other_metadata.items()])}")
        logger.info("---")

if __name__ == "__main__":
    main()
