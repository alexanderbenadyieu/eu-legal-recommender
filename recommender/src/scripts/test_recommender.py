#!/usr/bin/env python3
"""
Test script for the enhanced Pinecone recommender with legal-bert-base-uncased model and improved categorical feature processing.
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.pinecone_recommender import PineconeRecommender
from models.features import FeatureProcessor
from utils.db_connector import get_connector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Test the enhanced EU legal document recommender system')
    parser.add_argument('--db-path', type=str, default=os.getenv('DB_PATH', 'data/eu_legal_docs.db'),
                        help='Path to the SQLite database')
    parser.add_argument('--index-name', type=str, default=os.getenv('PINECONE_INDEX', 'eu-legal-docs'),
                        help='Name of the Pinecone index')
    parser.add_argument('--query', type=str,
                        default="Environmental regulations for climate change and pollution control",
                        help='Test query to use for recommendations')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of recommendations to return')
    parser.add_argument('--text-weight', type=float, default=0.7,
                        help='Weight for text similarity (0-1)')
    parser.add_argument('--categorical-weight', type=float, default=0.3,
                        help='Weight for categorical similarity (0-1)')
    return parser.parse_args()

def initialize_feature_processor(db_path):
    """Initialize and fit the feature processor with data from the database."""
    logger.info(f"Initializing feature processor with data from {db_path}")
    
    # Connect to the database
    db_connector = get_connector('sqlite', db_path=db_path, db_type='consolidated')
    
    # Fetch a sample of documents to train the feature processor
    documents = []
    for tier in range(1, 5):  # Tiers 1-4
        tier_docs = db_connector.fetch_tier_documents(tier)
        documents.extend(tier_docs[:50])  # Take 50 documents from each tier
        logger.info(f"Fetched {len(tier_docs[:50])} documents from tier {tier}")
    
    # Create and fit feature processor
    feature_processor = FeatureProcessor()
    feature_processor.fit(documents)
    logger.info(f"Fitted feature processor with {len(documents)} documents")
    logger.info(f"Feature dimensions: {feature_processor.get_feature_dims()}")
    
    # Close database connection
    db_connector.close()
    
    return feature_processor

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
    
    # Initialize feature processor
    feature_processor = initialize_feature_processor(args.db_path)
    
    # Initialize recommender
    logger.info("Initializing enhanced Pinecone recommender with legal-bert-base-uncased model")
    recommender = PineconeRecommender(
        api_key=pinecone_api_key,
        index_name=args.index_name,
        embedder_model='nlpaueb/legal-bert-base-uncased',
        feature_processor=feature_processor,
        text_weight=args.text_weight,
        categorical_weight=args.categorical_weight
    )
    
    # Test query
    test_query = args.query
    logger.info(f"Testing enhanced recommender with query: '{test_query}'")
    
    # Test with different embedding types
    embedding_types = ['combined', 'summary']
    
    for embedding_type in embedding_types:
        logger.info(f"\nTesting with embedding type: {embedding_type}")
        
        # Get recommendations for this embedding type
        recommendations = recommender.get_recommendations(
            query_text=test_query,
            embedding_type=embedding_type,
            top_k=args.top_k
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
    
    # Test with categorical features and client preferences
    logger.info("\nTesting with categorical features and client preferences")
    
    # Example categorical features for the query
    query_features = {
        "subject_matters": ["environment", "climate change", "emission trading", "pollution"],
        "eurovoc_descriptors": ["environmental protection", "climate change", "greenhouse gas"],
        "form": "regulation",
        "responsible_body": "European Parliament"
    }
    
    # Example client preferences (higher weights for certain feature values)
    client_preferences = {
        "subject_matters": 0.2,  # Boost documents with matching subject matters
        "form": 0.1            # Boost documents with matching form
    }
    
    recommendations_with_features = recommender.get_recommendations(
        query_text=test_query, 
        query_features=query_features,
        client_preferences=client_preferences,
        top_k=args.top_k
    )
    
    # Display results with categorical features and preferences
    logger.info(f"Found {len(recommendations_with_features)} recommendations with categorical features and preferences")
    for i, rec in enumerate(recommendations_with_features):
        logger.info(f"Recommendation with features and preferences {i+1}:")
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

    # Test document-to-document recommendations
    logger.info("\nTesting document-to-document recommendations")
    
    # Use the first recommendation from the previous query as the reference document
    if recommendations_with_features:
        reference_doc_id = recommendations_with_features[0]['id']
        logger.info(f"Finding documents similar to document ID: {reference_doc_id}")
        
        similar_docs = recommender.get_recommendations_by_id(
            document_id=reference_doc_id,
            top_k=args.top_k,
            include_categorical=True,
            client_preferences=client_preferences
        )
        
        # Display similar documents
        logger.info(f"Found {len(similar_docs)} similar documents")
        for i, doc in enumerate(similar_docs):
            logger.info(f"Similar document {i+1}:")
            logger.info(f"  Document ID: {doc['id']}")
            logger.info(f"  Score: {doc['score']:.4f}")
            logger.info(f"  Text Score: {doc['text_score']:.4f}")
            logger.info(f"  Categorical Score: {doc.get('categorical_score', 'N/A')}")
            
            if doc.get('metadata'):
                if doc['metadata'].get('title'):
                    logger.info(f"  Title: {doc['metadata']['title']}")
                if doc['metadata'].get('form'):
                    logger.info(f"  Form: {doc['metadata']['form']}")
            logger.info("---")

if __name__ == "__main__":
    main()
