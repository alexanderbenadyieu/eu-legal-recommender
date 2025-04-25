"""
Comprehensive example demonstrating how to use temporal boosting with both
recommendation methods (by ID and by query) using the consolidated database.
"""
import os
import sys
import logging
from datetime import datetime, date
import math
from typing import Dict, List, Any, Optional

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logging import get_logger
from src.models.pinecone_recommender import PineconeRecommender
import database_utils

# Set up logging
logger = get_logger(__name__)

def display_results(results, title, with_dates=True):
    """Display recommendation results with dates."""
    logger.info(f"\n--- {title} ---")
    for i, result in enumerate(results):
        doc_id = result['id']
        
        # Fetch document info from the database if needed
        date_str = "N/A"
        if with_dates:
            doc_info = database_utils.get_document_by_celex(doc_id)
            date_str = doc_info['date_of_document'] if doc_info and 'date_of_document' in doc_info else "Unknown"
        
        # Display the result with date information
        temporal_score = result.get('temporal_score', 0.0)
        original_score = result.get('original_similarity', result['score'])
        
        if 'temporal_score' in result:
            logger.info(f"  {i+1}. Document {doc_id} (Date: {date_str}, Score: {result['score']:.4f}, "
                       f"Temporal: {temporal_score:.4f}, Original: {original_score:.4f})")
        else:
            logger.info(f"  {i+1}. Document {doc_id} (Date: {date_str}, Score: {result['score']:.4f})")

def main():
    """Run the comprehensive temporal boost example."""
    logger.info("Starting comprehensive temporal boost example")
    
    # Check if the consolidated database exists
    db_path = database_utils.CONSOLIDATED_DB_PATH
    if not os.path.exists(db_path):
        logger.error(f"Consolidated database not found at: {db_path}")
        return
    
    logger.info(f"Using consolidated database at: {db_path}")
    
    # Initialize the recommender
    recommender = PineconeRecommender(
        api_key=os.environ.get('PINECONE_API_KEY'),
        index_name=os.environ.get('PINECONE_INDEX', 'eu-legal-documents-legal-bert'),
        embedder_model=os.environ.get('EMBEDDING_MODEL', 'nlpaueb/legal-bert-base-uncased')
    )
    
    # Example 1: Get recommendations by document ID
    document_id = "32024R1261"  # Example document ID
    logger.info(f"\nExample 1: Getting recommendations for document: {document_id}")
    
    # Get recommendations without temporal boosting
    standard_results = recommender.get_recommendations_by_id(
        document_id=document_id,
        top_k=5
    )
    
    display_results(standard_results, "Standard Ranking (No Temporal Boost)")
    
    # Get recommendations with temporal boosting
    temporal_results = recommender.get_recommendations_by_id(
        document_id=document_id,
        top_k=5,
        temporal_boost=0.5
    )
    
    display_results(temporal_results, "Ranking with Temporal Boost (0.5)")
    
    # Example 2: Get recommendations by query text
    query_text = "Regulation on artificial intelligence and data protection"
    logger.info(f"\nExample 2: Getting recommendations for query: '{query_text}'")
    
    # Get recommendations without temporal boosting
    standard_query_results = recommender.get_recommendations(
        query_text=query_text,
        top_k=5
    )
    
    display_results(standard_query_results, "Standard Query Ranking (No Temporal Boost)")
    
    # Get recommendations with temporal boosting
    temporal_query_results = recommender.get_recommendations(
        query_text=query_text,
        top_k=5,
        temporal_boost=0.5
    )
    
    display_results(temporal_query_results, "Query Ranking with Temporal Boost (0.5)")
    
    # Example 3: Different temporal boost weights
    logger.info(f"\nExample 3: Comparing different temporal boost weights")
    
    # Try different temporal boost weights
    boost_weights = [0.2, 0.5, 0.8]
    for weight in boost_weights:
        results = recommender.get_recommendations_by_id(
            document_id=document_id,
            top_k=3,
            temporal_boost=weight
        )
        
        display_results(results, f"Temporal Boost Weight: {weight}")
    
    logger.info("Comprehensive temporal boost example completed")

if __name__ == "__main__":
    main()
