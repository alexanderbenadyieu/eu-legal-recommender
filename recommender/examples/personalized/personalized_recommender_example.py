#!/usr/bin/env python
"""
Example script demonstrating the personalized recommender with user profiles.

This script shows how to create and use user profiles for personalized recommendations
based on expert-curated profiles, historical engagement, and categorical preferences.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import json
from typing import Dict, List, Optional, Union

# Add parent directory to path to import recommender modules
sys.path.append(str(Path(__file__).parent.parent))

from src.models.embeddings import BERTEmbedder
from src.models.features import FeatureProcessor
from src.models.personalized_recommender import PersonalizedRecommender

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Demonstrate personalized recommendations with user profiles"
    )
    parser.add_argument(
        "--pinecone-api-key",
        required=True,
        help="Pinecone API key"
    )
    parser.add_argument(
        "--index-name",
        default="eu-legal-documents-legal-bert",
        help="Name of the Pinecone index"
    )
    parser.add_argument(
        "--user-id",
        default="example_user",
        help="User ID for the profile"
    )
    parser.add_argument(
        "--profile-text",
        default=None,
        help="Expert-curated profile text describing user interests"
    )
    parser.add_argument(
        "--profile-file",
        default=None,
        help="Path to JSON file containing expert-curated profile text"
    )
    parser.add_argument(
        "--historical-documents",
        nargs="*",
        default=[],
        help="List of historical document IDs (CELEX numbers) of interest to the user"
    )
    parser.add_argument(
        "--categorical-preferences",
        default=None,
        help="JSON string of categorical preferences, e.g., '{\"document_type\": [\"regulation\"], \"subject_matters\": [\"environment\", \"energy\"]}'"
    )
    parser.add_argument(
        "--preferences-file",
        default=None,
        help="Path to JSON file containing categorical preferences"
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Optional query text to combine with the user profile"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of recommendations to return"
    )
    parser.add_argument(
        "--document-type",
        default=None,
        help="Filter results by document type (e.g., 'regulation', 'directive')"
    )
    parser.add_argument(
        "--profile-weight",
        type=float,
        default=0.4,
        help="Weight for user profile in recommendation scoring (0-1)"
    )
    parser.add_argument(
        "--query-weight",
        type=float,
        default=0.6,
        help="Weight for query in recommendation scoring (0-1)"
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to SQLite database for storing user profiles"
    )
    
    return parser.parse_args()

def main():
    """Run the personalized recommender example."""
    args = parse_args()
    
    # Validate weights
    if args.profile_weight + args.query_weight != 1.0:
        logger.error("Profile weight and query weight must sum to 1.0")
        sys.exit(1)
    
    # Get profile text from file if specified
    profile_text = args.profile_text
    if args.profile_file:
        try:
            with open(args.profile_file, 'r') as f:
                profile_data = json.load(f)
                profile_text = profile_data.get('profile_text')
        except Exception as e:
            logger.error(f"Error loading profile file: {e}")
            sys.exit(1)
    
    # Get categorical preferences from file if specified
    categorical_preferences = None
    if args.categorical_preferences:
        try:
            categorical_preferences = json.loads(args.categorical_preferences)
        except Exception as e:
            logger.error(f"Error parsing categorical preferences: {e}")
            sys.exit(1)
    
    if args.preferences_file:
        try:
            with open(args.preferences_file, 'r') as f:
                categorical_preferences = json.load(f)
        except Exception as e:
            logger.error(f"Error loading preferences file: {e}")
            sys.exit(1)
    
    # Initialize feature processor with default weights
    feature_processor = FeatureProcessor(
        feature_weights={
            'document_type': 0.4,
            'subject_matters': 0.3,
            'author': 0.2,
            'form': 0.1
        },
        multi_valued_features=['subject_matters']
    )
    
    # Initialize personalized recommender
    logger.info("Initializing personalized recommender...")
    recommender = PersonalizedRecommender(
        api_key=args.pinecone_api_key,
        index_name=args.index_name,
        feature_processor=feature_processor,
        db_path=args.db_path,
        profile_weight=args.profile_weight,
        query_weight=args.query_weight
    )
    
    # Create user profile if profile text is provided
    if profile_text:
        logger.info(f"Creating expert-curated profile for user {args.user_id}")
        recommender.create_expert_profile(args.user_id, profile_text)
    
    # Add historical documents if provided
    if args.historical_documents:
        logger.info(f"Adding {len(args.historical_documents)} historical documents to user profile")
        for doc_id in args.historical_documents:
            recommender.add_historical_document(args.user_id, doc_id)
    
    # Set categorical preferences if provided
    if categorical_preferences:
        logger.info(f"Setting categorical preferences for user {args.user_id}")
        recommender.set_categorical_preferences(args.user_id, categorical_preferences)
    
    # Prepare filter if document type is specified
    filter = None
    if args.document_type:
        filter = {'document_type': args.document_type}
    
    # Get personalized recommendations
    logger.info(f"Getting personalized recommendations for user {args.user_id}")
    recommendations = recommender.get_personalized_recommendations(
        user_id=args.user_id,
        query_text=args.query,
        top_k=args.top_k,
        filter=filter
    )
    
    # Display recommendations
    logger.info(f"Found {len(recommendations)} personalized recommendations")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. Document ID: {rec['id']} (Score: {rec['score']:.4f})")
        print(f"   Text Similarity: {rec['text_score']:.4f}")
        
        if rec['categorical_score'] is not None:
            print(f"   Categorical Similarity: {rec['categorical_score']:.4f}")
        
        if 'metadata' in rec and rec['metadata']:
            print(f"   Metadata:")
            
            # Display CELEX number
            if 'celex_number' in rec['metadata']:
                print(f"   - celex_number: {rec['metadata']['celex_number']}")
            
            # Display summary (truncated if too long)
            if 'summary' in rec['metadata']:
                summary = rec['metadata']['summary']
                if len(summary) > 200:
                    summary = summary[:200] + "..."
                print(f"   - summary: {summary}")
            
            # Display document type
            if 'document_type' in rec['metadata']:
                print(f"   - document_type: {rec['metadata']['document_type']}")
            
            # Display subject matters
            if 'subject_matters' in rec['metadata']:
                print(f"   - subject_matters: {rec['metadata']['subject_matters']}")
            
            # Display tier if available
            if 'tier' in rec['metadata']:
                print(f"   - tier: {rec['metadata']['tier']}")

if __name__ == "__main__":
    main()
