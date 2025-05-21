#!/usr/bin/env python
"""
Run the EU Legal Document Recommender System.

This script serves as the main entry point for the recommender system, providing
a command-line interface for generating recommendations based on queries or document IDs.
It handles configuration loading, model initialization, and recommendation generation.

For advanced operations such as document indexing, embedding recreation, and user profile management,
use the full CLI available at recommender/src/cli/cli.py.

Usage:
    python run_recommender.py --query "renewable energy policy" --profile profiles/renewable_energy_client.json
    python run_recommender.py --document-id 32018L2001 --profile profiles/renewable_energy_client.json
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import json

# Add the project root to the Python path to access all modules correctly
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(project_root))  # Add parent directory for database_utils

# Load environment variables from .env file
from dotenv import load_dotenv
# Try to load from recommender directory first, then from project root
env_path = os.path.join(project_root, '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
    print(f"Loaded environment variables from {env_path}")
else:
    # Try project root
    env_path = os.path.join(os.path.dirname(project_root), '.env')
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
        print(f"Loaded environment variables from {env_path}")

# Import recommender components
try:
    # When running from recommender directory
    from src.models.embeddings import BERTEmbedder
    from src.models.features import FeatureProcessor
    from src.models.pinecone_recommender import PineconeRecommender
    from src.models.personalized_recommender import PersonalizedRecommender
    from src.models.user_profile import UserProfile
    from src.config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, EMBEDDER
except ImportError:
    # When running from project root
    from recommender.src.models.embeddings import BERTEmbedder
    from recommender.src.models.features import FeatureProcessor
    from recommender.src.models.pinecone_recommender import PineconeRecommender
    from recommender.src.models.personalized_recommender import PersonalizedRecommender
    from recommender.src.models.user_profile import UserProfile
    from recommender.src.config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, EMBEDDER

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, 'logs', 'recommender.log'))
    ]
)
logger = logging.getLogger(__name__)

def load_user_profile(profile_path, embedder, feature_processor=None):
    """
    Load a user profile from a JSON file.
    
    Args:
        profile_path: Path to the user profile JSON file
        embedder: BERTEmbedder instance for generating text embeddings
        feature_processor: Optional FeatureProcessor for handling categorical features
        
    Returns:
        UserProfile instance
    """
    logger.info(f"Loading user profile from {profile_path}")
    
    try:
        with open(profile_path, 'r') as f:
            profile_dict = json.load(f)
            
        user_profile = UserProfile.from_dict(
            profile_dict=profile_dict,
            embedder=embedder,
            feature_processor=feature_processor
        )
        
        # Check if the profile has an expert profile description but no embedding
        if 'profile' in profile_dict and 'expert_profile' in profile_dict['profile'] and \
           'description' in profile_dict['profile']['expert_profile'] and \
           user_profile.expert_profile_embedding is None:
            # Create the expert profile embedding
            expert_description = profile_dict['profile']['expert_profile']['description']
            logger.info(f"Creating expert profile embedding for {profile_dict['user_id']}")
            user_profile.create_expert_profile(expert_description)
            
        logger.info(f"Successfully loaded user profile for {profile_dict['user_id']}")
        return user_profile
        
    except Exception as e:
        logger.error(f"Error loading user profile: {str(e)}")
        raise

def get_recommendations(args):
    """
    Get recommendations based on command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        List of recommendations
    """
    # Initialize embedder
    embedder = BERTEmbedder(
        model_name=args.model or EMBEDDER['model_name'],
        device=args.device or EMBEDDER['device']
    )
    
    # Initialize feature processor if needed
    feature_processor = None
    if args.features:
        try:
            with open(args.features, 'r') as f:
                feature_config = json.load(f)
                
            feature_processor = FeatureProcessor(feature_config=feature_config)
            logger.info(f"Initialized feature processor with config from {args.features}")
        except Exception as e:
            logger.warning(f"Error loading feature config, proceeding without: {str(e)}")
    
    # Initialize recommender
    api_key = args.api_key or PINECONE_API_KEY
    if not api_key:
        logger.error("No Pinecone API key provided. Set PINECONE_API_KEY in .env or pass --api-key")
        sys.exit(1)
        
    # Create recommender based on whether a profile is provided
    if args.profile:
        # Personalized recommendations
        user_profile = load_user_profile(args.profile, embedder, feature_processor)
        
        recommender = PersonalizedRecommender(
            api_key=api_key,
            index_name=args.index or 'eu-legal-documents-legal-bert',
            embedder_model=args.model or EMBEDDER['model_name']
        )
        
        # Get recommendations
        if args.query:
            logger.info(f"Getting personalized recommendations for query: {args.query}")
            # Store the user profile in the recommender's internal dictionary
            recommender._user_profiles[user_profile.user_id] = user_profile
            
            recommendations = recommender.get_personalized_recommendations(
                user_id=user_profile.user_id,
                query_text=args.query,
                top_k=args.top_k
            )
        elif args.document_id:
            logger.info(f"Getting personalized similar documents for document: {args.document_id}")
            # Store the user profile in the recommender's internal dictionary
            recommender._user_profiles[user_profile.user_id] = user_profile
            
            recommendations = recommender.get_recommendations_by_id(
                document_id=args.document_id,
                top_k=args.top_k,
                include_categorical=True,
                client_preferences=user_profile.categorical_preferences
            )
        else:
            logger.error("Either --query or --document-id must be provided")
            sys.exit(1)
    else:
        # Standard recommendations
        recommender = PineconeRecommender(
            api_key=api_key,
            index_name=args.index or 'eu-legal-documents-legal-bert',
            embedder_model=args.model or EMBEDDER['model_name'],
            feature_processor=feature_processor,
            text_weight=args.text_weight,
            categorical_weight=args.categorical_weight
        )
        
        # Get recommendations
        if args.query:
            logger.info(f"Getting standard recommendations for query: {args.query}")
            recommendations = recommender.get_recommendations(
                query_text=args.query,
                top_k=args.top_k
            )
        elif args.document_id:
            logger.info(f"Getting similar documents for document: {args.document_id}")
            recommendations = recommender.get_recommendations_by_id(
                document_id=args.document_id,
                top_k=args.top_k
            )
        else:
            logger.error("Either --query or --document-id must be provided")
            sys.exit(1)
            
    return recommendations

def display_recommendations(recommendations):
    """
    Display recommendations in a formatted way.
    
    Args:
        recommendations: List of recommendation dictionaries
    """
    print("\n===== RECOMMENDATIONS =====\n")
    
    for i, rec in enumerate(recommendations):
        print(f"{i+1}. Document ID: {rec['id']} (Score: {rec['score']:.4f})")
        print(f"   Text Similarity: {rec['text_score']:.4f}")
        
        if rec.get('categorical_score') is not None:
            print(f"   Categorical Similarity: {rec['categorical_score']:.4f}")
            
        if rec.get('metadata'):
            print("   Metadata:")
            for key, value in rec['metadata'].items():
                if key != 'categorical_features' and key != 'embedding':
                    print(f"      {key}: {value}")
                    
        print()

def main():
    """Main entry point for the recommender CLI."""
    parser = argparse.ArgumentParser(description="EU Legal Document Recommender System")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--query', type=str, help="Text query for recommendations")
    input_group.add_argument('--document-id', type=str, help="Document ID for similar document recommendations")
    
    # Profile options
    parser.add_argument('--profile', type=str, help="Path to user profile JSON file for personalized recommendations")
    
    # Recommender options
    parser.add_argument('--api-key', type=str, help="Pinecone API key (overrides environment variable)")
    parser.add_argument('--index', type=str, help="Pinecone index name")
    parser.add_argument('--model', type=str, help="BERT model name")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help="Device to run model on")
    parser.add_argument('--features', type=str, help="Path to feature configuration JSON file")
    parser.add_argument('--text-weight', type=float, default=0.7, help="Weight for text similarity (0-1)")
    parser.add_argument('--categorical-weight', type=float, default=0.3, help="Weight for categorical similarity (0-1)")
    parser.add_argument('--top-k', type=int, default=5, help="Number of recommendations to return")
    
    args = parser.parse_args()
    
    # Validate weights
    if args.text_weight + args.categorical_weight != 1.0:
        logger.warning(f"Weights don't sum to 1.0: text_weight={args.text_weight}, categorical_weight={args.categorical_weight}")
        logger.warning("Normalizing weights...")
        total = args.text_weight + args.categorical_weight
        args.text_weight /= total
        args.categorical_weight /= total
    
    try:
        # Get recommendations
        recommendations = get_recommendations(args)
        
        # Display recommendations
        display_recommendations(recommendations)
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
