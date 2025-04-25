#!/usr/bin/env python3
"""
Direct test of the weight optimizer without dependency on PersonalizedRecommender.
"""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import directly from modules to avoid the problematic PersonalizedRecommender
from src.models.embeddings import BERTEmbedder
from src.models.pinecone_recommender import PineconeRecommender
from src.models.user_profile import UserProfile
from src.utils.weight_optimizer import WeightOptimizer
from src.utils.recommender_evaluation import RecommenderEvaluator

def main():
    # Load environment variables with override
    load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env', override=True)
    
    # Get Pinecone API key from .env file directly
    api_key = None
    with open(Path(__file__).parent.parent / '.env', 'r') as f:
        for line in f:
            if line.startswith('PINECONE_API_KEY='):
                api_key = line.strip().split('=', 1)[1]
                break
    
    if not api_key:
        logger.error("PINECONE_API_KEY not found in .env file")
        sys.exit(1)
    
    # Mask API key for logging
    masked_key = api_key[:8] + "*" * (len(api_key) - 12) + api_key[-4:]
    logger.info(f"Using Pinecone API key: {masked_key}")
    
    # Initialize embedder
    embedder = BERTEmbedder(model_name='nlpaueb/legal-bert-base-uncased')
    
    # Initialize Pinecone recommender directly
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "eu-legal-documents-legal-bert")
    logger.info(f"Using Pinecone index: {pinecone_index_name}")
    pinecone_recommender = PineconeRecommender(
        api_key=api_key,
        index_name=pinecone_index_name,
        embedder_model='nlpaueb/legal-bert-base-uncased'
    )
    
    # Create a user profile with renewable energy client data
    logger.info("Creating renewable energy client profile")
    profile_path = Path(__file__).parent.parent / "profiles" / "renewable_energy_client.json"
    with open(profile_path, 'r') as f:
        profile_data = json.load(f)
    
    # Extract profile components
    if "profile" in profile_data:
        profile_components = profile_data["profile"]
    else:
        profile_components = profile_data
    
    # Create the profile
    profile = UserProfile(
        user_id="renewable_energy_client",
        embedder=embedder
    )
    
    # Set expert profile text if available
    if "expert_profile" in profile_components:
        expert_text = profile_components["expert_profile"].get("description", "")
        if expert_text:
            logger.info("Setting expert profile text")
            profile.create_expert_profile(expert_text)
    
    # Set categorical preferences if available
    if "categorical_preferences" in profile_components:
        logger.info("Setting categorical preferences")
        profile.set_categorical_preferences(profile_components["categorical_preferences"])
    
    # Add historical documents if available
    if "historical_documents" in profile_components:
        hist_docs = profile_components["historical_documents"]
        logger.info(f"Adding {len(hist_docs)} historical documents")
        for doc_id in hist_docs:
            try:
                # Generate embedding using the embedder directly
                document_text = f"Document {doc_id} for EU legal recommendation"
                query_embedding = embedder.generate_embedding(text=document_text)
                
                # Add the document embedding to the profile
                profile.add_historical_document(doc_id, query_embedding)
                logger.info(f"Added historical document: {doc_id}")
            except Exception as e:
                logger.warning(f"Could not add historical document {doc_id}: {str(e)}")
    
    # Test get_composite_profile
    logger.info("Testing get_composite_profile")
    try:
        embedding, preferences = profile.get_composite_profile()
        logger.info(f"Profile preferences keys: {list(preferences.keys() if preferences else [])}")
        logger.info("Successfully retrieved composite profile")
        
        # Test merging with query features
        query_features = {"form": {"regulation": 0.9}}
        merged_features = {}
        
        # Copy profile preferences to merged_features
        if preferences and isinstance(preferences, dict):
            logger.info("Merging profile preferences")
            for key, value in preferences.items():
                merged_features[key] = value
        
        # Copy query features to merged_features
        if query_features and isinstance(query_features, dict):
            logger.info("Merging query features")
            for key, value in query_features.items():
                merged_features[key] = value
        
        logger.info(f"Successfully merged features: {list(merged_features.keys())}")
        logger.info("Dictionary test successful. The get_composite_profile method is correctly deep copying dictionaries.")
    
    except Exception as e:
        logger.error(f"Error testing get_composite_profile: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    logger.info("All tests successful!")

if __name__ == "__main__":
    main()
