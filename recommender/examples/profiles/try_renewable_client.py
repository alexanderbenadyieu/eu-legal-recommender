#!/usr/bin/env python
"""
Script to test the personalized recommender with the renewable energy client profile.
"""
import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sklearn.decomposition import PCA

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Import the modules
from recommender.src.models.features import FeatureProcessor
from recommender.src.models.pinecone_recommender import PineconeRecommender
from recommender.src.models.personalized_recommender import PersonalizedRecommender
from recommender.src.models.user_profile import UserProfile

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the personalized recommender with the renewable energy client profile."""
    # Get Pinecone API key from environment variable
    pinecone_api_key = os.environ.get('PINECONE_API_KEY')
    if not pinecone_api_key:
        logger.error("PINECONE_API_KEY environment variable not set")
        sys.exit(1)

    # Load the renewable energy client profile
    profile_path = Path(__file__).parent.parent / "profiles" / "renewable_energy_client.json"
    try:
        with open(profile_path, 'r') as f:
            profile_data = json.load(f)
            logger.info(f"Loaded profile for user: {profile_data['user_id']}")
    except Exception as e:
        logger.error(f"Error loading profile: {e}")
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
    
    # Initialize the personalized recommender
    recommender = PersonalizedRecommender(
        api_key=pinecone_api_key,
        index_name="eu-legal-documents-legal-bert",
        embedder_model="nlpaueb/legal-bert-base-uncased",
        feature_processor=feature_processor,
        profile_weight=0.7,  # Give more weight to the profile
        query_weight=0.3     # Less weight to the query
    )
    
    # Get user ID
    user_id = profile_data['user_id']
    
    # Create user profile from the loaded data
    profile = profile_data['profile']
    
    # 1. Create expert profile
    if 'expert_profile' in profile and 'description' in profile['expert_profile']:
        expert_description = profile['expert_profile']['description']
        logger.info(f"Creating expert profile from description: {expert_description[:50]}...")
        recommender.create_expert_profile(user_id, expert_description)
    
    # 2. Add historical documents
    if 'historical_documents' in profile:
        historical_docs = profile['historical_documents']
        logger.info(f"Adding {len(historical_docs)} historical documents to profile")
        for doc_id in historical_docs:
            recommender.add_historical_document(user_id, doc_id)
    
    # 3. Set categorical preferences
    if 'categorical_preferences' in profile:
        categorical_prefs = profile['categorical_preferences']
        logger.info(f"Setting categorical preferences: {list(categorical_prefs.keys())}")
        recommender.set_categorical_preferences(user_id, categorical_prefs)
    
    # Try different queries
    queries = [
        "renewable energy targets 2030",
        "offshore wind farm regulations",
        "energy storage integration",
        "solar panel grid connection requirements",
        "carbon pricing impact on renewable energy"
    ]
    
    # Get recommendations for each query
    for query in queries:
        print(f"\n\n{'='*80}\nGetting recommendations for query: '{query}'\n{'='*80}")
        
        recommendations = recommender.get_personalized_recommendations(
            user_id=user_id,
            query_text=query,
            top_k=5
        )
        
        # Display recommendations
        print(f"\nFound {len(recommendations)} personalized recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. Document ID: {rec['id']} (Score: {rec['score']:.4f})")
            print(f"   Text Similarity: {rec['text_score']:.4f}")
            
            if rec['categorical_score'] is not None:
                print(f"   Categorical Similarity: {rec['categorical_score']:.4f}")
            
            if 'metadata' in rec and rec['metadata']:
                # Display CELEX number
                if 'celex_number' in rec['metadata']:
                    print(f"   - CELEX: {rec['metadata']['celex_number']}")
                
                # Display title
                if 'title' in rec['metadata']:
                    title = rec['metadata']['title']
                    if len(title) > 100:
                        title = title[:100] + "..."
                    print(f"   - Title: {title}")
                
                # Display document type
                if 'document_type' in rec['metadata']:
                    print(f"   - Type: {rec['metadata']['document_type']}")
                
                # Display subject matters
                if 'subject_matters' in rec['metadata']:
                    subjects = rec['metadata']['subject_matters']
                    if isinstance(subjects, list) and len(subjects) > 3:
                        subjects = subjects[:3] + ["..."]
                    print(f"   - Subjects: {subjects}")

    # Try getting recommendations without a query (pure profile-based)
    print(f"\n\n{'='*80}\nGetting recommendations based only on user profile (no query)\n{'='*80}")
    
    recommendations = recommender.get_personalized_recommendations(
        user_id=user_id,
        query_text=None,  # No query, just use the profile
        top_k=5
    )
    
    # Display recommendations
    print(f"\nFound {len(recommendations)} profile-based recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. Document ID: {rec['id']} (Score: {rec['score']:.4f})")
        
        if 'metadata' in rec and rec['metadata']:
            # Display CELEX number
            if 'celex_number' in rec['metadata']:
                print(f"   - CELEX: {rec['metadata']['celex_number']}")
            
            # Display title
            if 'title' in rec['metadata']:
                title = rec['metadata']['title']
                if len(title) > 100:
                    title = title[:100] + "..."
                print(f"   - Title: {title}")
            
            # Display document type
            if 'document_type' in rec['metadata']:
                print(f"   - Type: {rec['metadata']['document_type']}")

if __name__ == "__main__":
    main()
