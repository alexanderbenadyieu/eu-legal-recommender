#!/usr/bin/env python3
"""
Debug script to identify and fix dictionary operations in categorical preferences.
"""

import os
import sys
import json
import logging
import numpy as np
import copy
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

# Import directly from modules
from src.models.embeddings import BERTEmbedder
from src.models.user_profile import UserProfile
from src.models.features import FeatureProcessor

def debug_dict_operations(d1, d2):
    """Test dictionary operations and demonstrate safe ways to merge them."""
    logger.info("\n=== DEBUGGING DICTIONARY OPERATIONS ===")
    
    # Print original dictionaries
    logger.info(f"Dictionary 1: {d1}")
    logger.info(f"Dictionary 2: {d2}")
    
    # Try addition (will fail)
    try:
        result = d1 + d2
        logger.info("Dictionary addition worked (unexpected!)")
    except TypeError as e:
        logger.info(f"As expected, dictionary addition fails: {str(e)}")
    
    # Safe merge method 1: update()
    result1 = {}
    result1.update(d1)
    result1.update(d2)
    logger.info(f"Safe merge using update(): {result1}")
    
    # Safe merge method 2: manual copy
    result2 = {}
    for k, v in d1.items():
        result2[k] = v
    for k, v in d2.items():
        result2[k] = v
    logger.info(f"Safe merge using manual copy: {result2}")
    
    # Safe merge method 3: deepcopy
    result3 = copy.deepcopy(d1)
    for k, v in d2.items():
        result3[k] = copy.deepcopy(v)
    logger.info(f"Safe merge using deepcopy: {result3}")
    
    return result3

def patch_user_profile_class():
    """Monkey patch the UserProfile class to print detailed debug info in get_composite_profile."""
    original_get_composite_profile = UserProfile.get_composite_profile
    
    def debug_get_composite_profile(self, expert_weight=None, historical_weight=None, categorical_weight=None):
        logger.info("\n=== DEBUG GET_COMPOSITE_PROFILE ===")
        logger.info(f"Input weights: expert={expert_weight}, historical={historical_weight}, categorical={categorical_weight}")
        
        # Debug the categorical preferences
        if self.categorical_preferences:
            logger.info(f"Categorical preferences: {self.categorical_preferences}")
            for key, value in self.categorical_preferences.items():
                logger.info(f"Key: {key}, Type: {type(value)}")
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        logger.info(f"  Subkey: {subkey}, Type: {type(subvalue)}, Value: {subvalue}")
        
        try:
            # Call the original method
            result = original_get_composite_profile(self, expert_weight, historical_weight, categorical_weight)
            logger.info("get_composite_profile executed successfully")
            return result
        except Exception as e:
            logger.error(f"Error in get_composite_profile: {str(e)}")
            if "unsupported operand type(s) for +" in str(e):
                logger.error("Dictionary addition error detected - check categorical preferences handling")
            raise
    
    # Apply the monkey patch
    UserProfile.get_composite_profile = debug_get_composite_profile
    logger.info("Patched UserProfile.get_composite_profile with debug version")

def main():
    # Load environment variables
    load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env', override=True)
    
    # Initialize embedder
    embedder = BERTEmbedder(model_name='nlpaueb/legal-bert-base-uncased')
    
    # Apply debug patch to UserProfile class
    patch_user_profile_class()
    
    # Create a test profile
    profile = UserProfile(
        user_id="debug_user",
        embedder=embedder
    )
    
    # Set expert profile
    profile.create_expert_profile("This is a test profile for debugging dictionary operations.")
    
    # Set categorical preferences with nested dictionaries
    profile.set_categorical_preferences({
        "form": {
            "regulation": 0.8,
            "directive": 0.6
        },
        "subject_matters": ["environment", "energy"],
        "authors": {
            "commission": 0.9,
            "parliament": 0.7
        }
    })
    
    # Debug dictionary operations
    d1 = {"a": 1, "b": {"x": 10, "y": 20}}
    d2 = {"b": {"z": 30}, "c": 3}
    
    merged = debug_dict_operations(d1, d2)
    logger.info(f"Final merged dictionary: {merged}")
    
    # Now test the composite profile generation
    try:
        composite_embedding, preferences = profile.get_composite_profile()
        logger.info(f"Successfully generated composite profile with preferences: {preferences}")
    except Exception as e:
        logger.error(f"Error generating composite profile: {str(e)}")
    
    # Test how composite profile is handled in a personalized recommender-like context
    try:
        # This reproduces the standard pattern that might lead to dictionary addition errors
        query_features = {"form": {"regulation": 0.9}, "new_key": "value"}
        
        # First attempt: manual merging (should work)
        merged_features = {}
        for k, v in preferences.items():
            if isinstance(v, dict):
                merged_features[k] = copy.deepcopy(v)
            else:
                merged_features[k] = v
                
        for k, v in query_features.items():
            if isinstance(v, dict):
                if k in merged_features and isinstance(merged_features[k], dict):
                    # For dictionaries, we need to handle keys that exist in both
                    for subk, subv in v.items():
                        merged_features[k][subk] = subv
                else:
                    merged_features[k] = copy.deepcopy(v)
            else:
                merged_features[k] = v
                
        logger.info(f"Successful manual merge: {merged_features}")
    except Exception as e:
        logger.error(f"Error merging dictionaries: {str(e)}")
        
if __name__ == "__main__":
    main()
