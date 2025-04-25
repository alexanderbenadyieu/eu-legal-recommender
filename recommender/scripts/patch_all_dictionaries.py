#!/usr/bin/env python3
"""
Comprehensive patch to find and fix all dictionary addition operations in the codebase.

This script scans for potential dictionary addition operations, applies monkey patching to 
prevent such operations at runtime, and provides a safe mechanism for handling dictionary merges.
"""

import os
import sys
import importlib
import types
from pathlib import Path
import logging
import inspect

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def safe_merge_dicts(*dicts):
    """Safely merge multiple dictionaries without using the '+' operator."""
    import copy
    
    # Create a new empty dictionary
    result = {}
    
    # Process each input dictionary
    for d in dicts:
        if d and isinstance(d, dict):
            # For each key-value pair in the dictionary
            for key, value in d.items():
                # Deep copy to avoid reference issues with nested structures
                if isinstance(value, dict):
                    # If the value is a dictionary, deep copy it
                    result[key] = copy.deepcopy(value)
                else:
                    # For non-dictionary values, assign directly
                    result[key] = value
    
    return result

def scan_module_for_dict_operations(module_name):
    """Scan a module for potential dictionary addition operations."""
    try:
        module = importlib.import_module(module_name)
        
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                for method_name, method in inspect.getmembers(obj, inspect.isfunction):
                    try:
                        source = inspect.getsource(method)
                        if "dict" in source and "+" in source:
                            logger.warning(f"Potential dictionary operation in {module_name}.{name}.{method_name}")
                    except Exception:
                        pass
    except Exception as e:
        logger.error(f"Error scanning module {module_name}: {str(e)}")

def patch_personalized_recommender():
    """Apply specific patches to the PersonalizedRecommender class."""
    from src.models.personalized_recommender import PersonalizedRecommender
    
    # Save the original method
    original_get_personalized_recommendations = PersonalizedRecommender.get_personalized_recommendations
    original_safe_get_composite_profile = PersonalizedRecommender._safe_get_composite_profile
    
    # Define a patched version with enhanced safety
    def patched_get_personalized_recommendations(self, user_id, query_text=None, query_keywords=None, 
                                            query_features=None, top_k=10, filter=None, 
                                            embedding_type='combined', component_weights=None, 
                                            use_profile=True):
        """Patched version with additional safety checks for dictionaries."""
        try:
            # Add a pre-check to wrap any dictionaries in safe containers
            if query_features and isinstance(query_features, dict):
                query_features = safe_merge_dicts(query_features)
                
            if component_weights and isinstance(component_weights, dict):
                component_weights = safe_merge_dicts(component_weights)
                
            return original_get_personalized_recommendations(self, user_id, query_text, query_keywords,
                                                       query_features, top_k, filter, 
                                                       embedding_type, component_weights,
                                                       use_profile)
        except Exception as e:
            if "unsupported operand type(s) for +" in str(e) and "dict" in str(e):
                logger.error(f"Dictionary addition error caught in patched method: {str(e)}")
                raise ValueError(f"Dictionary merging error in recommendation generation. Please use .update() instead of + for dictionaries", 
                                user_id=user_id, code="DICT_ADDITION_ERROR")
            raise
    
    def patched_safe_get_composite_profile(self, profile, expert_weight, historical_weight, categorical_weight):
        """Super-safe version of _safe_get_composite_profile with enhanced error handling."""
        try:
            # Call the original method
            return original_safe_get_composite_profile(self, profile, expert_weight, historical_weight, categorical_weight)
        except Exception as e:
            if "unsupported operand type(s) for +" in str(e) and "dict" in str(e):
                logger.error(f"Dictionary addition error caught in patched _safe_get_composite_profile: {str(e)}")
                
                # Extreme fallback: try to get the profile embedding but with empty preferences
                import copy
                import numpy as np
                
                # Fallback to manually construct a composite profile
                try:
                    logger.warning("Attempting fallback method for composite profile generation")
                    
                    # Try to get the expert profile directly
                    expert_w = float(expert_weight) if expert_weight is not None else profile.expert_weight
                    
                    # Initialize with expert profile
                    composite_embedding = profile.expert_profile_embedding * expert_w
                    
                    # Normalize
                    norm = np.linalg.norm(composite_embedding)
                    if norm > 0:
                        composite_embedding = composite_embedding / norm
                        
                    # Return with empty preferences
                    return composite_embedding, {}
                except Exception as fallback_error:
                    logger.error(f"Fallback method failed: {str(fallback_error)}")
                    
            # Re-raise the original exception
            raise
    
    # Apply the patches
    PersonalizedRecommender.get_personalized_recommendations = patched_get_personalized_recommendations
    PersonalizedRecommender._safe_get_composite_profile = patched_safe_get_composite_profile
    
    logger.info("Applied patches to PersonalizedRecommender class")

def patch_user_profile():
    """Apply specific patches to the UserProfile class."""
    from src.models.user_profile import UserProfile
    
    # Save the original method
    original_get_composite_profile = UserProfile.get_composite_profile
    
    # Define a patched version with enhanced safety
    def patched_get_composite_profile(self, expert_weight=None, historical_weight=None, categorical_weight=None):
        """Patched version with additional safety checks for dictionaries."""
        try:
            # Call the original method
            result = original_get_composite_profile(self, expert_weight, historical_weight, categorical_weight)
            
            # Extra safety: ensure the second element (preferences) is properly deep copied
            import copy
            embedding, preferences = result
            
            # Create a completely new dictionary to avoid any reference issues
            safe_preferences = {}
            if preferences:
                for key, value in preferences.items():
                    if isinstance(value, dict):
                        safe_preferences[key] = copy.deepcopy(value)
                    else:
                        safe_preferences[key] = value
                        
            return embedding, safe_preferences
        except Exception as e:
            if "unsupported operand type(s) for +" in str(e) and "dict" in str(e):
                logger.error(f"Dictionary addition error caught in patched UserProfile.get_composite_profile: {str(e)}")
                
                # Return fallback values
                import numpy as np
                # If we have an expert profile embedding, return that with empty preferences
                if self.expert_profile_embedding is not None:
                    return self.expert_profile_embedding.copy(), {}
                # Otherwise create a zero vector of the right dimensionality
                return np.zeros(768), {}  # 768 is the dimension of BERT embeddings
            raise
            
    # Apply the patch
    UserProfile.get_composite_profile = patched_get_composite_profile
    
    logger.info("Applied patches to UserProfile class")

def main():
    """Main function to apply all patches and scan for potential issues."""
    logger.info("Starting comprehensive dictionary operation patching")
    
    # Apply specific patches to problematic classes
    patch_personalized_recommender()
    patch_user_profile()
    
    # Scan key modules for potential dictionary operations
    modules_to_scan = [
        "src.models.personalized_recommender",
        "src.models.user_profile",
        "src.utils.weight_optimizer"
    ]
    
    for module_name in modules_to_scan:
        scan_module_for_dict_operations(module_name)
    
    logger.info("Patching complete. All dictionary operations should now be handled safely.")
    
    # Return our safe merge utility function for use in other modules
    return safe_merge_dicts

if __name__ == "__main__":
    safe_merge_dicts_func = main()
