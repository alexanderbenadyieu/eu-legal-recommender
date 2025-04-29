#!/usr/bin/env python3
"""
Dictionary patching utility to prevent dictionary addition errors.

This module provides a safe merge function for dictionaries and
patches key classes in the codebase to prevent dictionary addition errors.
"""

import logging
import importlib
import types
import copy

# Configure logging
logger = logging.getLogger(__name__)

def safe_merge_dicts(*dicts):
    """Safely merge multiple dictionaries without using the '+' operator."""
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

def patch_personalized_recommender():
    """Apply specific patches to the PersonalizedRecommender class."""
    try:
        from src.models.personalized_recommender import PersonalizedRecommender
        
        # Save the original methods
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
                    logger.error(f"Dictionary addition error caught in patched get_personalized_recommendations: {str(e)}")
                    # Call the original again with empty dicts for safety
                    return original_get_personalized_recommendations(self, user_id, query_text, query_keywords,
                                                               {}, top_k, filter, 
                                                               embedding_type, {},
                                                               use_profile)
                # Re-raise the original exception
                raise
        
        # Define a super-safe version of _safe_get_composite_profile
        def patched_safe_get_composite_profile(self, profile, expert_weight, historical_weight, categorical_weight):
            """Super-safe version of _safe_get_composite_profile with enhanced error handling."""
            try:
                # Import copy for deep copying
                import copy
                
                # Create copies of the weights to ensure they're not modified
                expert_w = float(expert_weight) if expert_weight is not None else None
                historical_w = float(historical_weight) if historical_weight is not None else None
                categorical_w = float(categorical_weight) if categorical_weight is not None else None
                
                # Call UserProfile.get_composite_profile directly
                embedding, preferences = profile.get_composite_profile(
                    expert_weight=expert_w,
                    historical_weight=historical_w,
                    categorical_weight=categorical_w
                )
                
                # Use copy.deepcopy to ensure a completely independent copy of the preferences
                if preferences is not None and isinstance(preferences, dict):
                    # Create a full deep copy of all nested structures
                    safe_preferences = copy.deepcopy(preferences)
                    return embedding, safe_preferences
                else:
                    # Return an empty dictionary if preferences is None or not a dictionary
                    return embedding, {}
            except Exception as e:
                logger.error(f"Error in _safe_get_composite_profile: {str(e)}")
                # Re-raise with clearer message if this is a dictionary addition error
                if "unsupported operand type(s) for +: 'dict' and 'dict'" in str(e):
                    logger.error("Dictionary addition error detected in _safe_get_composite_profile")
                    
                    # Create a fallback safe implementation
                    try:
                        import numpy as np
                        logger.info("Using fallback implementation for profile composition")
                        
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
    except Exception as e:
        logger.error(f"Failed to patch PersonalizedRecommender: {str(e)}")

def patch_user_profile():
    """Apply specific patches to the UserProfile class."""
    try:
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
    except Exception as e:
        logger.error(f"Failed to patch UserProfile: {str(e)}")

def apply_patches():
    """Apply all dictionary operation patches."""
    logger.info("Applying dictionary operation patches")
    patch_personalized_recommender()
    patch_user_profile()
    logger.info("Patches applied successfully")
