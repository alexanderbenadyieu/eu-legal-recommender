"""
Personalized recommender system for EU legal documents.

This module extends the PineconeRecommender to provide personalized recommendations
based on user profiles that capture client interests and preferences. It integrates
expert-curated profiles, historical document engagement, and categorical preferences
to deliver tailored recommendations for legal professionals.
"""
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import json
from pathlib import Path

from .pinecone_recommender import PineconeRecommender
from .user_profile import UserProfile
from .embeddings import BERTEmbedder
from .features import FeatureProcessor

# Import configuration
from src.config import USER_PROFILE, SIMILARITY, PINECONE, EMBEDDER, LOGS_DIR
from src.utils.weight_config import WeightConfig

# Import utilities
from src.utils.logging import get_logger
from src.utils.exceptions import (
    RecommenderError, ValidationError, PineconeError, 
    EmbeddingError, RecommendationError, ProfileError
)

# Set up logging
logger = get_logger(__name__)

class PersonalizedRecommender:
    """Provide personalized legal document recommendations based on user profiles."""
    
    def __init__(
        self,
        api_key: str,
        index_name: str = None,
        embedder_model: str = None,
        feature_processor: Optional[FeatureProcessor] = None,
        text_weight: float = None,
        categorical_weight: float = None,
        db_path: Optional[str] = None,
        profile_weight: float = None,
        query_weight: float = None,
        expert_weight: float = None,
        historical_weight: float = None,
        categorical_preference_weight: float = None,
        weight_config_path: Optional[str] = None
    ):
        """
        Initialize personalized recommender.
        
        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index to use. Defaults to config value.
            embedder_model: Name of the BERT model to use. Defaults to config value.
            feature_processor: Optional FeatureProcessor for categorical features
            text_weight: Weight for text similarity (0-1). Defaults to config value.
            categorical_weight: Weight for categorical similarity (0-1). Defaults to config value.
            db_path: Optional path to SQLite database for storing user profiles
            profile_weight: Weight for user profile in scoring (0-1). Defaults to config value.
            query_weight: Weight for query in scoring (0-1). Defaults to config value.
            expert_weight: Weight for expert profile component (0-1). Defaults to config value.
            historical_weight: Weight for historical documents component (0-1). Defaults to config value.
            categorical_preference_weight: Weight for categorical preferences (0-1). Defaults to config value.
            
        Raises:
            ValidationError: If API key is missing or invalid
            ConfigurationError: If configuration values are invalid
            PineconeError: If there's an error connecting to Pinecone
            EmbeddingError: If there's an error initializing the embedder
        """
        logger.info(f"Initializing PersonalizedRecommender with model {embedder_model or EMBEDDER['model_name']}")
        
        # Validate API key
        if not api_key or not isinstance(api_key, str):
            logger.error("Missing or invalid Pinecone API key")
            raise ValidationError("Pinecone API key must be a non-empty string", 
                                field="api_key", 
                                code="INVALID_API_KEY")
        
        # Initialize weight configuration
        self.weight_config = WeightConfig(weight_config_path)
        
        # Initialize user profiles collection
        self._user_profiles = {}
        
        # Get weights from config or parameters
        similarity_weights = self.weight_config.get_weights("similarity")
        personalization_weights = self.weight_config.get_weights("personalization")
        profile_component_weights = self.weight_config.get_weights("profile_components")
        
        # Use configuration values if not provided
        self.index_name = index_name or PINECONE['index_name']
        self.embedder_model = embedder_model or EMBEDDER['model_name']
        
        # Set similarity weights (from parameters, config, or defaults)
        self.text_weight = text_weight or similarity_weights.get("text_weight", SIMILARITY['text_weight'])
        self.categorical_weight = categorical_weight or similarity_weights.get("categorical_weight", SIMILARITY['categorical_weight'])
        
        # Set personalization weights
        self.profile_weight = profile_weight or personalization_weights.get("profile_weight", USER_PROFILE['profile_weight'])
        self.query_weight = query_weight or personalization_weights.get("query_weight", USER_PROFILE['query_weight'])
        
        # Set profile component weights
        self.expert_weight = expert_weight or profile_component_weights.get("expert_weight", USER_PROFILE['expert_weight'])
        self.historical_weight = historical_weight or profile_component_weights.get("historical_weight", USER_PROFILE['historical_weight'])
        self.categorical_preference_weight = categorical_preference_weight or profile_component_weights.get("categorical_preference_weight", USER_PROFILE['categorical_weight'])
        
        # Validate weights
        if self.profile_weight + self.query_weight != 1.0:
            logger.warning(f"Profile weight ({self.profile_weight}) and query weight ({self.query_weight}) don't sum to 1.0. Normalizing.")
            total = self.profile_weight + self.query_weight
            self.profile_weight /= total
            self.query_weight /= total
            
        if self.expert_weight + self.historical_weight + self.categorical_preference_weight != 1.0:
            logger.warning(f"Profile component weights don't sum to 1.0. Normalizing.")
            total = self.expert_weight + self.historical_weight + self.categorical_preference_weight
            self.expert_weight /= total
            self.historical_weight /= total
            self.categorical_preference_weight /= total
        
        # Validate weight ranges
        for name, value in [
            ("text_weight", self.text_weight),
            ("categorical_weight", self.categorical_weight),
            ("profile_weight", self.profile_weight),
            ("query_weight", self.query_weight),
            ("expert_weight", self.expert_weight),
            ("historical_weight", self.historical_weight),
            ("categorical_preference_weight", self.categorical_preference_weight)
        ]:
            if not 0 <= value <= 1:
                logger.error(f"Invalid weight value for {name}: {value}")
                raise ValidationError(f"Weight must be between 0 and 1", 
                                    field=name, 
                                    code="INVALID_WEIGHT_RANGE")
                
        # Update weight config with current values
        self._update_weight_config()
            
        try:
            # Ensure feature_processor exists
            if feature_processor is None:
                feature_processor = FeatureProcessor()
            
            # Initialize base recommender
            self.recommender = PineconeRecommender(
                api_key=api_key,
                index_name=self.index_name,
                embedder_model=self.embedder_model,
                feature_processor=feature_processor,
                text_weight=self.text_weight,
                categorical_weight=self.categorical_weight
            )
            
            # Store additional parameters
            self.db_path = db_path
            self.embedder = self.recommender.embedder
            self.feature_processor = self.recommender.feature_processor
            
            logger.info(f"Initialized PersonalizedRecommender with profile_weight={self.profile_weight:.2f}, "
                      f"query_weight={self.query_weight:.2f}")
            
            # User profile cache
            self._user_profiles = {}
            
            # Store weight config path
            self.weight_config_path = weight_config_path
            
            # Save initial weights if path provided
            if self.weight_config_path:
                self.weight_config.save_config(self.weight_config_path)
            
        except PineconeError as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            raise
        except EmbeddingError as e:
            logger.error(f"Error initializing embedder: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during initialization: {str(e)}")
            raise RecommenderError(f"Failed to initialize recommender: {str(e)}", 
                                 code="INIT_ERROR")
    
    def _update_weight_config(self) -> None:
        """
        Update weight configuration with current weight values.
        
        This internal method synchronizes the weight_config object with the
        current weight values stored in instance attributes.
        """
        # Update similarity weights
        similarity_weights = {
            "text_weight": self.text_weight,
            "categorical_weight": self.categorical_weight
        }
        self.weight_config.set_weights(similarity_weights, "similarity")
        
        # Update personalization weights
        personalization_weights = {
            "profile_weight": self.profile_weight,
            "query_weight": self.query_weight
        }
        self.weight_config.set_weights(personalization_weights, "personalization")
        
        # Update profile component weights
        profile_component_weights = {
            "expert_weight": self.expert_weight,
            "historical_weight": self.historical_weight,
            "categorical_preference_weight": self.categorical_preference_weight
        }
        self.weight_config.set_weights(profile_component_weights, "profile_components")
        
        logger.debug("Updated weight configuration with current values")
    
    def set_weights(self, **weights) -> None:
        """
        Set multiple weights across different components.
        
        This method allows setting weights for different components of the recommender
        system in a single call. It validates and normalizes weights as needed.
        
        Args:
            **weights: Keyword arguments for different weights
                Supported weights:
                - text_weight: Weight for text similarity
                - categorical_weight: Weight for categorical similarity
                - profile_weight: Weight for user profile in scoring
                - query_weight: Weight for query in scoring
                - expert_weight: Weight for expert profile component
                - historical_weight: Weight for historical documents component
                - categorical_preference_weight: Weight for categorical preferences
                
        Raises:
            ValidationError: If any weight is invalid
        """
        # Track which weight categories were modified
        modified_categories = set()
        
        # Process similarity weights
        similarity_weights = {}
        if "text_weight" in weights:
            similarity_weights["text_weight"] = weights["text_weight"]
            self.text_weight = weights["text_weight"]
            modified_categories.add("similarity")
            
        if "categorical_weight" in weights:
            similarity_weights["categorical_weight"] = weights["categorical_weight"]
            self.categorical_weight = weights["categorical_weight"]
            modified_categories.add("similarity")
            
        # Process personalization weights
        personalization_weights = {}
        if "profile_weight" in weights:
            personalization_weights["profile_weight"] = weights["profile_weight"]
            self.profile_weight = weights["profile_weight"]
            modified_categories.add("personalization")
            
        if "query_weight" in weights:
            personalization_weights["query_weight"] = weights["query_weight"]
            self.query_weight = weights["query_weight"]
            modified_categories.add("personalization")
            
        # Process profile component weights
        profile_component_weights = {}
        if "expert_weight" in weights:
            profile_component_weights["expert_weight"] = weights["expert_weight"]
            self.expert_weight = weights["expert_weight"]
            modified_categories.add("profile_components")
            
        if "historical_weight" in weights:
            profile_component_weights["historical_weight"] = weights["historical_weight"]
            self.historical_weight = weights["historical_weight"]
            modified_categories.add("profile_components")
            
        if "categorical_preference_weight" in weights:
            profile_component_weights["categorical_preference_weight"] = weights["categorical_preference_weight"]
            self.categorical_preference_weight = weights["categorical_preference_weight"]
            modified_categories.add("profile_components")
        
        # Validate and normalize weights
        if "similarity" in modified_categories:
            # Ensure text_weight and categorical_weight sum to 1.0
            if self.text_weight + self.categorical_weight != 1.0:
                logger.warning(f"Similarity weights don't sum to 1.0. Normalizing.")
                total = self.text_weight + self.categorical_weight
                self.text_weight /= total
                self.categorical_weight /= total
                similarity_weights["text_weight"] = self.text_weight
                similarity_weights["categorical_weight"] = self.categorical_weight
            
            # Apply to similarity computer
            if hasattr(self.recommender, "similarity_computer"):
                self.recommender.similarity_computer.text_weight = self.text_weight
                self.recommender.similarity_computer.categorical_weight = self.categorical_weight
                logger.info(f"Applied similarity weights: text={self.text_weight:.2f}, categorical={self.categorical_weight:.2f}")
        
        if "personalization" in modified_categories:
            # Ensure profile_weight and query_weight sum to 1.0
            if self.profile_weight + self.query_weight != 1.0:
                logger.warning(f"Personalization weights don't sum to 1.0. Normalizing.")
                total = self.profile_weight + self.query_weight
                self.profile_weight /= total
                self.query_weight /= total
                personalization_weights["profile_weight"] = self.profile_weight
                personalization_weights["query_weight"] = self.query_weight
                
            logger.info(f"Applied personalization weights: profile={self.profile_weight:.2f}, query={self.query_weight:.2f}")
        
        if "profile_components" in modified_categories:
            # Ensure profile component weights sum to 1.0
            total = self.expert_weight + self.historical_weight + self.categorical_preference_weight
            if total != 1.0:
                logger.warning(f"Profile component weights don't sum to 1.0. Normalizing.")
                self.expert_weight /= total
                self.historical_weight /= total
                self.categorical_preference_weight /= total
                profile_component_weights["expert_weight"] = self.expert_weight
                profile_component_weights["historical_weight"] = self.historical_weight
                profile_component_weights["categorical_preference_weight"] = self.categorical_preference_weight
                
            logger.info(f"Applied profile component weights: expert={self.expert_weight:.2f}, "
                      f"historical={self.historical_weight:.2f}, categorical={self.categorical_preference_weight:.2f}")
        
        # Update weight configuration
        if "similarity" in modified_categories:
            self.weight_config.set_weights(similarity_weights, "similarity")
            
        if "personalization" in modified_categories:
            self.weight_config.set_weights(personalization_weights, "personalization")
            
        if "profile_components" in modified_categories:
            self.weight_config.set_weights(profile_component_weights, "profile_components")
        
        # Save updated configuration if path provided
        if self.weight_config_path and modified_categories:
            self.weight_config.save_config(self.weight_config_path)
            logger.info(f"Saved updated weight configuration to {self.weight_config_path}")
    
    def set_similarity_weights(self, text_weight: float, categorical_weight: float) -> None:
        """
        Set weights for similarity computation.
        
        Args:
            text_weight: Weight for text similarity (0-1)
            categorical_weight: Weight for categorical similarity (0-1)
            
        Raises:
            ValidationError: If weights are invalid
        """
        self.set_weights(text_weight=text_weight, categorical_weight=categorical_weight)
    
    def set_personalization_weights(self, profile_weight: float, query_weight: float) -> None:
        """
        Set weights for personalization components.
        
        Args:
            profile_weight: Weight for user profile in scoring (0-1)
            query_weight: Weight for query in scoring (0-1)
            
        Raises:
            ValidationError: If weights are invalid
        """
        self.set_weights(profile_weight=profile_weight, query_weight=query_weight)
    
    def set_profile_component_weights(self, expert_weight: float, historical_weight: float, 
                                    categorical_preference_weight: float) -> None:
        """
        Set weights for profile components.
        
        Args:
            expert_weight: Weight for expert profile component (0-1)
            historical_weight: Weight for historical documents component (0-1)
            categorical_preference_weight: Weight for categorical preferences (0-1)
            
        Raises:
            ValidationError: If weights are invalid
        """
        self.set_weights(
            expert_weight=expert_weight,
            historical_weight=historical_weight,
            categorical_preference_weight=categorical_preference_weight
        )
    
    def load_weights(self, file_path: str) -> None:
        """
        Load weights from configuration file.
        
        Args:
            file_path: Path to weight configuration file (JSON)
            
        Raises:
            ValueError: If file cannot be loaded
        """
        try:
            # Load configuration
            self.weight_config.load_config(file_path)
            
            # Apply weights to recommender
            self.weight_config.apply_to_recommender(self)
            
            # Update instance attributes
            similarity_weights = self.weight_config.get_weights("similarity")
            personalization_weights = self.weight_config.get_weights("personalization")
            profile_component_weights = self.weight_config.get_weights("profile_components")
            
            # Update similarity weights
            if "text_weight" in similarity_weights:
                self.text_weight = similarity_weights["text_weight"]
            if "categorical_weight" in similarity_weights:
                self.categorical_weight = similarity_weights["categorical_weight"]
                
            # Update personalization weights
            if "profile_weight" in personalization_weights:
                self.profile_weight = personalization_weights["profile_weight"]
            if "query_weight" in personalization_weights:
                self.query_weight = personalization_weights["query_weight"]
                
            # Update profile component weights
            if "expert_weight" in profile_component_weights:
                self.expert_weight = profile_component_weights["expert_weight"]
            if "historical_weight" in profile_component_weights:
                self.historical_weight = profile_component_weights["historical_weight"]
            if "categorical_preference_weight" in profile_component_weights:
                self.categorical_preference_weight = profile_component_weights["categorical_preference_weight"]
            
            # Store path for future saves
            self.weight_config_path = file_path
            
            logger.info(f"Loaded weights from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading weights from {file_path}: {str(e)}")
            raise ValueError(f"Failed to load weights: {str(e)}")
    
    def save_weights(self, file_path: Optional[str] = None) -> None:
        """
        Save current weights to configuration file.
        
        Args:
            file_path: Path to save configuration (JSON). If None, uses the path from initialization.
            
        Raises:
            ValueError: If file cannot be saved
        """
        file_path = file_path or self.weight_config_path
        
        if not file_path:
            logger.warning("No file path provided for saving weights")
            return
        
        try:
            # Update weight config with current values
            self._update_weight_config()
            
            # Save to file
            self.weight_config.save_config(file_path)
            
            logger.info(f"Saved weights to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving weights to {file_path}: {str(e)}")
            raise ValueError(f"Failed to save weights: {str(e)}")
    
    def get_user_profile(self, user_id: str) -> UserProfile:
        """
        Get or create a user profile.
        
        Args:
            user_id: Unique identifier for the user/client
            
        Returns:
            UserProfile instance
            
        Raises:
            ValidationError: If user_id is invalid
            ProfileError: If there's an error creating or retrieving the profile
        """
        # Validate user_id
        if not user_id or not isinstance(user_id, str):
            logger.error("Invalid user ID provided")
            raise ValidationError("User ID must be a non-empty string", 
                                field="user_id", 
                                code="INVALID_USER_ID")
        
        try:
            # Create profile if it doesn't exist
            if user_id not in self._user_profiles:
                logger.info(f"Creating new profile for user {user_id}")
                self._user_profiles[user_id] = UserProfile(
                    user_id=user_id,
                    embedder=self.embedder,
                    feature_processor=self.feature_processor,
                    db_path=self.db_path
                )
            else:
                logger.debug(f"Using cached profile for user {user_id}")
            
            return self._user_profiles[user_id]
            
        except Exception as e:
            logger.error(f"Error creating or retrieving profile for user {user_id}: {str(e)}")
            raise ProfileError(f"Failed to get user profile: {str(e)}", 
                             profile_id=user_id, 
                             code="PROFILE_ACCESS_ERROR")
    
    def create_expert_profile(
        self,
        user_id: str,
        profile_text: str
    ) -> None:
        """
        Create an expert-curated profile for a user.
        
        Args:
            user_id: Unique identifier for the user/client
            profile_text: Detailed description of client's regulatory interests
            
        Raises:
            ValidationError: If user_id or profile_text is invalid
            ProfileError: If there's an error creating the expert profile
            EmbeddingError: If there's an error generating embeddings for the profile
        """
        # Validate inputs
        if not profile_text or not isinstance(profile_text, str):
            logger.error("Invalid profile text provided")
            raise ValidationError("Profile text must be a non-empty string", 
                                field="profile_text", 
                                code="INVALID_PROFILE_TEXT")
        
        try:
            # Get the user profile
            profile = self.get_user_profile(user_id)
            
            # Create the expert profile
            profile.create_expert_profile(profile_text)
            logger.info(f"Created expert profile for user {user_id}")
            
        except ValidationError:
            # Re-raise validation errors
            raise
        except EmbeddingError as e:
            logger.error(f"Error generating embeddings for expert profile: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error creating expert profile for user {user_id}: {str(e)}")
            raise ProfileError(f"Failed to create expert profile: {str(e)}", 
                             profile_id=user_id, 
                             code="EXPERT_PROFILE_ERROR")
    
    def _safe_merge_dicts(self, *dicts) -> Dict:
        """
        Safely merge multiple dictionaries without using the '+' operator.
        
        This utility method creates a new dictionary and copies all key-value pairs
        from the input dictionaries, avoiding dictionary addition errors.
        
        Args:
            *dicts: Variable number of dictionaries to merge
            
        Returns:
            A new dictionary containing all key-value pairs from the input dictionaries
        """
        # Import copy for deep copying
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

    def _safe_get_composite_profile(self, profile, expert_weight, historical_weight, categorical_weight):
        """
        A safer wrapper around the UserProfile.get_composite_profile method to avoid dictionary addition errors.
        
        This method ensures proper handling of the return values from get_composite_profile and avoids any
        direct dictionary operations that might lead to the '+' operator being used on dictionaries.
        
        Args:
            profile: UserProfile instance
            expert_weight: Weight for expert profile component
            historical_weight: Weight for historical documents
            categorical_weight: Weight for categorical preferences
            
        Returns:
            Tuple containing:
            - Composite embedding vector
            - Dictionary of categorical preferences (or empty dict if None)
        """
        try:
            # Import copy module for deep copying
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
            # This prevents any issues with nested dictionaries or shared references
            if preferences is not None and isinstance(preferences, dict):
                # Create a full deep copy of all nested structures 
                # Using our safe dictionary merge utility to ensure no + operations
                safe_preferences = self._safe_merge_dicts(preferences)
                return embedding, safe_preferences
            else:
                # Return an empty dictionary if preferences is None or not a dictionary
                return embedding, {}
        except Exception as e:
            logger.error(f"Error in _safe_get_composite_profile: {str(e)}")
            # Re-raise with clearer message if this is a dictionary addition error
            if "unsupported operand type(s) for +: 'dict' and 'dict'" in str(e):
                logger.error("Dictionary addition error detected in _safe_get_composite_profile")
                raise ValueError("Dictionary addition error in get_composite_profile. Check for any dictionary + operations.")
            # Otherwise, re-raise the original exception
            raise
    
    def add_historical_document(
        self,
        user_id: str,
        document_id: str
    ) -> None:
        """Add a historical document of interest to a user's profile.
        
        Args:
            user_id: Unique identifier for the user/client
            document_id: ID of the document (e.g., CELEX number)
            
        Raises:
            ValidationError: If user_id or document_id is invalid
            EmbeddingError: If there's an error generating embeddings
            ProfileError: If there's an error adding the document to the profile
        """
        # Validate inputs
        if not user_id or not isinstance(user_id, str):
            logger.error("Invalid user ID provided")
            raise ValidationError("User ID must be a non-empty string", 
                                field="user_id", 
                                code="INVALID_USER_ID")
        
        if not document_id:
            logger.error("Cannot add document with empty ID")
            raise ValueError("Document ID cannot be empty")
        
        try:
            # Get document data - this includes metadata
            document_data = self.recommender.get_document_by_id(document_id)
            
            if not document_data:
                # Document not found in Pinecone
                logger.error(f"Document not found in vector store: {document_id}")
                raise ValidationError(f"Document not found: {document_id}", 
                                     field="document_id",
                                     code="DOCUMENT_NOT_FOUND")
            
            # Add document to profile
            try:
                # Add by ID to historical documents
                profile = self.get_user_profile(user_id)
                profile.add_historical_document(document_id, document_data)
                logger.info(f"Added historical document {document_id} to profile for {user_id}")
                return True
            except Exception as e:
                logger.error(f"Error adding historical document {document_id} to profile for {user_id}: {str(e)}")
                raise ProfileError(f"Failed to add historical document: {str(e)}", 
                                 profile_id=user_id, 
                                 code="HISTORICAL_DOC_ERROR")
        except Exception as e:
            logger.error(f"Error retrieving or adding document {document_id}: {str(e)}")
            raise ValidationError(f"Document not found: {document_id}", 
                                   field="document_id",
                                   code="DOCUMENT_NOT_FOUND")
    
    def set_categorical_preferences(
        self,
        user_id: str,
        categorical_preferences: Dict[str, Dict[str, float]]
    ) -> None:
        """
        Set categorical preferences for a user profile.
        
        Args:
            user_id: Unique identifier for the user/client
            categorical_preferences: Dictionary mapping category types to their preference dictionaries
                Example: {
                    "form": {"regulation": 0.8, "directive": 0.5},
                    "subject_matters": {"Agriculture": 1.0, "Environment": 0.7},
                    "eurovoc_descriptors": {"renewable energy": 1.0, "climate change": 0.8}
                }
            
        Raises:
            ValidationError: If user_id or categorical_preferences are invalid
            ProfileError: If there's an error updating the profile
        """
        # Validate inputs
        if not user_id or not isinstance(user_id, str):
            logger.error("Invalid user ID provided")
            raise ValidationError("User ID must be a non-empty string", 
                                field="user_id", 
                                code="INVALID_USER_ID")
        
        if not isinstance(categorical_preferences, dict):
            logger.error("Invalid categorical preferences format")
            raise ValidationError("Categorical preferences must be a dictionary", 
                                field="categorical_preferences", 
                                code="INVALID_CATEGORICAL_PREFS")
        
        # Get or create profile
        profile = self.get_user_profile(user_id)
        
        try:
            # Set categorical preferences in the profile
            profile.set_categorical_preferences(categorical_preferences)
            
            # Count total preferences
            pref_count = sum(len(prefs) for cat, prefs in categorical_preferences.items() 
                            if isinstance(prefs, dict))
            
            logger.info(f"Set {pref_count} categorical preferences for {user_id}")
            
        except Exception as e:
            logger.error(f"Error setting categorical preferences for {user_id}: {str(e)}")
            if isinstance(e, (ValidationError, ProfileError)):
                # Re-raise known error types
                raise
            else:
                # Wrap other errors
                raise ProfileError(f"Failed to set categorical preferences: {str(e)}", 
                                profile_id=user_id, 
                                code="SET_PREFERENCES_ERROR")
    
    def get_recommendations_by_id(
        self,
        document_id: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_categorical: bool = True,
        client_preferences: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """
        Get personalized document recommendations based on a similar document ID.
        
        This method delegates to the underlying PineconeRecommender's get_recommendations_by_id
        method, but allows for client preferences to be applied to the results.
        
        Args:
            document_id: ID of the document to find similar documents for
            top_k: Number of recommendations to return
            filter: Optional Pinecone metadata filter
            include_categorical: Whether to include categorical features in similarity calculation
            client_preferences: Optional dictionary of client preferences for categorical features
            
        Returns:
            List of recommended documents with scores and metadata
            
        Raises:
            ValidationError: If document_id is invalid
            PineconeError: If there's an error querying Pinecone
            RecommendationError: If there's an error generating recommendations
        """
        logger.info(f"Getting recommendations for document ID: {document_id}")
        
        # Delegate to the base recommender's get_recommendations_by_id method
        try:
            return self.recommender.get_recommendations_by_id(
                document_id=document_id,
                top_k=top_k,
                filter=filter,
                include_categorical=include_categorical,
                client_preferences=client_preferences
            )
        except Exception as e:
            logger.error(f"Error getting recommendations by document ID: {str(e)}")
            raise
    
    def get_personalized_recommendations(
        self,
        user_id: str,
        query_text: Optional[str] = None,
        query_keywords: Optional[List[str]] = None,
        query_features: Optional[Dict[str, Union[str, List[str]]]] = None,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        embedding_type: str = 'combined',
        component_weights: Optional[Dict[str, float]] = None,
        use_profile: bool = True
    ) -> List[Dict]:
        """
        Get personalized document recommendations based on user profile and query.
        
        Args:
            user_id: Unique identifier for the user/client
            query_text: Optional query text (if None, uses only the user profile)
            query_keywords: Optional list of keywords to enhance the query
            query_features: Optional dictionary of categorical features for the query
            top_k: Number of recommendations to return
            filter: Optional Pinecone metadata filter
            embedding_type: Type of embedding to use ('combined', 'summary', or 'keyword')
            component_weights: Optional dictionary to override default component weights
                              {'profile_weight', 'query_weight', 'expert_weight', 
                               'historical_weight', 'categorical_weight'}
            use_profile: Whether to use the user profile for recommendations (default: True)
            
        Returns:
            List of recommended documents with scores and metadata
            
        Raises:
            ValidationError: If input parameters are invalid
            ProfileError: If there's an error retrieving or using the user profile
            EmbeddingError: If there's an error generating embeddings
            RecommendationError: If there's an error generating recommendations
            PineconeError: If there's an error querying Pinecone
        """
        # Validate inputs
        if not user_id or not isinstance(user_id, str):
            logger.error("Invalid user ID provided")
            raise ValidationError("User ID must be a non-empty string", field="user_id", code="INVALID_USER_ID")
        
        # Initialize weights based on component_weights or defaults
        profile_weight = self.profile_weight
        query_weight = self.query_weight
        expert_weight = self.expert_weight
        historical_weight = self.historical_weight
        categorical_weight = self.categorical_preference_weight
        
        # Allow overriding of weights via component_weights parameter
        if component_weights and isinstance(component_weights, dict):
            if 'profile_weight' in component_weights:
                profile_weight = component_weights['profile_weight']
            if 'query_weight' in component_weights:
                query_weight = component_weights['query_weight']
            if 'expert_weight' in component_weights:
                expert_weight = component_weights['expert_weight']
            if 'historical_weight' in component_weights:
                historical_weight = component_weights['historical_weight']
            if 'categorical_weight' in component_weights:
                categorical_weight = component_weights['categorical_weight']
        
        # Check if we should use the profile
        if not use_profile:
            profile_weight = 0.0
            query_weight = 1.0
        
        # Validate inputs
        if profile_weight > 0 and not self._user_profiles.get(user_id):
            raise ProfileError(f"User profile not found: {user_id}")
                
        if profile_weight == 0 and not query_text and not query_keywords:
            raise ValidationError("Cannot generate recommendations without either a user profile or a query")
                
        # Process personalized recommendations
        try:
            profile = self._user_profiles.get(user_id) if profile_weight > 0 else None
            
            # Generate query embedding if we have query text or keywords
            query_embedding = None
            if query_text or query_keywords:
                try:
                    # Generate query embedding
                    query_embedding = self.recommender.generate_embedding(
                        text=query_text,
                        keywords=query_keywords,
                        embedding_type=embedding_type
                    )
                except Exception as e:
                    logger.error(f"Error generating query embedding: {str(e)}")
                    if isinstance(e, EmbeddingError):
                        raise
                    else:
                        raise EmbeddingError(f"Failed to generate query embedding: {str(e)}")
                    
                # If no profile or profile weight is 0, just use query embedding
                if not profile or profile_weight == 0:
                    logger.info(f"Using only query embedding for user {user_id}")
                    return self.recommender.get_recommendations_with_embedding(
                        query_embedding=query_embedding,
                        query_text=query_text or "[Keywords Only]",
                        query_keywords=query_keywords,
                        query_features=query_features,
                        top_k=top_k,
                        filter=filter,
                        embedding_type=embedding_type
                    )
                
                # If we reach here, we need to blend the query with the profile
                logger.info(f"Blending query and profile for user {user_id} with weights: "
                          f"profile={profile_weight:.2f}, query={query_weight:.2f}")
                
                # Get profile embedding and preferences
                logger.info(f"Getting composite profile for user {user_id}")
                profile_embedding, profile_preferences = self._safe_get_composite_profile(
                    profile,
                    expert_weight,
                    historical_weight,
                    categorical_weight
                )
                
                # Use the _safe_merge_dicts utility to safely merge profile preferences and query features
                # This ensures no dictionary addition errors occur and creates a completely clean, deep copy
                logger.debug(f"Using profile preferences: {list(profile_preferences.keys()) if profile_preferences else []}")
                logger.debug(f"Using query features: {list(query_features.keys()) if query_features else []}")
                
                # Safely merge the dictionaries using our utility method
                merged_features = self._safe_merge_dicts(profile_preferences, query_features)
                
                logger.debug(f"Successfully merged features with keys: {list(merged_features.keys())}")
                
                # Blend the profile and query embeddings
                blended_embedding = (
                    profile_embedding * profile_weight + 
                    query_embedding * query_weight
                )
                
                # Normalize the blended embedding
                norm = np.linalg.norm(blended_embedding)
                if norm > 0:
                    blended_embedding = blended_embedding / norm
                
                # Get recommendations using the blended embedding
                try:
                    return self.recommender.get_recommendations_with_embedding(
                        query_embedding=blended_embedding,
                        query_text=query_text or "[Personalized]",
                        query_keywords=query_keywords,
                        query_features=merged_features,
                        top_k=top_k,
                        filter=filter,
                        embedding_type=embedding_type
                    )
                except Exception as e:
                    logger.error(f"Error generating or using query embedding: {str(e)}")
                    if isinstance(e, EmbeddingError):
                        raise
                    else:
                        raise EmbeddingError(f"Failed to process query embedding: {str(e)}")
            else:
                # Handle case with no query (profile only)
                logger.info(f"Using profile only for recommendations")
                profile_embedding, profile_preferences = self._safe_get_composite_profile(
                    profile,
                    expert_weight,
                    historical_weight,
                    categorical_weight
                )
                
                # Get recommendations with only profile embedding
                # Ensure the profile_preferences are safely copied to prevent dictionary addition errors
                safe_preferences = self._safe_merge_dicts(profile_preferences)
                logger.debug(f"Using profile-only preferences with keys: {list(safe_preferences.keys()) if safe_preferences else []}")
                
                return self.recommender.get_recommendations_with_embedding(
                    query_embedding=profile_embedding,
                    query_text="[User Profile Only]",
                    query_features=safe_preferences,
                    top_k=top_k,
                    filter=filter,
                    embedding_type=embedding_type
                )
                
        except ProfileError as e:
            logger.error(f"Error with user profile: {str(e)}")
            raise
        except EmbeddingError:
            # Re-raise embedding errors
            raise
        except PineconeError:
            # Re-raise Pinecone errors
            raise
        except RecommendationError:
            # Re-raise recommendation errors
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating personalized recommendations for user {user_id}: {str(e)}")
            # Check if this is a dictionary addition error and provide more specific error message
            if "unsupported operand type(s) for +: 'dict' and 'dict'" in str(e):
                logger.error(f"Dictionary addition error detected in get_personalized_recommendations. This is likely caused by improper dictionary merging.")
                raise RecommendationError(f"Dictionary merging error in recommendation generation. Please use .update() instead of + for dictionaries", 
                                        query=query_text[:50] if query_text else "[User Profile]", 
                                        code="DICT_ADDITION_ERROR")
            else:
                # Re-raise with original error message
                raise RecommendationError(f"Failed to generate personalized recommendations: {str(e)}", 
                                        query=query_text[:50] if query_text else "[User Profile]", 
                                        code="PERSONALIZED_REC_ERROR")

    def get_recommendations(self, query: str, limit: int = 10, **kwargs) -> List[Dict]:
        """
        Get recommendations for evaluation purposes.
        
        This method is specifically designed to work with the evaluation framework.
        It handles different scenarios:
        1. If query matches a client ID exactly (e.g., 'renewable_energy_client'), it will use
           profile-based recommendations with no additional query text.
        2. If query is empty or None, it will use profile-based recommendations.
        3. Otherwise, it will use regular personalized recommendations combining query and profile.
        
        The method supports profile configuration through component_weights parameter:
        - To test with expert profile only: component_weights={'expert_weight': 1.0, 'historical_weight': 0.0, 'categorical_weight': 0.0}
        - To test with historical docs only: component_weights={'expert_weight': 0.0, 'historical_weight': 1.0, 'categorical_weight': 0.0}
        - To test with categorical prefs only: component_weights={'expert_weight': 0.0, 'historical_weight': 0.0, 'categorical_weight': 1.0}
        
        Args:
            query: Query string or client ID
            limit: Number of recommendations to return
            **kwargs: Additional arguments for get_personalized_recommendations
            
        Returns:
            List of recommended documents with scores and metadata
        """
        # Special handling for evaluation use cases
        if query in self._user_profiles or not query or query.strip() == '':
            # This is likely a client ID or an empty query for evaluation
            # Use pure profile-based recommendations (no additional query text)
            user_id = query if query else kwargs.get('user_id', '')
            if not user_id and '_client' in query:
                # If we have a client_id but no user_id in kwargs, use query as user_id
                user_id = query
                
            # If we have 'evaluation_mode' in kwargs, we'll use specific profile configurations
            evaluation_mode = kwargs.pop('evaluation_mode', None)
            component_weights = kwargs.get('component_weights', {})
            
            if evaluation_mode == 'expert_only':
                component_weights = {'expert_weight': 1.0, 'historical_weight': 0.0, 'categorical_weight': 0.0}
            elif evaluation_mode == 'historical_only':
                component_weights = {'expert_weight': 0.0, 'historical_weight': 1.0, 'categorical_weight': 0.0}
            elif evaluation_mode == 'categorical_only':
                component_weights = {'expert_weight': 0.0, 'historical_weight': 0.0, 'categorical_weight': 1.0}
            
            # Create a new dictionary for the arguments to pass to get_personalized_recommendations
            # This ensures we don't modify the original kwargs and prevents dictionary addition errors
            new_kwargs = kwargs.copy()
            
            # If component_weights is provided, add it to the new kwargs dictionary
            if component_weights:
                new_kwargs['component_weights'] = component_weights.copy()
                
            logger.info(f"Using profile-based recommendations for evaluation of '{user_id}'")
            return self.get_personalized_recommendations(
                user_id=user_id,
                query_text=None,  # No additional query text, use profile only
                top_k=limit,
                **new_kwargs
            )
        else:
            # Regular use case - treat as normal query with personalization
            # Create a copy of kwargs to prevent dictionary addition errors
            new_kwargs = kwargs.copy()
            return self.get_personalized_recommendations(user_id=query, top_k=limit, **new_kwargs)
