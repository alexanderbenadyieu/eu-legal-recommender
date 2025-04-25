"""
User profile management for personalized legal document recommendations.

This module handles the creation, storage, and retrieval of user profiles
that capture client interests and preferences for EU legal documents.
It supports multiple profile components including expert-curated descriptions,
historical document engagement, and categorical preferences.
"""
from typing import List, Dict, Union, Optional, Any, Tuple
import numpy as np
import json
import logging
from pathlib import Path
import sqlite3
from datetime import datetime

from .embeddings import BERTEmbedder
from .features import FeatureProcessor

# Import configuration
from src.config import USER_PROFILE, LOGS_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / 'user_profile.log')
    ]
)
logger = logging.getLogger(__name__)

class UserProfile:
    """Manage user profiles for personalized recommendations."""
    
    def __init__(
        self,
        user_id: str,
        embedder: BERTEmbedder,
        feature_processor: Optional[FeatureProcessor] = None,
        db_path: Optional[str] = None,
        expert_weight: Optional[float] = None,
        historical_weight: Optional[float] = None,
        categorical_weight: Optional[float] = None
    ):
        """
        Initialize user profile manager.
        
        Args:
            user_id: Unique identifier for the user/client
            embedder: BERTEmbedder instance for generating text embeddings
            feature_processor: Optional FeatureProcessor for handling categorical features
            db_path: Optional path to SQLite database for storing profiles
            expert_weight: Weight for expert-curated profile. Defaults to config value.
            historical_weight: Weight for historical engagement. Defaults to config value.
            categorical_weight: Weight for categorical preferences. Defaults to config value.
        """
        # Import copy for deep copying (ensure it's available throughout the class)
        import copy
        
        self.user_id = user_id
        self.embedder = embedder
        self.feature_processor = feature_processor
        self.db_path = db_path
        
        # Use configuration values if not provided
        self.expert_weight = expert_weight or USER_PROFILE['expert_weight']
        self.historical_weight = historical_weight or USER_PROFILE['historical_weight']
        self.categorical_weight = categorical_weight or USER_PROFILE['categorical_weight']
        
        # Normalize weights if they don't sum to 1.0
        total_weight = self.expert_weight + self.historical_weight + self.categorical_weight
        if total_weight != 1.0:
            logger.warning(f"Profile component weights don't sum to 1.0 ({total_weight}). Normalizing.")
            self.expert_weight /= total_weight
            self.historical_weight /= total_weight
            self.categorical_weight /= total_weight
            
        logger.info(f"Initialized UserProfile for {user_id} with weights: "
                   f"expert={self.expert_weight:.2f}, historical={self.historical_weight:.2f}, "
                   f"categorical={self.categorical_weight:.2f}")
        
        # Initialize profile components
        self.expert_profile_embedding = None
        self.expert_profile = None  # For test compatibility
        self.expert_profile_text = None  # For storing the original text
        self.historical_embeddings = []
        self.historical_documents = []
        # Always initialize categorical_preferences as an empty dict
        # This will be deep copied whenever accessed to prevent reference issues
        self._categorical_preferences = {}
    
    @property
    def categorical_preferences(self):
        """Get a safe deep copy of categorical preferences."""
        # Always return a deep copy to ensure no shared references
        import copy
        return copy.deepcopy(self._categorical_preferences)
    
    @categorical_preferences.setter
    def categorical_preferences(self, value):
        """Set categorical preferences with a deep copy to avoid reference issues."""
        import copy
        if value is None:
            self._categorical_preferences = {}
        else:
            # Always store a deep copy to ensure no shared references
            self._categorical_preferences = copy.deepcopy(value)
        
        # Note: We don't want to load from database when setting preferences
        # That would create an endless recursion loop with _load_profile_from_db
        # The database operations should be handled separately
    
    def create_expert_profile(self, profile_text: str) -> np.ndarray:
        """
        Create an expert-curated profile embedding from descriptive text.
        
        Args:
            profile_text: Detailed description of client's regulatory interests
                         using language consistent with EU legislative texts
        
        Returns:
            Embedding vector for the expert profile
        """
        logger.info(f"Creating expert profile for user {self.user_id}")
        
        # Generate embedding for the expert profile text
        self.expert_profile_embedding = self.embedder.generate_embeddings(
            [profile_text], show_progress=False
        )[0]
        
        # Store the profile text and set the expert_profile for compatibility
        self.expert_profile_text = profile_text
        self.expert_profile = self.expert_profile_embedding
        
        # Save to database if available
        if self.db_path:
            self._save_expert_profile_to_db(profile_text)
        
        return self.expert_profile_embedding
    
    def add_historical_document(self, document_id: str, document_embedding: Optional[np.ndarray] = None) -> None:
        """
        Add a historical document of interest to the user profile.
        
        Args:
            document_id: ID of the document (e.g., CELEX number)
            document_embedding: Optional pre-computed embedding for the document
        """
        logger.info(f"Adding historical document {document_id} to profile for user {self.user_id}")
        
        # Add document ID to historical documents list
        if document_id not in self.historical_documents:
            self.historical_documents.append(document_id)
        
        if document_embedding is not None:
            # Use provided embedding
            self.historical_embeddings.append(document_embedding)
        else:
            # Fetch document embedding from database or compute it
            # This would typically connect to your document database
            # For now, we'll just log a warning
            logger.warning(f"Document embedding for {document_id} not provided and cannot be retrieved")
        
        # Save to database if available
        if self.db_path:
            self._save_historical_document_to_db(document_id)
    
    def set_categorical_preferences(self, preferences: Dict[str, Union[str, List[str]]]) -> None:
        """
        Set categorical preferences for the user profile.
        
        Args:
            preferences: Dictionary mapping feature names to preferred values
                        e.g., {'document_type': 'regulation', 'subject_matters': ['environment', 'energy']}
        """
        logger.info(f"Setting categorical preferences for user {self.user_id}")
        
        # Use our setter which automatically handles deep copying
        self.categorical_preferences = preferences
        
        # Save to database if available
        if self.db_path:
            self._save_categorical_preferences_to_db()
    
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
        
    def get_composite_profile(self, expert_weight: float = None, historical_weight: float = None, 
                        categorical_weight: float = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate a composite user profile combining all components.
        
        Args:
            expert_weight: Optional weight for expert profile component. Defaults to self.expert_weight.
            historical_weight: Optional weight for historical documents. Defaults to self.historical_weight.
            categorical_weight: Optional weight for categorical preferences. Defaults to self.categorical_weight.
            
        Returns:
            Tuple containing:
            - Composite embedding vector
            - Dictionary of categorical preferences
        """
        # Use provided weights or default to instance weights
        expert_w = expert_weight if expert_weight is not None else self.expert_weight
        historical_w = historical_weight if historical_weight is not None else self.historical_weight
        categorical_w = categorical_weight if categorical_weight is not None else self.categorical_weight
        
        # Normalize weights if they don't sum to 1.0
        total_weight = expert_w + historical_w + categorical_w
        if total_weight != 1.0:
            logger.debug(f"Normalizing provided component weights: {expert_w:.2f}, {historical_w:.2f}, {categorical_w:.2f}")
            expert_w /= total_weight
            historical_w /= total_weight
            categorical_w /= total_weight
        
        # Check if we have an expert profile
        if self.expert_profile_embedding is None:
            raise ValueError("Expert profile has not been created yet")
        
        # Initialize with expert profile
        components = [self.expert_profile_embedding * expert_w]
        component_weights = [expert_w]
        
        # Add historical embeddings if available
        if self.historical_embeddings:
            # Average the historical embeddings
            history_embedding = np.mean(self.historical_embeddings, axis=0)
            components.append(history_embedding * historical_w)
            component_weights.append(historical_w)
        
        # Log the component weights being used
        logger.debug(f"Creating composite profile with weights: expert={expert_w:.2f}, "
                   f"historical={historical_w:.2f}, categorical={categorical_w:.2f}")
        
        # Combine components
        composite_embedding = np.sum(components, axis=0)
        
        # Normalize the composite embedding
        norm = np.linalg.norm(composite_embedding)
        if norm > 0:
            composite_embedding = composite_embedding / norm
        
        # Instead of using copy.deepcopy directly, use our safe merge utility
        # This ensures we're creating a completely new dictionary with all keys copied safely
        if self.categorical_preferences:
            safe_preferences = self._safe_merge_dicts(self.categorical_preferences)
            return composite_embedding, safe_preferences
        else:
            return composite_embedding, {}
    
    def _load_profile_from_db(self) -> None:
        """Load user profile from SQLite database."""
        if not self.db_path:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load expert profile
            cursor.execute(
                "SELECT profile_text, embedding FROM user_expert_profiles WHERE user_id = ?",
                (self.user_id,)
            )
            result = cursor.fetchone()
            if result:
                profile_text, embedding_blob = result
                self.expert_profile_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                logger.info(f"Loaded expert profile for user {self.user_id}")
            
            # Load historical documents
            cursor.execute(
                "SELECT document_id, embedding FROM user_historical_documents WHERE user_id = ?",
                (self.user_id,)
            )
            for document_id, embedding_blob in cursor.fetchall():
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                self.historical_embeddings.append(embedding)
                logger.info(f"Loaded historical document {document_id} for user {self.user_id}")
            
            # Load categorical preferences
            cursor.execute(
                "SELECT preferences FROM user_categorical_preferences WHERE user_id = ?",
                (self.user_id,)
            )
            result = cursor.fetchone()
            if result:
                preferences_json = result[0]
                # Parse the JSON and set using our property setter which handles deep copying
                loaded_preferences = json.loads(preferences_json)
                self.categorical_preferences = loaded_preferences
                logger.info(f"Loaded categorical preferences for user {self.user_id}")
            
            conn.close()
        except Exception as e:
            logger.error(f"Error loading user profile from database: {e}")
    
    def _save_expert_profile_to_db(self, profile_text: str) -> None:
        """Save expert profile to SQLite database."""
        if not self.db_path or self.expert_profile_embedding is None:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_expert_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_text TEXT,
                    embedding BLOB,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)
            
            # Convert embedding to binary blob
            embedding_blob = self.expert_profile_embedding.tobytes()
            
            # Insert or update profile
            cursor.execute("""
                INSERT OR REPLACE INTO user_expert_profiles 
                (user_id, profile_text, embedding, created_at, updated_at)
                VALUES (?, ?, ?, COALESCE(
                    (SELECT created_at FROM user_expert_profiles WHERE user_id = ?), 
                    CURRENT_TIMESTAMP
                ), CURRENT_TIMESTAMP)
            """, (self.user_id, profile_text, embedding_blob, self.user_id))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved expert profile for user {self.user_id} to database")
        except Exception as e:
            logger.error(f"Error saving expert profile to database: {e}")
    
    def _save_historical_document_to_db(self, document_id: str) -> None:
        """Save historical document to SQLite database."""
        if not self.db_path or not self.historical_embeddings:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_historical_documents (
                    user_id TEXT,
                    document_id TEXT,
                    embedding BLOB,
                    added_at TIMESTAMP,
                    PRIMARY KEY (user_id, document_id)
                )
            """)
            
            # Get the latest embedding (the one we just added)
            embedding_blob = self.historical_embeddings[-1].tobytes()
            
            # Insert document
            cursor.execute("""
                INSERT OR REPLACE INTO user_historical_documents 
                (user_id, document_id, embedding, added_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (self.user_id, document_id, embedding_blob))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved historical document {document_id} for user {self.user_id} to database")
        except Exception as e:
            logger.error(f"Error saving historical document to database: {e}")
    
    def _save_categorical_preferences_to_db(self) -> None:
        """Save categorical preferences to SQLite database."""
        # Get a safe copy of preferences via the property
        preferences = self.categorical_preferences
        
        if not self.db_path or not preferences:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_categorical_preferences (
                    user_id TEXT PRIMARY KEY,
                    preferences TEXT,
                    updated_at TIMESTAMP
                )
            """)
            
            # Convert preferences to JSON
            preferences_json = json.dumps(preferences)
            
            # Insert or update preferences
            cursor.execute("""
                INSERT OR REPLACE INTO user_categorical_preferences 
                (user_id, preferences, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (self.user_id, preferences_json))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved categorical preferences for user {self.user_id} to database")
        except Exception as e:
            logger.error(f"Error saving categorical preferences to database: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert user profile to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the user profile
        """
        # Create a fresh dictionary to avoid reference issues
        profile_dict = {
            'user_id': self.user_id,
            'weights': {
                'expert_weight': float(self.expert_weight),
                'historical_weight': float(self.historical_weight),
                'categorical_weight': float(self.categorical_weight)
            },
            'expert_profile_text': self.expert_profile_text,
            'historical_documents': list(self.historical_documents) if self.historical_documents else [],
        }
        
        # Handle categorical preferences carefully to avoid dictionary addition issues
        # Create a deep copy of the categorical preferences
        if self.categorical_preferences and isinstance(self.categorical_preferences, dict):
            cat_prefs = {}
            for key, value in self.categorical_preferences.items():
                if isinstance(value, dict):
                    # Handle nested dictionaries
                    cat_prefs[key] = {}
                    for subkey, subvalue in value.items():
                        cat_prefs[key][subkey] = subvalue
                else:
                    cat_prefs[key] = value
            profile_dict['categorical_preferences'] = cat_prefs
        else:
            profile_dict['categorical_preferences'] = {}
            
        return profile_dict
    
    @classmethod
    def from_dict(cls, profile_dict: Dict[str, Any], embedder: BERTEmbedder, 
                 feature_processor: Optional[FeatureProcessor] = None,
                 db_path: Optional[str] = None) -> 'UserProfile':
        """
        Create a UserProfile instance from a dictionary.
        
        Args:
            profile_dict: Dictionary representation of a user profile
            embedder: BERTEmbedder instance for generating text embeddings
            feature_processor: Optional FeatureProcessor for handling categorical features
            db_path: Optional path to SQLite database for storing profiles
            
        Returns:
            UserProfile instance
        """
        # Extract weights
        weights = profile_dict.get('weights', {})
        expert_weight = weights.get('expert_weight', None)
        historical_weight = weights.get('historical_weight', None)
        categorical_weight = weights.get('categorical_weight', None)
        
        # Create profile instance
        profile = cls(
            user_id=profile_dict['user_id'],
            embedder=embedder,
            feature_processor=feature_processor,
            db_path=db_path,
            expert_weight=expert_weight,
            historical_weight=historical_weight,
            categorical_weight=categorical_weight
        )
        
        # Set expert profile if available
        expert_profile_text = profile_dict.get('expert_profile_text')
        if expert_profile_text:
            profile.create_expert_profile(expert_profile_text)
        
        # Set historical documents if available
        for doc_id in profile_dict.get('historical_documents', []):
            # Note: This just adds the document ID, not the embedding
            # The actual embeddings would need to be loaded separately
            profile.historical_documents.append(doc_id)
        
        # Set categorical preferences if available, ensuring deep copying
        categorical_preferences = profile_dict.get('categorical_preferences', {})
        if categorical_preferences:
            # Use our safe dictionary merging utility instead of manual deep copying
            # This creates a completely new dictionary with all keys safely copied
            safe_preferences = profile._safe_merge_dicts(categorical_preferences)
            
            # Now set the safely copied preferences
            profile.set_categorical_preferences(safe_preferences)
        
        return profile
