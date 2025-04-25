"""
Weight configuration module for the EU Legal Recommender system.

This module provides a centralized configuration system for managing weights
across different components of the recommender system.
"""

from typing import Dict, Any, Optional, List
import json
from pathlib import Path

from src.utils.logging import get_logger

# Set up logger
logger = get_logger(__name__)

class WeightConfig:
    """
    Centralized configuration for weights in the recommender system.
    
    This class manages weights for different components of the recommender system,
    including similarity computation, embedding generation, and feature processing.
    It provides methods for loading, saving, and applying weights.
    """
    
    DEFAULT_WEIGHTS = {
        # Similarity weights
        "similarity": {
            "text_weight": 0.7,
            "categorical_weight": 0.3
        },
        
        # Embedding weights
        "embedding": {
            "summary_weight": 0.6,
            "keywords_weight": 0.4
        },
        
        # Feature weights
        "features": {
            "document_type_weight": 0.3,
            "subject_matter_weight": 0.4,
            "date_weight": 0.1,
            "author_weight": 0.2
        },
        
        # Personalization weights
        "personalization": {
            "profile_weight": 0.5,
            "query_weight": 0.5
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize weight configuration.
        
        Args:
            config_path: Optional path to load configuration from
        """
        # Initialize with default weights
        self.weights = self.DEFAULT_WEIGHTS.copy()
        
        # Load from file if provided
        if config_path:
            self.load_config(config_path)
            
        logger.info(f"Initialized WeightConfig with {len(self.weights)} weight categories")
    
    def get_weights(self, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Get weights for a specific category or all weights.
        
        Args:
            category: Optional category name (e.g., 'similarity', 'embedding')
            
        Returns:
            Dictionary of weights
        """
        if category:
            return self.weights.get(category, {})
        return self.weights
    
    def set_weights(self, weights: Dict[str, Any], category: Optional[str] = None) -> None:
        """
        Set weights for a specific category or update all weights.
        
        Args:
            weights: Dictionary of weights to set
            category: Optional category name (e.g., 'similarity', 'embedding')
        """
        if category:
            if category not in self.weights:
                self.weights[category] = {}
            self.weights[category].update(weights)
            logger.info(f"Updated weights for category '{category}'")
        else:
            # Update entire weight structure
            for cat, cat_weights in weights.items():
                if cat not in self.weights:
                    self.weights[cat] = {}
                self.weights[cat].update(cat_weights)
            logger.info(f"Updated weights for {len(weights)} categories")
    
    def load_config(self, file_path: str) -> None:
        """
        Load weights from configuration file.
        
        Args:
            file_path: Path to configuration file (JSON)
        """
        try:
            with open(file_path, 'r') as f:
                loaded_weights = json.load(f)
            
            # Update weights
            self.set_weights(loaded_weights)
            logger.info(f"Loaded weights from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading weights from {file_path}: {str(e)}")
            logger.info("Using default weights instead")
    
    def save_config(self, file_path: str) -> None:
        """
        Save weights to configuration file.
        
        Args:
            file_path: Path to save configuration (JSON)
        """
        try:
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save weights
            with open(file_path, 'w') as f:
                json.dump(self.weights, f, indent=2)
                
            logger.info(f"Saved weights to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving weights to {file_path}: {str(e)}")
    
    def apply_to_recommender(self, recommender) -> None:
        """
        Apply weights to a recommender instance.
        
        This method applies weights to different components of the recommender
        based on their availability.
        
        Args:
            recommender: Recommender instance to apply weights to
        """
        try:
            # Apply similarity weights
            if "similarity" in self.weights and hasattr(recommender, "similarity_computer"):
                sim_weights = self.weights["similarity"]
                if hasattr(recommender.similarity_computer, "text_weight") and "text_weight" in sim_weights:
                    recommender.similarity_computer.text_weight = sim_weights["text_weight"]
                if hasattr(recommender.similarity_computer, "categorical_weight") and "categorical_weight" in sim_weights:
                    recommender.similarity_computer.categorical_weight = sim_weights["categorical_weight"]
                logger.debug(f"Applied similarity weights: {sim_weights}")
            
            # Apply embedding weights
            if "embedding" in self.weights and hasattr(recommender, "embedder"):
                emb_weights = self.weights["embedding"]
                if hasattr(recommender.embedder, "set_weights"):
                    recommender.embedder.set_weights(**emb_weights)
                elif hasattr(recommender.embedder, "summary_weight") and "summary_weight" in emb_weights:
                    recommender.embedder.summary_weight = emb_weights["summary_weight"]
                    if hasattr(recommender.embedder, "keywords_weight") and "keywords_weight" in emb_weights:
                        recommender.embedder.keywords_weight = emb_weights["keywords_weight"]
                logger.debug(f"Applied embedding weights: {emb_weights}")
            
            # Apply feature weights
            if "features" in self.weights and hasattr(recommender, "feature_processor"):
                feat_weights = self.weights["features"]
                if hasattr(recommender.feature_processor, "set_weights"):
                    recommender.feature_processor.set_weights(**feat_weights)
                logger.debug(f"Applied feature weights: {feat_weights}")
            
            # Apply personalization weights
            if "personalization" in self.weights and hasattr(recommender, "set_personalization_weights"):
                pers_weights = self.weights["personalization"]
                recommender.set_personalization_weights(**pers_weights)
                logger.debug(f"Applied personalization weights: {pers_weights}")
            
            logger.info(f"Applied weights to {type(recommender).__name__} instance")
            
        except Exception as e:
            logger.error(f"Error applying weights to recommender: {str(e)}")
            raise ValueError(f"Failed to apply weights: {str(e)}")
