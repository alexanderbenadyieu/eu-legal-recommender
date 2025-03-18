"""
Process and encode categorical features for legal documents.
"""
from typing import Dict, List, Union, Any
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureProcessor:
    """Process and encode categorical features for legal documents."""
    
    def __init__(self, feature_config: Dict[str, List[str]] = None, multi_valued_features: List[str] = None, feature_weights: Dict[str, float] = None):
        """
        Initialize the feature processor.
        
        Args:
            feature_config: Dictionary mapping feature names to possible values
                          If None, will learn from data
            multi_valued_features: List of feature names that can have multiple values
                                 These will be encoded using multi-hot encoding
            feature_weights: Dictionary mapping feature names to importance weights
                           If None, default weights will be used
        """
        self.feature_config = feature_config
        self.encoders = {}
        self.fitted = False
        
        # Default multi-valued features if not provided
        self.multi_valued_features = multi_valued_features or [
            "subject_matters", 
            "eurovoc_descriptors", 
            "authors", 
            "directory_codes"
        ]
        
        # Default feature weights (importance of each feature type)
        self.feature_weights = feature_weights or {
            "form": 0.15,                # Document type (regulation, directive, etc.)
            "subject_matters": 0.25,     # Subject matters are highly relevant
            "eurovoc_descriptors": 0.25, # EuroVoc descriptors are also highly relevant
            "authors": 0.10,            # Authors have moderate relevance
            "directory_codes": 0.15,     # Directory codes indicate document classification
            "responsible_body": 0.10     # Body responsible for the document
        }
        
    def fit(self, features_data: List[Dict[str, Union[str, List[str]]]]) -> None:
        """
        Fit encoders on feature data.
        
        Args:
            features_data: List of dictionaries containing feature values
                         For multi-valued features, values should be lists of strings
        """
        # Extract unique feature names if no config provided
        if not self.feature_config:
            self.feature_config = {}
            for features in features_data:
                for name, value in features.items():
                    # Skip non-categorical features
                    if name in ['id', 'celex_number', 'title', 'summary', 'total_words', 
                               'summary_word_count', 'tier', 'date_of_document', 
                               'date_of_effect', 'date_of_end_validity']:
                        continue
                        
                    # Skip keywords as they're handled separately by the embedder
                    if name == 'keywords':
                        continue
                        
                    # Initialize feature list if needed
                    if name not in self.feature_config:
                        self.feature_config[name] = []
                    
                    # Handle directory_codes specially (they're dicts with 'code' and 'label')
                    if name == 'directory_codes' and isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict) and 'code' in item:
                                code_value = item['code']
                                if code_value and code_value not in self.feature_config[name]:
                                    self.feature_config[name].append(code_value)
                        continue
                    
                    # Handle multi-valued features
                    if name in self.multi_valued_features and isinstance(value, list):
                        for val in value:
                            if val and val not in self.feature_config[name]:
                                self.feature_config[name].append(val)
                    else:
                        # Handle single-valued features
                        if isinstance(value, list) and len(value) > 0:
                            # If a list was provided for a single-valued feature, use the first value
                            value = value[0]
                        if value and value not in self.feature_config[name]:
                            self.feature_config[name].append(value)
        
        # Create and fit encoders
        for feature_name, possible_values in self.feature_config.items():
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoder.fit([[v] for v in possible_values])
            self.encoders[feature_name] = encoder
            
            logger.info(f"Fitted encoder for '{feature_name}' with {len(possible_values)} possible values")
            
        self.fitted = True
        
    def encode_features(self, features: Dict[str, Union[str, List[str]]]) -> np.ndarray:
        """
        Encode a single document's features.
        
        Args:
            features: Dictionary of feature name-value pairs
                    For multi-valued features, the value should be a list of strings
            
        Returns:
            Concatenated encoded features (one-hot for single-valued, multi-hot for multi-valued)
        """
        if not self.fitted:
            raise ValueError("FeatureProcessor must be fitted before encoding")
            
        encoded_features = []
        feature_dimensions = {}
        
        for feature_name, encoder in self.encoders.items():
            value = features.get(feature_name, None)
            feature_dim = len(encoder.categories_[0])
            feature_dimensions[feature_name] = feature_dim
            
            # Handle directory_codes specially
            if feature_name == 'directory_codes' and isinstance(value, list):
                # Extract codes from the list of dictionaries
                codes = [item['code'] for item in value if isinstance(item, dict) and 'code' in item]
                value = codes
            
            if value is None or (isinstance(value, list) and len(value) == 0):
                # Use zero vector for missing features
                encoded = np.zeros(feature_dim)
            elif feature_name in self.multi_valued_features and isinstance(value, list):
                # Multi-hot encoding for multi-valued features
                encoded = np.zeros(feature_dim)
                for val in value:
                    try:
                        # Find the index of this value in the categories
                        val_idx = np.where(encoder.categories_[0] == val)[0]
                        if len(val_idx) > 0:
                            encoded[val_idx[0]] = 1
                    except Exception as e:
                        logger.warning(f"Error encoding value '{val}' for feature '{feature_name}': {e}")
            else:
                # One-hot encoding for single-valued features
                if isinstance(value, list) and len(value) > 0:
                    # If a list was provided for a single-valued feature, use the first value
                    value = value[0]
                try:
                    encoded = encoder.transform([[value]])[0]
                except Exception as e:
                    logger.warning(f"Error encoding value '{value}' for feature '{feature_name}': {e}")
                    encoded = np.zeros(feature_dim)
            
            # Apply feature weight if available
            if feature_name in self.feature_weights:
                weight = self.feature_weights[feature_name]
                encoded = encoded * weight
                
            encoded_features.append(encoded)
            
        return np.concatenate(encoded_features)
    
    def get_feature_dims(self) -> Dict[str, int]:
        """
        Get the dimensionality of each encoded feature.
        
        Returns:
            Dictionary mapping feature names to their encoded dimensions
        """
        return {
            name: len(encoder.categories_[0])
            for name, encoder in self.encoders.items()
        }
        
    def get_feature_weights(self) -> Dict[str, float]:
        """
        Get the weight of each feature type.
        
        Returns:
            Dictionary mapping feature names to their weights
        """
        return self.feature_weights.copy()
        
    def get_total_dims(self) -> int:
        """
        Get total dimensionality of encoded features.
        
        Returns:
            Sum of all feature dimensions
        """
        return sum(self.get_feature_dims().values())
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the feature processor state.
        
        Args:
            path: Directory to save state to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save feature config
        with open(path / 'feature_config.json', 'w') as f:
            json.dump(self.feature_config, f)
            
        # Save feature weights
        with open(path / 'feature_weights.json', 'w') as f:
            json.dump(self.feature_weights, f)
            
        # Save multi-valued features list
        with open(path / 'multi_valued_features.json', 'w') as f:
            json.dump(self.multi_valued_features, f)
            
        # Save encoder states
        for name, encoder in self.encoders.items():
            np.save(
                path / f'encoder_{name}_categories.npy',
                encoder.categories_[0]
            )
            
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'FeatureProcessor':
        """
        Load a saved feature processor.
        
        Args:
            path: Directory containing saved state
            
        Returns:
            Loaded FeatureProcessor instance
        """
        path = Path(path)
        
        # Load feature config
        with open(path / 'feature_config.json', 'r') as f:
            feature_config = json.load(f)
        
        # Load feature weights if available
        feature_weights = None
        if (path / 'feature_weights.json').exists():
            with open(path / 'feature_weights.json', 'r') as f:
                feature_weights = json.load(f)
        
        # Load multi-valued features if available
        multi_valued_features = None
        if (path / 'multi_valued_features.json').exists():
            with open(path / 'multi_valued_features.json', 'r') as f:
                multi_valued_features = json.load(f)
            
        processor = cls(feature_config, multi_valued_features, feature_weights)
        
        # Load and recreate encoders
        for feature_name in feature_config:
            categories = np.load(path / f'encoder_{feature_name}_categories.npy')
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoder.fit([[c] for c in categories])
            processor.encoders[feature_name] = encoder
            
        processor.fitted = True
        return processor
        
    def calculate_feature_importance(self, features_data: List[Dict[str, Union[str, List[str]]]]) -> Dict[str, float]:
        """
        Calculate feature importance based on frequency and diversity of values.
        
        Args:
            features_data: List of dictionaries containing feature values
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.fitted:
            raise ValueError("FeatureProcessor must be fitted before calculating importance")
            
        importance_scores = {}
        total_docs = len(features_data)
        
        for feature_name, encoder in self.encoders.items():
            # Count occurrences of each value
            value_counts = {val: 0 for val in encoder.categories_[0]}
            docs_with_feature = 0
            
            for doc_features in features_data:
                value = doc_features.get(feature_name)
                
                if value is not None:
                    docs_with_feature += 1
                    
                    # Handle multi-valued features
                    if feature_name in self.multi_valued_features and isinstance(value, list):
                        for val in value:
                            if val in value_counts:
                                value_counts[val] += 1
                    else:
                        # Handle single-valued features
                        if isinstance(value, list) and len(value) > 0:
                            value = value[0]
                        if value in value_counts:
                            value_counts[value] += 1
            
            # Calculate entropy (diversity of values)
            total_occurrences = sum(value_counts.values())
            if total_occurrences > 0:
                probabilities = [count / total_occurrences for count in value_counts.values() if count > 0]
                entropy = -sum(p * np.log(p) for p in probabilities)
            else:
                entropy = 0
                
            # Calculate coverage (percentage of documents with this feature)
            coverage = docs_with_feature / total_docs if total_docs > 0 else 0
            
            # Combine entropy and coverage for importance score
            importance = 0.7 * entropy + 0.3 * coverage
            importance_scores[feature_name] = importance
        
        # Normalize importance scores
        max_importance = max(importance_scores.values()) if importance_scores else 1.0
        normalized_scores = {name: score / max_importance for name, score in importance_scores.items()}
        
        return normalized_scores
