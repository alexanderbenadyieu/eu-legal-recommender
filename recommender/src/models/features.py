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
    
    def __init__(self, feature_config: Dict[str, List[str]] = None):
        """
        Initialize the feature processor.
        
        Args:
            feature_config: Dictionary mapping feature names to possible values
                          If None, will learn from data
        """
        self.feature_config = feature_config
        self.encoders = {}
        self.fitted = False
        
    def fit(self, features_data: List[Dict[str, str]]) -> None:
        """
        Fit encoders on feature data.
        
        Args:
            features_data: List of dictionaries containing feature values
        """
        # Extract unique feature names if no config provided
        if not self.feature_config:
            self.feature_config = {}
            for features in features_data:
                for name, value in features.items():
                    if name not in self.feature_config:
                        self.feature_config[name] = []
                    if value not in self.feature_config[name]:
                        self.feature_config[name].append(value)
        
        # Create and fit encoders
        for feature_name, possible_values in self.feature_config.items():
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoder.fit([[v] for v in possible_values])
            self.encoders[feature_name] = encoder
            
        self.fitted = True
        
    def encode_features(self, features: Dict[str, str]) -> np.ndarray:
        """
        Encode a single document's features.
        
        Args:
            features: Dictionary of feature name-value pairs
            
        Returns:
            Concatenated one-hot encoded features
        """
        if not self.fitted:
            raise ValueError("FeatureProcessor must be fitted before encoding")
            
        encoded_features = []
        for feature_name, encoder in self.encoders.items():
            value = features.get(feature_name, None)
            if value is None:
                # Use zero vector for missing features
                encoded = np.zeros(len(encoder.categories_[0]))
            else:
                encoded = encoder.transform([[value]])[0]
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
            
        processor = cls(feature_config)
        
        # Load and recreate encoders
        for feature_name in feature_config:
            categories = np.load(path / f'encoder_{feature_name}_categories.npy')
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoder.fit([[c] for c in categories])
            processor.encoders[feature_name] = encoder
            
        processor.fitted = True
        return processor
