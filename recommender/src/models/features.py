"""
Process and encode categorical features for legal documents.

This module provides the FeatureProcessor class for handling categorical features
of legal documents. It supports one-hot encoding for single-valued features and
multi-hot encoding for multi-valued features, feature importance calculation,
and persistence of feature encoders.
"""
from typing import Dict, List, Union, Any, Optional, ClassVar, Type
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import json
from pathlib import Path

# Import utilities
from ..utils.logging import get_logger
from ..utils.exceptions import ValidationError, DataError, ProcessingError

# Set up logger for this module
logger = get_logger(__name__)

class FeatureProcessor:
    """Process and encode categorical features for legal documents.
    
    This class handles the processing and encoding of categorical features for legal documents,
    supporting both single-valued features (one-hot encoding) and multi-valued features 
    (multi-hot encoding). It provides functionality for fitting encoders on feature data,
    encoding features, calculating feature importance, and persisting the processor state.
    
    Attributes:
        feature_config (Dict[str, List[str]]): Dictionary mapping feature names to possible values
        encoders (Dict[str, OneHotEncoder]): Dictionary of fitted encoders for each feature
        fitted (bool): Whether the processor has been fitted on data
        multi_valued_features (List[str]): List of feature names that can have multiple values
        feature_weights (Dict[str, float]): Dictionary mapping feature names to importance weights
    """
    
    def __init__(self, feature_config: Optional[Dict[str, List[str]]] = None, 
                 multi_valued_features: Optional[List[str]] = None, 
                 feature_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the feature processor.
        
        Args:
            feature_config (Dict[str, List[str]], optional): Dictionary mapping feature names 
                to possible values. If None, will learn from data.
            multi_valued_features (List[str], optional): List of feature names that can have 
                multiple values. These will be encoded using multi-hot encoding.
            feature_weights (Dict[str, float], optional): Dictionary mapping feature names 
                to importance weights. If None, default weights will be used.
                
        Raises:
            ValidationError: If feature_weights contains invalid values or doesn't sum to 1.0
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
        
        logger.debug(f"Initialized with {len(self.multi_valued_features)} multi-valued features")
        
        # Default feature weights (importance of each feature type)
        default_weights = {
            "form": 0.15,                # Document type (regulation, directive, etc.)
            "subject_matters": 0.25,     # Subject matters are highly relevant
            "eurovoc_descriptors": 0.25, # EuroVoc descriptors are also highly relevant
            "authors": 0.10,            # Authors have moderate relevance
            "directory_codes": 0.15,     # Directory codes indicate document classification
            "responsible_body": 0.10     # Body responsible for the document
        }
        
        # Validate feature weights if provided
        if feature_weights is not None:
            # Check for negative weights
            for feature, weight in feature_weights.items():
                if weight < 0:
                    logger.warning(f"Negative weight {weight} for feature {feature}, setting to 0")
                    feature_weights[feature] = 0
            
            # Check if weights sum to approximately 1.0
            weight_sum = sum(feature_weights.values())
            if abs(weight_sum - 1.0) > 0.01:  # Allow small floating point error
                logger.warning(f"Feature weights sum to {weight_sum}, normalizing to 1.0")
                feature_weights = {k: v / weight_sum for k, v in feature_weights.items()}
        
        self.feature_weights = feature_weights or default_weights
        logger.info(f"Feature processor initialized with {len(self.feature_weights)} weighted features")
        
    def fit(self, features_data: List[Dict[str, Union[str, List[str]]]]) -> None:
        """
        Fit encoders on feature data.
        
        This method analyzes the provided feature data to extract unique feature values
        and creates appropriate encoders for each feature. If feature_config was provided
        during initialization, it will use that configuration instead of learning from data.
        
        Args:
            features_data (List[Dict[str, Union[str, List[str]]]]): List of dictionaries 
                containing feature values. For multi-valued features, values should be 
                lists of strings.
                
        Raises:
            ValidationError: If features_data is None or empty
            DataError: If there's an error processing the feature data
        """
        # Validate input
        if features_data is None:
            logger.error("features_data is None")
            raise ValidationError("features_data cannot be None")
            
        if not features_data:
            logger.error("Empty features_data provided")
            raise ValidationError("features_data cannot be empty")
            
        logger.info(f"Fitting feature processor on {len(features_data)} documents")
        
        try:
            # Extract unique feature names if no config provided
            if not self.feature_config:
                self.feature_config = {}
                non_categorical_features = ['id', 'celex_number', 'title', 'summary', 'total_words', 
                                        'summary_word_count', 'tier', 'date_of_document', 
                                        'date_of_effect', 'date_of_end_validity']
                
                logger.debug("Learning feature configuration from data")
                
                for features in features_data:
                    for name, value in features.items():
                        # Skip non-categorical features
                        if name in non_categorical_features:
                            continue
                            
                        # Skip keywords as they're handled separately by the embedder
                        if name == 'keywords':
                            continue
                            
                        # Initialize feature list if needed
                        if name not in self.feature_config:
                            self.feature_config[name] = []
                            logger.debug(f"Discovered new feature: {name}")
                        
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
                
                logger.info(f"Learned configuration for {len(self.feature_config)} features")
            else:
                logger.info(f"Using provided feature configuration with {len(self.feature_config)} features")
        
            # Create and fit encoders
            for feature_name, possible_values in self.feature_config.items():
                if not possible_values:
                    logger.warning(f"No values found for feature '{feature_name}', skipping encoder creation")
                    continue
                    
                try:
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoder.fit([[v] for v in possible_values])
                    self.encoders[feature_name] = encoder
                    
                    logger.info(f"Fitted encoder for '{feature_name}' with {len(possible_values)} possible values")
                except Exception as e:
                    logger.error(f"Error creating encoder for feature '{feature_name}': {str(e)}")
                    raise DataError(f"Failed to create encoder for feature '{feature_name}': {str(e)}")
            
            self.fitted = True
            logger.info(f"Successfully fitted {len(self.encoders)} feature encoders")
            
        except Exception as e:
            logger.error(f"Error fitting feature processor: {str(e)}")
            raise DataError(f"Failed to fit feature processor: {str(e)}")
        
    def encode_features(self, features: Dict[str, Union[str, List[str]]]) -> np.ndarray:
        """
        Encode a single document's features.
        
        This method encodes the categorical features of a document using the fitted encoders.
        For single-valued features, it uses one-hot encoding. For multi-valued features,
        it uses multi-hot encoding (multiple 1s in the encoding vector). Feature weights are
        applied to the encoded vectors before concatenation.
        
        Args:
            features (Dict[str, Union[str, List[str]]]): Dictionary of feature name-value pairs.
                For multi-valued features, the value should be a list of strings.
            
        Returns:
            np.ndarray: Concatenated encoded features with weights applied.
            
        Raises:
            ValidationError: If features is None or if the processor has not been fitted
            ProcessingError: If there's an error during the encoding process
        """
        # Validate input
        if features is None:
            logger.error("features is None")
            raise ValidationError("features cannot be None")
            
        if not self.fitted:
            logger.error("Attempted to encode features with unfitted processor")
            raise ValidationError("FeatureProcessor must be fitted before encoding")
            
        logger.debug(f"Encoding features with {len(self.encoders)} encoders")
        
        try:
            encoded_features = []
            feature_dimensions = {}
            
            for feature_name, encoder in self.encoders.items():
                value = features.get(feature_name, None)
                feature_dim = len(encoder.categories_[0])
                feature_dimensions[feature_name] = feature_dim
                
                # Handle directory_codes specially
                if feature_name == 'directory_codes' and isinstance(value, list):
                    # Extract codes from the list of dictionaries
                    try:
                        codes = [item['code'] for item in value if isinstance(item, dict) and 'code' in item]
                        value = codes
                        logger.debug(f"Processed directory_codes: {len(codes)} codes extracted")
                    except Exception as e:
                        logger.warning(f"Error processing directory_codes: {str(e)}")
                        value = []
                
                if value is None or (isinstance(value, list) and len(value) == 0):
                    # Use zero vector for missing features
                    encoded = np.zeros(feature_dim)
                    logger.debug(f"No value for feature '{feature_name}', using zeros")
                elif feature_name in self.multi_valued_features and isinstance(value, list):
                    # Multi-hot encoding for multi-valued features
                    encoded = np.zeros(feature_dim)
                    matched_values = 0
                    for val in value:
                        try:
                            # Find the index of this value in the categories
                            val_idx = np.where(encoder.categories_[0] == val)[0]
                            if len(val_idx) > 0:
                                encoded[val_idx[0]] = 1
                                matched_values += 1
                            else:
                                logger.debug(f"Value '{val}' not found in categories for feature '{feature_name}'")
                        except Exception as e:
                            logger.warning(f"Error encoding value '{val}' for feature '{feature_name}': {str(e)}")
                    
                    logger.debug(f"Encoded multi-valued feature '{feature_name}' with {matched_values}/{len(value)} values")
                else:
                    # One-hot encoding for single-valued features
                    if isinstance(value, list) and len(value) > 0:
                        # If a list was provided for a single-valued feature, use the first value
                        logger.debug(f"List provided for single-valued feature '{feature_name}', using first value")
                        value = value[0]
                    try:
                        encoded = encoder.transform([[value]])[0]
                        logger.debug(f"Encoded single-valued feature '{feature_name}' with value '{value}'")
                    except Exception as e:
                        logger.warning(f"Error encoding value '{value}' for feature '{feature_name}': {str(e)}")
                        encoded = np.zeros(feature_dim)
                
                # Apply feature weight if available
                if feature_name in self.feature_weights:
                    weight = self.feature_weights[feature_name]
                    encoded = encoded * weight
                    logger.debug(f"Applied weight {weight} to feature '{feature_name}'")
                    
                encoded_features.append(encoded)
            
            result = np.concatenate(encoded_features)
            logger.debug(f"Successfully encoded features with total dimension {len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"Error encoding features: {str(e)}")
            raise ProcessingError(f"Failed to encode features: {str(e)}")
    
    def get_feature_dims(self) -> Dict[str, int]:
        """
        Get the dimensionality of each encoded feature.
        
        This method returns a dictionary that maps each feature name to the dimensionality
        of its encoded representation (number of possible values).
        
        Returns:
            Dict[str, int]: Dictionary mapping feature names to their encoded dimensions.
            
        Raises:
            ValidationError: If the processor has not been fitted
        """
        if not self.fitted:
            logger.warning("Attempted to get feature dimensions from unfitted processor")
            raise ValidationError("FeatureProcessor must be fitted before getting dimensions")
            
        dimensions = {
            name: len(encoder.categories_[0])
            for name, encoder in self.encoders.items()
        }
        
        logger.debug(f"Retrieved dimensions for {len(dimensions)} features")
        return dimensions
        
    def get_feature_weights(self) -> Dict[str, float]:
        """
        Get the weight of each feature type.
        
        This method returns a copy of the feature weights dictionary, which maps
        each feature name to its importance weight used during encoding.
        
        Returns:
            Dict[str, float]: Dictionary mapping feature names to their weights.
        """
        weights = self.feature_weights.copy()
        logger.debug(f"Retrieved weights for {len(weights)} features")
        return weights
        
    def get_total_dims(self) -> int:
        """
        Get total dimensionality of encoded features.
        
        This method calculates and returns the total dimensionality of all encoded features
        combined, which is the sum of the dimensions of each individual feature.
        
        Returns:
            int: Sum of all feature dimensions.
            
        Raises:
            ValidationError: If the processor has not been fitted
        """
        if not self.fitted:
            logger.warning("Attempted to get total dimensions from unfitted processor")
            raise ValidationError("FeatureProcessor must be fitted before getting total dimensions")
            
        total = sum(self.get_feature_dims().values())
        logger.debug(f"Total feature dimensions: {total}")
        return total
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the feature processor state.
        
        This method persists the state of the feature processor to disk, including the feature
        configuration, weights, multi-valued features list, and encoder states. The state can
        later be restored using the load() class method.
        
        Args:
            path (Union[str, Path]): Directory to save state to. Will be created if it doesn't exist.
            
        Raises:
            ValidationError: If the processor has not been fitted
            ProcessingError: If there's an error during the saving process
        """
        if not self.fitted:
            logger.error("Attempted to save unfitted processor")
            raise ValidationError("FeatureProcessor must be fitted before saving")
            
        try:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving feature processor state to {path}")
            
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
                
            logger.info(f"Successfully saved feature processor with {len(self.encoders)} encoders")
            
        except Exception as e:
            logger.error(f"Error saving feature processor: {str(e)}")
            raise ProcessingError(f"Failed to save feature processor: {str(e)}")
            
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'FeatureProcessor':
        """
        Load a saved feature processor.
        
        This class method loads a previously saved feature processor state from disk,
        recreating the feature configuration, weights, multi-valued features list,
        and encoder states.
        
        Args:
            path (Union[str, Path]): Directory containing saved state.
            
        Returns:
            FeatureProcessor: Loaded FeatureProcessor instance.
            
        Raises:
            ValidationError: If the path doesn't exist or required files are missing
            ProcessingError: If there's an error during the loading process
        """
        try:
            path = Path(path)
            
            if not path.exists():
                logger.error(f"Path does not exist: {path}")
                raise ValidationError(f"Path does not exist: {path}")
                
            logger.info(f"Loading feature processor from {path}")
            
            # Check for required files
            if not (path / 'feature_config.json').exists():
                logger.error(f"Missing required file: feature_config.json")
                raise ValidationError(f"Missing required file: feature_config.json")
            
            # Load feature config
            with open(path / 'feature_config.json', 'r') as f:
                feature_config = json.load(f)
            
            # Load feature weights if available
            feature_weights = None
            if (path / 'feature_weights.json').exists():
                with open(path / 'feature_weights.json', 'r') as f:
                    feature_weights = json.load(f)
                logger.debug("Loaded feature weights")
            else:
                logger.debug("No feature weights file found, using defaults")
            
            # Load multi-valued features if available
            multi_valued_features = None
            if (path / 'multi_valued_features.json').exists():
                with open(path / 'multi_valued_features.json', 'r') as f:
                    multi_valued_features = json.load(f)
                logger.debug(f"Loaded {len(multi_valued_features)} multi-valued features")
            else:
                logger.debug("No multi-valued features file found, using defaults")
                
            # Create processor instance
            processor = cls(feature_config, multi_valued_features, feature_weights)
            
            # Load and recreate encoders
            encoder_count = 0
            for feature_name in feature_config:
                encoder_file = path / f'encoder_{feature_name}_categories.npy'
                if not encoder_file.exists():
                    logger.warning(f"Missing encoder file for feature '{feature_name}', skipping")
                    continue
                    
                try:
                    categories = np.load(encoder_file)
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoder.fit([[c] for c in categories])
                    processor.encoders[feature_name] = encoder
                    encoder_count += 1
                except Exception as e:
                    logger.warning(f"Error loading encoder for feature '{feature_name}': {str(e)}")
                
            if encoder_count == 0:
                logger.error("No encoders could be loaded")
                raise ProcessingError("Failed to load any encoders")
                
            processor.fitted = True
            logger.info(f"Successfully loaded feature processor with {encoder_count} encoders")
            return processor
            
        except Exception as e:
            if isinstance(e, (ValidationError, ProcessingError)):
                raise
            logger.error(f"Error loading feature processor: {str(e)}")
            raise ProcessingError(f"Failed to load feature processor: {str(e)}")
        
    def calculate_feature_importance(self, features_data: List[Dict[str, Union[str, List[str]]]]) -> Dict[str, float]:
        """
        Calculate feature importance based on frequency and diversity of values.
        
        This method analyzes the provided feature data to calculate importance scores for each
        feature based on two factors:
        1. Entropy (diversity of values) - Features with more diverse values get higher scores
        2. Coverage (percentage of documents with this feature) - Features present in more
           documents get higher scores
        
        The final importance score is a weighted combination of entropy (70%) and coverage (30%),
        normalized to a 0-1 scale.
        
        Args:
            features_data (List[Dict[str, Union[str, List[str]]]]): List of dictionaries 
                containing feature values. For multi-valued features, values should be 
                lists of strings.
            
        Returns:
            Dict[str, float]: Dictionary mapping feature names to importance scores (0-1 scale).
            
        Raises:
            ValidationError: If features_data is None or empty, or if the processor has not been fitted
            DataError: If there's an error processing the feature data
        """
        # Validate input
        if features_data is None:
            logger.error("features_data is None")
            raise ValidationError("features_data cannot be None")
            
        if not features_data:
            logger.error("Empty features_data provided")
            raise ValidationError("features_data cannot be empty")
            
        if not self.fitted:
            logger.error("Attempted to calculate importance with unfitted processor")
            raise ValidationError("FeatureProcessor must be fitted before calculating importance")
            
        logger.info(f"Calculating feature importance based on {len(features_data)} documents")
        
        try:
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
                    logger.debug(f"Feature '{feature_name}' entropy: {entropy:.4f}")
                else:
                    entropy = 0
                    logger.debug(f"Feature '{feature_name}' has no occurrences, entropy = 0")
                    
                # Calculate coverage (percentage of documents with this feature)
                coverage = docs_with_feature / total_docs if total_docs > 0 else 0
                logger.debug(f"Feature '{feature_name}' coverage: {coverage:.2f} ({docs_with_feature}/{total_docs} docs)")
                
                # Combine entropy and coverage for importance score
                importance = 0.7 * entropy + 0.3 * coverage
                importance_scores[feature_name] = importance
                logger.debug(f"Feature '{feature_name}' raw importance score: {importance:.4f}")
            
            # Normalize importance scores
            max_importance = max(importance_scores.values()) if importance_scores else 1.0
            normalized_scores = {name: score / max_importance for name, score in importance_scores.items()}
            
            logger.info(f"Calculated importance scores for {len(normalized_scores)} features")
            for name, score in sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"Top feature importance: '{name}' = {score:.4f}")
                
            return normalized_scores
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            raise DataError(f"Failed to calculate feature importance: {str(e)}")
