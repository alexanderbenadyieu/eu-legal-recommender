"""
Compute similarities between documents using text embeddings and categorical features.

This module provides the SimilarityComputer class for calculating similarity between
documents based on a weighted combination of text embeddings and categorical features.
It supports both direct similarity computation and fast similarity search using FAISS.
"""
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import faiss
import torch
import os

# Import utilities
from src.utils.logging import get_logger
from src.utils.exceptions import ValidationError, ProcessingError, ConfigurationError

# Set up logger for this module
logger = get_logger(__name__)

class SimilarityComputer:
    """Compute similarities between documents and queries.
    
    This class provides methods for computing similarity between documents based on
    a weighted combination of text embeddings and categorical features. It supports
    both direct similarity computation and fast similarity search using FAISS.
    
    Attributes:
        text_weight (float): Weight for text embedding similarity (0-1)
        categorical_weight (float): Weight for categorical feature similarity (0-1)
        use_faiss (bool): Whether to use FAISS for faster similarity search
        index (faiss.Index): FAISS index for fast similarity search
    """
    
    def __init__(self,
                 text_weight: float = 0.7,
                 categorical_weight: float = 0.3,
                 use_faiss: bool = True,
                 use_gpu: bool = False,
                 categorical_feature_weights: Optional[Dict[str, float]] = None):
        """
        Initialize similarity computer.
        
        Args:
            text_weight (float, optional): Weight for text embedding similarity. 
                Defaults to 0.7.
            categorical_weight (float, optional): Weight for categorical feature similarity. 
                Defaults to 0.3.
            use_faiss (bool, optional): Whether to use FAISS for faster similarity search. 
                Defaults to True.
                
        Raises:
            ConfigurationError: If weights don't sum to 1.0 or are outside valid range
        """
        # Validate weights
        if text_weight < 0 or text_weight > 1:
            logger.error(f"Invalid text_weight: {text_weight}. Must be between 0 and 1")
            raise ConfigurationError(f"text_weight must be between 0 and 1, got {text_weight}")
            
        if categorical_weight < 0 or categorical_weight > 1:
            logger.error(f"Invalid categorical_weight: {categorical_weight}. Must be between 0 and 1")
            raise ConfigurationError(f"categorical_weight must be between 0 and 1, got {categorical_weight}")
        
        if not np.isclose(text_weight + categorical_weight, 1.0, rtol=1e-5):
            logger.error(f"Weights must sum to 1.0: text_weight={text_weight}, categorical_weight={categorical_weight}")
            raise ConfigurationError(f"Weights must sum to 1.0, got {text_weight + categorical_weight}")
            
        self.text_weight = text_weight
        self.categorical_weight = categorical_weight
        self.use_faiss = use_faiss
        self.use_gpu = use_gpu
        self.index = None
        self.gpu_resources = None
        
        # Store categorical feature weights (for fine-grained control)
        self.categorical_feature_weights = categorical_feature_weights or {}
        
        # Initialize GPU resources if needed
        if self.use_gpu and self.use_faiss:
            try:
                self._init_gpu_resources()
            except Exception as e:
                logger.warning(f"Failed to initialize GPU resources: {str(e)}. Falling back to CPU.")
                self.use_gpu = False
        
        logger.info(f"Initialized SimilarityComputer with text_weight={text_weight}, "
                   f"categorical_weight={categorical_weight}, use_faiss={use_faiss}, "
                   f"use_gpu={self.use_gpu}, "
                   f"categorical_feature_weights={len(self.categorical_feature_weights) or 'default'}")
        
    def set_weights(self, text_weight: float, categorical_weight: float) -> None:
        """
        Set new weights for text and categorical similarity.
        
        This method updates the weights used for computing similarity and
        invalidates any existing FAISS index, as it would need to be rebuilt
        with the new weights.
        
        Args:
            text_weight (float): New weight for text embedding similarity (0-1)
            categorical_weight (float): New weight for categorical feature similarity (0-1)
            
        Raises:
            ConfigurationError: If weights don't sum to 1.0 or are outside valid range
        """
        # Validate weights
        if text_weight < 0 or text_weight > 1:
            logger.error(f"Invalid text_weight: {text_weight}. Must be between 0 and 1")
            raise ConfigurationError(f"text_weight must be between 0 and 1, got {text_weight}")
            
        if categorical_weight < 0 or categorical_weight > 1:
            logger.error(f"Invalid categorical_weight: {categorical_weight}. Must be between 0 and 1")
            raise ConfigurationError(f"categorical_weight must be between 0 and 1, got {categorical_weight}")
        
        if not np.isclose(text_weight + categorical_weight, 1.0, rtol=1e-5):
            logger.error(f"Weights must sum to 1.0: text_weight={text_weight}, categorical_weight={categorical_weight}")
            raise ConfigurationError(f"Weights must sum to 1.0, got {text_weight + categorical_weight}")
        
        # Update weights
        old_text_weight = self.text_weight
        old_categorical_weight = self.categorical_weight
        self.text_weight = text_weight
        self.categorical_weight = categorical_weight
        
        # Invalidate index if weights changed
        if self.index is not None and (old_text_weight != text_weight or old_categorical_weight != categorical_weight):
            logger.info("Weights changed, invalidating FAISS index")
            self.index = None
        
        logger.info(f"Updated weights: text_weight={text_weight:.2f}, categorical_weight={categorical_weight:.2f}")
    
    def set_categorical_feature_weights(self, feature_weights: Dict[str, float]) -> None:
        """
        Set weights for individual categorical features.
        
        This method allows fine-grained control over how different categorical
        features contribute to the overall categorical similarity score.
        
        Args:
            feature_weights (Dict[str, float]): Dictionary mapping feature names to weights
            
        Raises:
            ConfigurationError: If weights are invalid
        """
        # Validate weights
        total = sum(feature_weights.values())
        if not np.isclose(total, 1.0, rtol=1e-5):
            logger.warning(f"Categorical feature weights don't sum to 1.0 (sum={total:.2f}). Normalizing.")
            # Normalize weights
            feature_weights = {k: v / total for k, v in feature_weights.items()}
        
        # Update weights
        self.categorical_feature_weights = feature_weights
        
        # Invalidate index
        if self.index is not None:
            logger.info("Categorical feature weights changed, invalidating FAISS index")
            self.index = None
            
        logger.info(f"Updated categorical feature weights: {feature_weights}")
    
    def get_weights(self) -> Dict[str, Any]:
        """
        Get current weight configuration.
        
        Returns:
            Dict[str, Any]: Dictionary containing current weights
        """
        return {
            "text_weight": self.text_weight,
            "categorical_weight": self.categorical_weight,
            "categorical_feature_weights": self.categorical_feature_weights
        }
        
    def compute_similarity(self,
                         doc_text_embedding: np.ndarray,
                         doc_categorical: np.ndarray,
                         query_text_embedding: np.ndarray,
                         query_categorical: np.ndarray,
                         feature_names: Optional[List[str]] = None) -> float:
        """
        Compute weighted similarity between a document and query.
        
        This method calculates a combined similarity score between a document and a query
        based on a weighted combination of text embedding similarity and categorical feature
        similarity. The weights are determined by the text_weight and categorical_weight
        attributes set during initialization.
        
        Args:
            doc_text_embedding (np.ndarray): Document's text embedding vector
            doc_categorical (np.ndarray): Document's categorical features vector
            query_text_embedding (np.ndarray): Query's text embedding vector
            query_categorical (np.ndarray): Query's categorical features vector
            
        Returns:
            float: Combined similarity score between 0 and 1
            
        Raises:
            ValidationError: If any of the input vectors are None or have invalid shapes
            ProcessingError: If there's an error during similarity computation
        """
        try:
            # Validate inputs
            if doc_text_embedding is None or query_text_embedding is None:
                logger.error("Text embeddings cannot be None")
                raise ValidationError("Text embeddings cannot be None")
                
            if doc_categorical is None or query_categorical is None:
                logger.error("Categorical features cannot be None")
                raise ValidationError("Categorical features cannot be None")
            
            # Ensure vectors are 2D
            doc_text_embedding = doc_text_embedding.reshape(1, -1)
            doc_categorical = doc_categorical.reshape(1, -1)
            query_text_embedding = query_text_embedding.reshape(1, -1)
            query_categorical = query_categorical.reshape(1, -1)
            
            # Compute text similarity
            text_sim = cosine_similarity(doc_text_embedding, query_text_embedding)[0, 0]
            
            # Compute categorical similarity
            cat_sim = cosine_similarity(doc_categorical, query_categorical)[0, 0]
            
            # Apply feature-level weights if provided
            if feature_names and self.categorical_feature_weights and len(doc_categorical[0]) == len(feature_names):
                # Calculate weighted categorical similarity
                weighted_cat_sim = 0.0
                for i, feature_name in enumerate(feature_names):
                    if feature_name in self.categorical_feature_weights:
                        weight = self.categorical_feature_weights[feature_name]
                        feature_sim = doc_categorical[0, i] * query_categorical[0, i]
                        weighted_cat_sim += weight * feature_sim
                # Use weighted similarity if we have weights for all features
                if len(self.categorical_feature_weights) == len(feature_names):
                    cat_sim = weighted_cat_sim
            
            # Combine with weights
            combined_sim = self.text_weight * text_sim + self.categorical_weight * cat_sim
            
            logger.debug(f"Computed similarity: text_sim={text_sim:.4f}, cat_sim={cat_sim:.4f}, combined={combined_sim:.4f}")
            return combined_sim
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error computing similarity: {str(e)}")
            raise ProcessingError(f"Failed to compute similarity: {str(e)}")
        
    def build_index(self,
                   text_embeddings: np.ndarray,
                   categorical_features: np.ndarray) -> None:
        """
        Build FAISS index for fast similarity search.
        
        This method builds a FAISS index for efficient similarity search using the provided
        text embeddings and categorical features. The index combines both feature types
        with appropriate weights for unified similarity computation. If use_faiss is False,
        this method will return without building an index.
        
        Args:
            text_embeddings (np.ndarray): Matrix of text embeddings (n_docs x embedding_dim)
            categorical_features (np.ndarray): Matrix of categorical features (n_docs x feature_dim)
            
        Raises:
            ValidationError: If inputs are None, empty, or have mismatched dimensions
            ProcessingError: If there's an error during index building
        """
        # Skip if FAISS is disabled
        if not self.use_faiss:
            logger.info("FAISS is disabled, skipping index building")
            return
            
        try:
            # Validate inputs
            if text_embeddings is None or categorical_features is None:
                logger.error("Embeddings or features cannot be None")
                raise ValidationError("Embeddings or features cannot be None")
                
            if len(text_embeddings) == 0 or len(categorical_features) == 0:
                logger.error("Embeddings or features cannot be empty")
                raise ValidationError("Embeddings or features cannot be empty")
                
            if len(text_embeddings) != len(categorical_features):
                logger.error(f"Mismatched dimensions: {len(text_embeddings)} text embeddings vs "
                           f"{len(categorical_features)} categorical features")
                raise ValidationError("Number of text embeddings and categorical features must match")
            
            n_docs = len(text_embeddings)
            text_dim = text_embeddings.shape[1]
            cat_dim = categorical_features.shape[1]
            
            logger.info(f"Building FAISS index for {n_docs} documents with "
                      f"{text_dim} text dimensions and {cat_dim} categorical dimensions")
            
            # Normalize and weight the features
            text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1)[:, np.newaxis]
            categorical_features = categorical_features / np.linalg.norm(categorical_features, axis=1)[:, np.newaxis]
            
            # Combine features
            combined_features = np.hstack([
                self.text_weight * text_embeddings,
                self.categorical_weight * categorical_features
            ])
            
            # Build FAISS index
            if self.use_gpu and torch.cuda.is_available():
                # Create index on CPU first
                cpu_index = faiss.IndexFlatIP(text_dim + cat_dim)
                
                # Move to GPU if available
                if self.gpu_resources is None:
                    self._init_gpu_resources()
                    
                if self.gpu_resources is not None:
                    self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, cpu_index)
                    logger.info("Using GPU-enabled FAISS index")
                else:
                    self.index = cpu_index
                    logger.warning("GPU requested but not available, using CPU index")
            else:
                self.index = faiss.IndexFlatIP(text_dim + cat_dim)
                
            self.index.add(combined_features.astype('float32'))
            
            logger.info(f"Successfully built FAISS index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {str(e)}")
            raise ProcessingError(f"Failed to build FAISS index: {str(e)}")
        
    def find_similar(self,
                    query_text_embedding: np.ndarray,
                    query_categorical: np.ndarray,
                    k: int = 10,
                    feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k most similar documents using FAISS.
        
        This method searches the FAISS index for the k most similar documents to the
        provided query embeddings. It combines text and categorical features with
        appropriate weights for the search. The FAISS index must be built before calling
        this method.
        
        Args:
            query_text_embedding (np.ndarray): Query text embedding vector
            query_categorical (np.ndarray): Query categorical features vector
            k (int, optional): Number of similar documents to return. Defaults to 10.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (indices, distances) for most similar documents.
                Indices are the positions in the original data used to build the index.
                Distances are similarity scores (higher is more similar).
            
        Raises:
            ValidationError: If inputs are None or invalid, or if k is not positive
            ConfigurationError: If FAISS is disabled or index is not built
            ProcessingError: If there's an error during similarity search
        """
        try:
            # Validate inputs
            if query_text_embedding is None or query_categorical is None:
                logger.error("Query embeddings or features cannot be None")
                raise ValidationError("Query embeddings or features cannot be None")
                
            if k <= 0:
                logger.error(f"Invalid k value: {k}. Must be positive.")
                raise ValidationError(f"k must be positive, got {k}")
                
            if not self.use_faiss:
                logger.error("FAISS is disabled, cannot perform similarity search")
                raise ConfigurationError("FAISS is disabled, cannot perform similarity search")
                
            if self.index is None:
                logger.error("FAISS index not built. Call build_index first.")
                raise ConfigurationError("FAISS index not built. Call build_index first.")
            
            # Normalize and weight query features
            query_text_embedding = query_text_embedding / np.linalg.norm(query_text_embedding)
            query_categorical = query_categorical / np.linalg.norm(query_categorical)
            
            # Combine query features
            query = np.hstack([
                self.text_weight * query_text_embedding,
                self.categorical_weight * query_categorical
            ])
            
            logger.debug(f"Searching for {k} most similar documents")
            
            # Search index
            distances, indices = self.index.search(
                query.astype('float32').reshape(1, -1),
                k
            )
            
            logger.debug(f"Found {len(indices[0])} similar documents with similarity scores: "
                       f"{', '.join([f'{d:.4f}' for d in distances[0][:5]])}...")
            
            return indices[0], distances[0]
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error finding similar documents: {str(e)}")
            raise ProcessingError(f"Failed to find similar documents: {str(e)}")
        
    def save_index(self, path: Union[str, Path]) -> None:
        """
        Save FAISS index to disk.
        
        This method saves the FAISS index to the specified path for later use. The index
        must have been built before calling this method.
        
        Args:
            path (Union[str, Path]): Path where the index will be saved
            
        Raises:
            ConfigurationError: If the index has not been built
            ProcessingError: If there's an error during saving
        """
        try:
            # Convert string path to Path object if necessary
            if isinstance(path, str):
                path = Path(path)
                
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.index is None:
                logger.error("Cannot save index: index has not been built")
                raise ConfigurationError("Cannot save index: index has not been built")
                
            logger.info(f"Saving FAISS index to {path}")
            faiss.write_index(self.index, str(path))
            logger.info(f"Successfully saved FAISS index with {self.index.ntotal} vectors to {path}")
            
        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}")
            raise ProcessingError(f"Failed to save FAISS index: {str(e)}")
            
    def _init_gpu_resources(self) -> None:
        """
        Initialize GPU resources for FAISS.
        
        This method sets up the GPU resources for FAISS to use. It's called
        automatically during initialization if use_gpu is True.
        
        Raises:
            ProcessingError: If there's an error initializing GPU resources
        """
        try:
            # Check if GPU is available
            if not torch.cuda.is_available():
                logger.warning("CUDA is not available. Falling back to CPU.")
                self.use_gpu = False
                return
                
            # Get number of available GPUs
            gpu_count = torch.cuda.device_count()
            if gpu_count == 0:
                logger.warning("No CUDA devices found. Falling back to CPU.")
                self.use_gpu = False
                return
                
            # Initialize GPU resources
            self.gpu_resources = faiss.StandardGpuResources()
            logger.info(f"Successfully initialized GPU resources with {gpu_count} CUDA devices")
            
            # Set environment variable to suppress warnings
            os.environ['FAISS_NO_TEMP_MEMORY'] = '1'
            
        except Exception as e:
            logger.error(f"Error initializing GPU resources: {str(e)}")
            self.use_gpu = False
            self.gpu_resources = None
            raise ProcessingError(f"Failed to initialize GPU resources: {str(e)}")
    
    def load_index(self, path: Union[str, Path]) -> None:
        """
        Load FAISS index from disk.
        
        This method loads a previously saved FAISS index from the specified path.
        
        Args:
            path (Union[str, Path]): Path to the saved index file
            
        Raises:
            ValidationError: If the path does not exist or is not a valid index file
            ProcessingError: If there's an error during loading
        """
        try:
            # Convert string path to Path object if necessary
            if isinstance(path, str):
                path = Path(path)
                
            if not path.exists():
                logger.error(f"Index file does not exist: {path}")
                raise ValidationError(f"Index file does not exist: {path}")
                
            logger.info(f"Loading FAISS index from {path}")
            cpu_index = faiss.read_index(str(path))
            
            # Move to GPU if requested and available
            if self.use_gpu and torch.cuda.is_available():
                if self.gpu_resources is None:
                    self._init_gpu_resources()
                    
                if self.gpu_resources is not None:
                    self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, cpu_index)
                    logger.info("Moved FAISS index to GPU")
                else:
                    self.index = cpu_index
                    logger.warning("GPU requested but not available, using CPU index")
            else:
                self.index = cpu_index
                
            self.use_faiss = True  # Enable FAISS when loading an index
            logger.info(f"Successfully loaded FAISS index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            raise ProcessingError(f"Failed to load FAISS index: {str(e)}")
