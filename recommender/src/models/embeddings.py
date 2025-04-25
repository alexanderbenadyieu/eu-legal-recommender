"""
BERT-based embeddings generator for legal documents.

This module provides the BERTEmbedder class which handles the generation of
embeddings for legal documents using the Legal-BERT model. It supports
batch processing, text preprocessing, caching, and device configuration.
"""
from typing import List, Dict, Union, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import torch
from sklearn.decomposition import PCA

# Import configuration and utilities
from ..config import EMBEDDER, LOGS_DIR
from ..utils.logging import get_logger
from ..utils.exceptions import EmbeddingError, ValidationError

# Set up logger for this module
logger = get_logger(__name__)

class BERTEmbedder:
    """Generate and manage BERT embeddings for legal documents.
    
    This class provides functionality for generating embeddings from text using
    BERT-based models (particularly Legal-BERT). It handles model loading,
    text preprocessing, batch processing, and embedding operations like
    combining text features.
    
    Attributes:
        model_name (str): Name of the sentence-transformer model being used
        model (SentenceTransformer): The loaded transformer model
        embedding_dim (int): Dimension of the generated embeddings
        batch_size (int): Default batch size for processing
        cache_dir (Path, optional): Directory for caching embeddings if enabled
    """
    
    def __init__(self, 
                 model_name: str = None,
                 cache_dir: Union[str, Path] = None,
                 device: str = None,
                 batch_size: int = None):
        """
        Initialize the BERT embedder.
        
        Args:
            model_name (str, optional): Name of the sentence-transformer model to use. 
                Defaults to config value.
            cache_dir (Union[str, Path], optional): Directory to cache embeddings.
                If provided, enables caching of embeddings to disk.
            device (str, optional): Device to run model on ('cuda' or 'cpu'). 
                If None, automatically detects available hardware.
            batch_size (int, optional): Batch size for processing. 
                Defaults to config value.
                
        Raises:
            EmbeddingError: If there's an error loading the model.
        """
        self.model_name = model_name or EMBEDDER['model_name']
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.batch_size = batch_size or EMBEDDER['batch_size']
        
        # Automatically select device
        if device is None:
            device = EMBEDDER['device']
            # If config specifies 'auto', detect available hardware
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Initializing {self.model_name} on {device}")
        
        try:
            self.model = SentenceTransformer(self.model_name, device=device)
            logger.info(f"Successfully loaded model {self.model_name}")
            
            # Set embedding dimension based on model
            self.embedding_dim = EMBEDDER['dimension']
            logger.debug(f"Embedding dimension: {self.embedding_dim}")
            
            # Create cache directory if it doesn't exist
            if self.cache_dir:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Cache directory set to {self.cache_dir}")
                
        except Exception as e:
            error_msg = f"Failed to initialize model {self.model_name}: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg)
        
    def generate_embeddings(self, 
                          texts: List[str], 
                          batch_size: int = None,
                          show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of text strings to embed
            batch_size (int, optional): Batch size for processing. 
                If None, uses the default batch size.
            show_progress (bool): Whether to show progress bar. 
                Defaults to True.
            
        Returns:
            np.ndarray: Array of embeddings with shape (n_texts, embedding_dim)
            
        Raises:
            ValidationError: If texts is None or contains invalid items
            EmbeddingError: If there's an error during embedding generation
        """
        # Validate input
        if texts is None:
            logger.error("Received None instead of texts list for embedding")
            raise ValidationError("Texts cannot be None", field="texts")
            
        if not texts:
            logger.warning("Received empty texts list for embedding")
            return np.array([])
        
        # Check for invalid text items
        invalid_indices = [i for i, text in enumerate(texts) if not isinstance(text, str)]
        if invalid_indices:
            logger.error(f"Invalid text items at indices: {invalid_indices}")
            raise ValidationError(f"All texts must be strings, found non-string items at indices: {invalid_indices}", 
                                 field="texts")
            
        # Use instance batch size if not specified
        if batch_size is None:
            batch_size = self.batch_size
            
        logger.info(f"Generating embeddings for {len(texts)} texts with batch size {batch_size}")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            logger.debug(f"Successfully generated embeddings with shape {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            error_msg = f"Error generating embeddings: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg)
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.
        
        Args:
            text (str): Text string to embed
            
        Returns:
            np.ndarray: Embedding vector with shape (embedding_dim,)
            
        Raises:
            ValidationError: If text is None or not a string
            EmbeddingError: If there's an error during embedding generation
        """
        # Validate input
        if text is None:
            logger.error("Received None instead of text for embedding")
            raise ValidationError("Text cannot be None", field="text")
            
        if not isinstance(text, str):
            logger.error(f"Received non-string input of type {type(text)}")
            raise ValidationError("Text must be a string", field="text")
            
        if not text.strip():
            logger.warning("Received empty text for embedding")
            return np.zeros(self.embedding_dim)
            
        try:
            embeddings = self.generate_embeddings([text], show_progress=False)
            return embeddings[0]
        except EmbeddingError as e:
            # Re-raise with more specific context
            raise EmbeddingError(f"Failed to generate embedding for text: {str(e)}")
        except Exception as e:
            error_msg = f"Unexpected error generating embedding: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg)
            
    def combine_text_features(self,
                            summary: str,
                            keywords: List[str],
                            fixed_summary_weight: float = None,
                            normalize: bool = True) -> np.ndarray:
        """
        Combine summary and keyword embeddings with dynamic weights based on the number of keywords.
        
        The weights are calculated as follows:
        - keyword_weight(n) = 0.1 + 0.01667 * (n - 2)
        - summary_weight(n) = 1 - keyword_weight(n)
        
        This ensures that for documents with few keywords (n=2), the summary gets more weight (0.9),
        while for documents with many keywords (n=20), the keywords get more weight (0.4).
        
        Args:
            summary (str): Document summary text
            keywords (List[str]): List of keywords
            fixed_summary_weight (float, optional): If provided, use this fixed weight instead 
                of the dynamic calculation. Defaults to None.
            normalize (bool, optional): Whether to normalize the resulting vector. Defaults to True.
            
        Returns:
            np.ndarray: Combined embedding vector
            
        Raises:
            ValidationError: If summary or keywords are invalid
            EmbeddingError: If there's an error generating embeddings
        """
        # Validate inputs
        if summary is None or keywords is None:
            logger.error("Summary or keywords is None")
            raise ValidationError("Neither summary nor keywords can be None")
            
        if not summary.strip():
            logger.error("Empty summary provided")
            raise ValidationError("Summary cannot be empty", field="summary")
            
        if not keywords:
            logger.error("Empty keywords list provided")
            raise ValidationError("Keywords list cannot be empty", field="keywords")
            
        logger.info(f"Combining embeddings for summary and {len(keywords)} keywords")
        
        try:
            # Generate embeddings
            summary_emb = self.generate_embeddings([summary], show_progress=False)[0]
            keyword_text = ' '.join(keywords)
            keyword_emb = self.generate_embeddings([keyword_text], show_progress=False)[0]
            
            # Calculate weights dynamically based on number of keywords
            if fixed_summary_weight is None:
                n = len(keywords)
                # Implement the formula: w_keyword(n) = 0.1 + 0.01667 × (n - 2)
                keyword_weight = max(0.1, min(0.4, 0.1 + 0.01667 * (n - 2)))
                summary_weight = 1.0 - keyword_weight
                logger.info(f"Dynamic weighting: {n} keywords, summary_weight={summary_weight:.4f}, keyword_weight={keyword_weight:.4f}")
            else:
                # Validate fixed weight
                if not 0 <= fixed_summary_weight <= 1:
                    logger.warning(f"Invalid fixed_summary_weight {fixed_summary_weight}, clamping to [0,1]")
                    fixed_summary_weight = max(0, min(1, fixed_summary_weight))
                    
                summary_weight = fixed_summary_weight
                keyword_weight = 1.0 - summary_weight
                logger.info(f"Fixed weighting: summary_weight={summary_weight:.4f}, keyword_weight={keyword_weight:.4f}")
            
            # Combine with weights
            combined_emb = summary_weight * summary_emb + keyword_weight * keyword_emb
            
            # Normalize if requested
            if normalize:
                norm = np.linalg.norm(combined_emb)
                if norm > 0:
                    combined_emb = combined_emb / norm
                    logger.debug("Normalized combined embedding")
                else:
                    logger.warning("Zero norm for combined embedding, cannot normalize")
                
            return combined_emb
            
        except EmbeddingError as e:
            # Re-raise with more specific context
            raise EmbeddingError(f"Failed to combine text features: {str(e)}")
        except Exception as e:
            error_msg = f"Unexpected error combining text features: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg)
    
    def cache_embeddings(self, 
                        document_id: str,
                        embedding: np.ndarray) -> None:
        """
        Cache embeddings to disk.
        
        Args:
            document_id (str): Unique identifier for the document
            embedding (np.ndarray): Embedding vector to cache
            
        Raises:
            ValidationError: If document_id or embedding is invalid
            IOError: If there's an error writing to the cache file
        """
        # Validate inputs
        if not document_id or not isinstance(document_id, str):
            logger.error(f"Invalid document_id: {document_id}")
            raise ValidationError("document_id must be a non-empty string", field="document_id")
            
        if embedding is None or not isinstance(embedding, np.ndarray):
            logger.error(f"Invalid embedding type: {type(embedding)}")
            raise ValidationError("embedding must be a numpy array", field="embedding")
            
        # Skip if caching is disabled
        if not self.cache_dir:
            logger.debug(f"Caching disabled, skipping cache for document {document_id}")
            return
        
        try:
            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"{document_id}.npy"
            
            # Save embedding to file
            np.save(cache_file, embedding)
            logger.debug(f"Cached embedding for document {document_id} to {cache_file}")
            
        except Exception as e:
            error_msg = f"Failed to cache embedding for document {document_id}: {str(e)}"
            logger.error(error_msg)
            raise IOError(error_msg)
        
    def load_cached_embedding(self, document_id: str) -> Union[np.ndarray, None]:
        """
        Load cached embedding if it exists.
        
        Args:
            document_id (str): Unique identifier for the document
            
        Returns:
            Union[np.ndarray, None]: Embedding vector if found, None otherwise
            
        Raises:
            ValidationError: If document_id is invalid
            IOError: If there's an error reading the cache file
        """
        # Validate input
        if not document_id or not isinstance(document_id, str):
            logger.error(f"Invalid document_id: {document_id}")
            raise ValidationError("document_id must be a non-empty string", field="document_id")
            
        # Skip if caching is disabled
        if not self.cache_dir:
            logger.debug("Caching disabled, no embeddings to load")
            return None
        
        try:
            cache_file = self.cache_dir / f"{document_id}.npy"
            
            if cache_file.exists():
                embedding = np.load(cache_file)
                logger.debug(f"Loaded cached embedding for document {document_id} from {cache_file}")
                return embedding
            else:
                logger.debug(f"No cached embedding found for document {document_id}")
                return None
                
        except Exception as e:
            error_msg = f"Failed to load cached embedding for document {document_id}: {str(e)}"
            logger.error(error_msg)
            raise IOError(error_msg)
