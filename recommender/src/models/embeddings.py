"""
BERT-based embeddings generator for legal documents.
"""
from typing import List, Dict, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BERTEmbedder:
    """Generate and manage BERT embeddings for legal documents."""
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 cache_dir: Union[str, Path] = None,
                 device: str = None):
        """
        Initialize the BERT embedder.
        
        Args:
            model_name: Name of the sentence-transformer model to use
            cache_dir: Directory to cache embeddings
            device: Device to run model on ('cuda' or 'cpu'). If None, automatically detect.
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Automatically select device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Initializing {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        
    def generate_embeddings(self, 
                          texts: List[str], 
                          batch_size: int = 32,
                          show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings (shape: n_texts x embedding_dim)
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
    
    def combine_text_features(self,
                            summary: str,
                            keywords: List[str],
                            summary_weight: float = 0.7) -> np.ndarray:
        """
        Combine summary and keyword embeddings with weights.
        
        Args:
            summary: Document summary text
            keywords: List of keywords
            summary_weight: Weight for summary embedding (1 - this will be keyword weight)
            
        Returns:
            Combined embedding vector
        """
        if not summary or not keywords:
            raise ValueError("Both summary and keywords must be provided")
            
        # Generate embeddings
        summary_emb = self.generate_embeddings([summary], show_progress=False)[0]
        keyword_text = ' '.join(keywords)
        keyword_emb = self.generate_embeddings([keyword_text], show_progress=False)[0]
        
        # Combine with weights
        keyword_weight = 1.0 - summary_weight
        combined_emb = summary_weight * summary_emb + keyword_weight * keyword_emb
        
        # Normalize
        return combined_emb / np.linalg.norm(combined_emb)
    
    def cache_embeddings(self, 
                        document_id: str,
                        embedding: np.ndarray) -> None:
        """
        Cache embeddings to disk.
        
        Args:
            document_id: Unique identifier for the document
            embedding: Embedding vector to cache
        """
        if not self.cache_dir:
            return
            
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"{document_id}.npy"
        np.save(cache_file, embedding)
        
    def load_cached_embedding(self, document_id: str) -> Union[np.ndarray, None]:
        """
        Load cached embedding if it exists.
        
        Args:
            document_id: Unique identifier for the document
            
        Returns:
            Embedding vector if found, None otherwise
        """
        if not self.cache_dir:
            return None
            
        cache_file = self.cache_dir / f"{document_id}.npy"
        if cache_file.exists():
            return np.load(cache_file)
        return None
