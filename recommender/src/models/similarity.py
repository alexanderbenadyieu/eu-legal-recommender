"""
Compute similarities between documents using text embeddings and categorical features.
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from pathlib import Path
import faiss
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimilarityComputer:
    """Compute similarities between documents and queries."""
    
    def __init__(self,
                 text_weight: float = 0.7,
                 categorical_weight: float = 0.3,
                 use_faiss: bool = True):
        """
        Initialize similarity computer.
        
        Args:
            text_weight: Weight for text embedding similarity
            categorical_weight: Weight for categorical feature similarity
            use_faiss: Whether to use FAISS for faster similarity search
        """
        if not np.isclose(text_weight + categorical_weight, 1.0):
            raise ValueError("Weights must sum to 1.0")
            
        self.text_weight = text_weight
        self.categorical_weight = categorical_weight
        self.use_faiss = use_faiss
        self.index = None
        
    def compute_similarity(self,
                         doc_text_embedding: np.ndarray,
                         doc_categorical: np.ndarray,
                         query_text_embedding: np.ndarray,
                         query_categorical: np.ndarray) -> float:
        """
        Compute weighted similarity between a document and query.
        
        Args:
            doc_text_embedding: Document's text embedding
            doc_categorical: Document's categorical features
            query_text_embedding: Query's text embedding
            query_categorical: Query's categorical features
            
        Returns:
            Combined similarity score
        """
        # Ensure vectors are 2D
        doc_text_embedding = doc_text_embedding.reshape(1, -1)
        doc_categorical = doc_categorical.reshape(1, -1)
        query_text_embedding = query_text_embedding.reshape(1, -1)
        query_categorical = query_categorical.reshape(1, -1)
        
        # Compute similarities
        text_sim = cosine_similarity(doc_text_embedding, query_text_embedding)[0, 0]
        cat_sim = cosine_similarity(doc_categorical, query_categorical)[0, 0]
        
        # Combine with weights
        return self.text_weight * text_sim + self.categorical_weight * cat_sim
        
    def build_index(self,
                   text_embeddings: np.ndarray,
                   categorical_features: np.ndarray) -> None:
        """
        Build FAISS index for fast similarity search.
        
        Args:
            text_embeddings: Matrix of text embeddings (n_docs x embedding_dim)
            categorical_features: Matrix of categorical features (n_docs x feature_dim)
        """
        if not self.use_faiss:
            return
            
        n_docs = len(text_embeddings)
        text_dim = text_embeddings.shape[1]
        cat_dim = categorical_features.shape[1]
        
        # Normalize and weight the features
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1)[:, np.newaxis]
        categorical_features = categorical_features / np.linalg.norm(categorical_features, axis=1)[:, np.newaxis]
        
        # Combine features
        combined_features = np.hstack([
            self.text_weight * text_embeddings,
            self.categorical_weight * categorical_features
        ])
        
        # Build FAISS index
        self.index = faiss.IndexFlatIP(text_dim + cat_dim)
        self.index.add(combined_features.astype('float32'))
        
    def find_similar(self,
                    query_text_embedding: np.ndarray,
                    query_categorical: np.ndarray,
                    k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k most similar documents using FAISS.
        
        Args:
            query_text_embedding: Query text embedding
            query_categorical: Query categorical features
            k: Number of similar documents to return
            
        Returns:
            Tuple of (indices, distances) for most similar documents
        """
        if not self.use_faiss or self.index is None:
            raise ValueError("FAISS index not built. Call build_index first.")
            
        # Normalize and weight query features
        query_text_embedding = query_text_embedding / np.linalg.norm(query_text_embedding)
        query_categorical = query_categorical / np.linalg.norm(query_categorical)
        
        # Combine query features
        query = np.hstack([
            self.text_weight * query_text_embedding,
            self.categorical_weight * query_categorical
        ])
        
        # Search index
        distances, indices = self.index.search(
            query.astype('float32').reshape(1, -1),
            k
        )
        return indices[0], distances[0]
        
    def save_index(self, path: Path) -> None:
        """Save FAISS index to disk."""
        if self.index is not None:
            faiss.write_index(self.index, str(path))
            
    def load_index(self, path: Path) -> None:
        """Load FAISS index from disk."""
        self.index = faiss.read_index(str(path))
