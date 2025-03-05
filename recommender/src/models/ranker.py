"""
Rank documents based on combined similarity to query profiles.
"""
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime

from .embeddings import BERTEmbedder
from .features import FeatureProcessor
from .similarity import SimilarityComputer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentRanker:
    """Rank documents based on similarity to query profiles."""
    
    def __init__(self,
                 embedder: BERTEmbedder,
                 feature_processor: FeatureProcessor,
                 similarity_computer: SimilarityComputer,
                 cache_dir: Optional[Union[str, Path]] = None):
        """
        Initialize document ranker.
        
        Args:
            embedder: BERTEmbedder instance
            feature_processor: FeatureProcessor instance
            similarity_computer: SimilarityComputer instance
            cache_dir: Directory to cache document vectors
        """
        self.embedder = embedder
        self.feature_processor = feature_processor
        self.similarity_computer = similarity_computer
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Document cache
        self.document_vectors = {}
        
    def process_document(self,
                        doc_id: str,
                        summary: str,
                        keywords: List[str],
                        features: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single document into text and categorical vectors.
        
        Args:
            doc_id: Document identifier
            summary: Document summary
            keywords: Document keywords
            features: Document categorical features
            
        Returns:
            Tuple of (text_embedding, categorical_features)
        """
        # Check cache first
        if doc_id in self.document_vectors:
            return self.document_vectors[doc_id]
            
        # Generate text embedding
        text_embedding = self.embedder.combine_text_features(summary, keywords)
        
        # Process categorical features
        categorical_features = self.feature_processor.encode_features(features)
        
        # Cache results
        self.document_vectors[doc_id] = (text_embedding, categorical_features)
        
        return text_embedding, categorical_features
        
    def process_documents(self,
                         documents: List[Dict]) -> None:
        """
        Process and index a batch of documents.
        
        Args:
            documents: List of document dictionaries containing:
                     - id: Document identifier
                     - summary: Document summary
                     - keywords: List of keywords
                     - features: Dictionary of categorical features
        """
        text_embeddings = []
        categorical_features = []
        
        for doc in documents:
            text_emb, cat_feat = self.process_document(
                doc['id'],
                doc['summary'],
                doc['keywords'],
                doc['features']
            )
            text_embeddings.append(text_emb)
            categorical_features.append(cat_feat)
            
        # Convert to arrays
        text_embeddings = np.array(text_embeddings)
        categorical_features = np.array(categorical_features)
        
        # Build similarity index
        self.similarity_computer.build_index(text_embeddings, categorical_features)
        
    def rank_documents(self,
                      query_profile: Dict,
                      top_k: int = 10,
                      min_similarity: float = 0.0) -> List[Tuple[str, float]]:
        """
        Rank documents based on similarity to query profile.
        
        Args:
            query_profile: Dictionary containing:
                         - interests: Text description of interests
                         - keywords: List of interest keywords
                         - features: Dictionary of categorical preferences
            top_k: Number of recommendations to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (document_id, similarity_score) tuples
        """
        # Process query
        query_text_emb = self.embedder.combine_text_features(
            query_profile['interests'],
            query_profile['keywords']
        )
        query_categorical = self.feature_processor.encode_features(
            query_profile['features']
        )
        
        # Find similar documents
        indices, similarities = self.similarity_computer.find_similar(
            query_text_emb,
            query_categorical,
            k=top_k
        )
        
        # Filter by similarity threshold
        valid_idx = similarities >= min_similarity
        indices = indices[valid_idx]
        similarities = similarities[valid_idx]
        
        # Get document IDs
        doc_ids = list(self.document_vectors.keys())
        results = [(doc_ids[idx], float(sim)) for idx, sim in zip(indices, similarities)]
        
        return results
        
    def save_state(self, directory: Union[str, Path]) -> None:
        """
        Save ranker state to directory.
        
        Args:
            directory: Directory to save state in
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save similarity index
        self.similarity_computer.save_index(directory / 'similarity_index.faiss')
        
        # Save document vectors
        np.save(directory / 'document_vectors.npy', self.document_vectors)
        
    @classmethod
    def load_state(cls,
                  directory: Union[str, Path],
                  embedder: BERTEmbedder,
                  feature_processor: FeatureProcessor) -> 'DocumentRanker':
        """
        Load ranker state from directory.
        
        Args:
            directory: Directory containing saved state
            embedder: BERTEmbedder instance
            feature_processor: FeatureProcessor instance
            
        Returns:
            Loaded DocumentRanker instance
        """
        directory = Path(directory)
        
        # Create similarity computer and load index
        similarity_computer = SimilarityComputer()
        similarity_computer.load_index(directory / 'similarity_index.faiss')
        
        # Create ranker
        ranker = cls(embedder, feature_processor, similarity_computer)
        
        # Load document vectors
        ranker.document_vectors = np.load(directory / 'document_vectors.npy',
                                        allow_pickle=True).item()
        
        return ranker
