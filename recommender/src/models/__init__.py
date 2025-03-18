"""
EU Legal Document Recommender System - Models

This package contains the core models for the EU legal document recommender system.
"""

from .embeddings import BERTEmbedder
from .features import FeatureProcessor
from .similarity import SimilarityComputer
from .pinecone_recommender import PineconeRecommender
from .ranker import DocumentRanker

__all__ = [
    'BERTEmbedder',
    'FeatureProcessor',
    'SimilarityComputer',
    'PineconeRecommender',
    'DocumentRanker'
]
