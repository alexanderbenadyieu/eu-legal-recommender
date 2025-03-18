"""
EU Legal Document Recommender System - Core Module

This package contains the core functionality for the EU legal document recommender system.
"""

# Import main components for easier access
from recommender.src.models.embeddings import TextEmbedder
from recommender.src.models.features import FeatureProcessor
from recommender.src.models.similarity import SimilarityCalculator
from recommender.src.models.pinecone_recommender import PineconeRecommender
from recommender.src.models.ranker import DocumentRanker

__all__ = [
    'TextEmbedder',
    'FeatureProcessor',
    'SimilarityCalculator',
    'PineconeRecommender',
    'DocumentRanker'
]
