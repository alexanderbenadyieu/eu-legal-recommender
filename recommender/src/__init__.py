"""
EU Legal Document Recommender System - Core Module

This package contains the core functionality for the EU legal document recommender system.
"""

# Import main components for easier access
from .models.embeddings import BERTEmbedder
from .models.features import FeatureProcessor
from .models.similarity import SimilarityComputer
from .models.pinecone_recommender import PineconeRecommender
from .models.ranker import DocumentRanker
from .models.user_profile import UserProfile
from .models.personalized_recommender import PersonalizedRecommender

__all__ = [
    'BERTEmbedder',
    'FeatureProcessor',
    'SimilarityComputer',
    'PineconeRecommender',
    'DocumentRanker',
    'UserProfile',
    'PersonalizedRecommender'
]
