"""
EU Legal Document Recommender System - Utilities

This package contains utility functions and classes for the EU legal document recommender system.
"""

from .db_connector import DocumentDBConnector, get_connector
# Import only what's needed from modules
# generate_embeddings is a script, not a module with classes
from .pinecone_embeddings import PineconeEmbeddingManager

__all__ = [
    'DocumentDBConnector',
    'get_connector',

    'PineconeEmbeddingManager'
]
