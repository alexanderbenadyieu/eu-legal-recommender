"""Test fixtures for the EU Legal Recommender system."""
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models.features import FeatureProcessor


@pytest.fixture
def feature_weights():
    """Return sample feature weights for testing."""
    return {
        "document_type": 0.3,
        "subject_matter": 0.4,
        "author": 0.2,
        "form": 0.1,
    }


@pytest.fixture
def multi_valued_features():
    """Return sample multi-valued features for testing."""
    return ["subject_matter"]


@pytest.fixture
def feature_processor(feature_weights, multi_valued_features):
    """Return a sample feature processor for testing."""
    return FeatureProcessor(
        feature_weights=feature_weights,
        multi_valued_features=multi_valued_features
    )


@pytest.fixture
def sample_documents():
    """Return sample documents for testing."""
    return [
        {
            "id": "doc1",
            "title": "Regulation on Renewable Energy",
            "summary": "This regulation establishes a framework for renewable energy.",
            "keywords": ["renewable", "energy", "regulation"],
            "document_type": "regulation",
            "subject_matter": ["energy", "environment"],
            "author": "European Commission",
            "form": "regulation",
        },
        {
            "id": "doc2",
            "title": "Directive on Energy Efficiency",
            "summary": "This directive establishes measures for energy efficiency.",
            "keywords": ["energy", "efficiency", "directive"],
            "document_type": "directive",
            "subject_matter": ["energy"],
            "author": "European Parliament",
            "form": "directive",
        },
    ]


@pytest.fixture
def sample_embeddings():
    """Return sample embeddings for testing."""
    # Create deterministic but random-looking embeddings
    np.random.seed(42)
    return {
        "doc1": np.random.rand(384),
        "doc2": np.random.rand(384),
    }


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        old_cache_dir = os.environ.get("CACHE_DIR")
        os.environ["CACHE_DIR"] = temp_dir
        yield Path(temp_dir)
        if old_cache_dir:
            os.environ["CACHE_DIR"] = old_cache_dir
        else:
            del os.environ["CACHE_DIR"]


@pytest.fixture
def mock_pinecone():
    """Mock the Pinecone client."""
    with patch("pinecone.Pinecone") as mock:
        # Configure the mock
        pinecone_mock = MagicMock()
        mock.return_value = pinecone_mock
        
        # Mock the index existence check
        mock_list_indexes = MagicMock()
        mock_list_indexes.names.return_value = ["test-index"]
        pinecone_mock.list_indexes.return_value = mock_list_indexes
        
        # Mock the index
        index_mock = MagicMock()
        pinecone_mock.Index.return_value = index_mock
        
        # Configure query response
        index_mock.query.return_value = {
            "matches": [
                {"id": "doc1", "score": 0.9, "metadata": {"document_type": "regulation"}},
                {"id": "doc2", "score": 0.7, "metadata": {"document_type": "directive"}},
            ]
        }
        
        yield pinecone_mock, index_mock
