"""
Tests for the BERTEmbedder class.
"""
import unittest
import numpy as np
from pathlib import Path
import sys
import os
import tempfile
import shutil

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.embeddings import BERTEmbedder
from src.config import EMBEDDER

class TestBERTEmbedder(unittest.TestCase):
    """Test cases for the BERTEmbedder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for cache
        self.temp_dir = tempfile.mkdtemp()
        # Initialize the embedder with a small model for faster tests
        self.model_name = 'sentence-transformers/all-MiniLM-L6-v2'  # Smaller model for testing
        self.embedder = BERTEmbedder(
            model_name=self.model_name,
            cache_dir=self.temp_dir
        )
        # The actual embedding dimension of the test model (384 for all-MiniLM-L6-v2)
        self.actual_embedding_dim = 384
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test that the embedder initializes correctly."""
        # Check that the model name is set correctly
        self.assertEqual(self.embedder.model_name, self.model_name)
        # Check that the cache directory is set correctly
        self.assertEqual(str(self.embedder.cache_dir), self.temp_dir)
        # For tests, we're using a different model with different dimensions
        # so we don't check the embedding_dim against the config value
    
    def test_generate_embeddings_single_text(self):
        """Test generating embeddings for a single text."""
        text = "This is a test document about EU legal regulations."
        embedding = self.embedder.generate_embeddings([text], show_progress=False)[0]
        
        # Check that the embedding has the correct shape
        self.assertEqual(embedding.shape, (self.actual_embedding_dim,))
        # Check that the embedding is normalized
        self.assertAlmostEqual(np.linalg.norm(embedding), 1.0, places=5)
    
    def test_generate_embeddings_multiple_texts(self):
        """Test generating embeddings for multiple texts."""
        texts = [
            "This is the first test document about EU legal regulations.",
            "This is the second test document about environmental policy.",
            "This is the third test document about data protection."
        ]
        embeddings = self.embedder.generate_embeddings(texts, show_progress=False)
        
        # Check that the embeddings have the correct shape
        self.assertEqual(embeddings.shape, (len(texts), self.actual_embedding_dim))
        # Check that all embeddings are normalized
        for embedding in embeddings:
            self.assertAlmostEqual(np.linalg.norm(embedding), 1.0, places=5)
    
    def test_generate_embeddings_empty_list(self):
        """Test generating embeddings for an empty list."""
        embeddings = self.embedder.generate_embeddings([], show_progress=False)
        
        # Check that the result is an empty array
        self.assertEqual(embeddings.shape, (0,))
    
    def test_combine_text_features(self):
        """Test combining summary and keyword embeddings."""
        summary = "This is a test summary about EU legal regulations."
        keywords = ["EU", "legal", "regulations", "test"]
        
        combined_embedding = self.embedder.combine_text_features(
            summary=summary,
            keywords=keywords,
            fixed_summary_weight=0.7
        )
        
        # Check that the combined embedding has the correct shape
        self.assertEqual(combined_embedding.shape, (self.actual_embedding_dim,))
        # Check that the combined embedding is normalized
        self.assertAlmostEqual(np.linalg.norm(combined_embedding), 1.0, places=5)
    
    def test_cache_and_load_embeddings(self):
        """Test caching and loading embeddings."""
        document_id = "test_doc_123"
        embedding = np.random.rand(self.actual_embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        
        # Cache the embedding
        self.embedder.cache_embeddings(document_id, embedding)
        
        # Load the cached embedding
        loaded_embedding = self.embedder.load_cached_embedding(document_id)
        
        # Check that the loaded embedding is the same as the original
        np.testing.assert_array_almost_equal(embedding, loaded_embedding)
    
    def test_load_nonexistent_cached_embedding(self):
        """Test loading a nonexistent cached embedding."""
        loaded_embedding = self.embedder.load_cached_embedding("nonexistent_doc")
        
        # Check that the result is None
        self.assertIsNone(loaded_embedding)

if __name__ == '__main__':
    unittest.main()
