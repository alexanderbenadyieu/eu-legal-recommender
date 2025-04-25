"""
Tests for the PineconeRecommender class.
"""
import unittest
import numpy as np
from pathlib import Path
import sys
import os
import json
from unittest.mock import MagicMock, patch, Mock

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.pinecone_recommender import PineconeRecommender
from src.models.features import FeatureProcessor
from src.config import SIMILARITY, EMBEDDER

class TestPineconeRecommender(unittest.TestCase):
    """Test cases for the PineconeRecommender class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the Pinecone client and index
        self.pinecone_mock = MagicMock()
        self.index_mock = MagicMock()
        
        # Add describe_index_stats method to mock
        self.index_mock.describe_index_stats.return_value = {
            'dimension': 384,
            'index_fullness': 0.0,
            'namespaces': {},
            'total_vector_count': 0
        }
        
        # Sample query results from Pinecone
        self.sample_results = {
            'matches': [
                {
                    'id': 'doc1',
                    'score': 0.95,
                    'metadata': {
                        'celex_number': '32018L2001',
                        'title': 'Renewable Energy Directive',
                        'document_type': 'directive',
                        'subject_matters': ['energy', 'environment', 'climate change'],
                        'author': 'European Parliament',
                        'form': 'legislative'
                    }
                },
                {
                    'id': 'doc2',
                    'score': 0.85,
                    'metadata': {
                        'celex_number': '32019R0943',
                        'title': 'Electricity Market Regulation',
                        'document_type': 'regulation',
                        'subject_matters': ['energy', 'market'],
                        'author': 'European Commission',
                        'form': 'legislative'
                    }
                },
                {
                    'id': 'doc3',
                    'score': 0.75,
                    'metadata': {
                        'celex_number': '32012L0027',
                        'title': 'Energy Efficiency Directive',
                        'document_type': 'directive',
                        'subject_matters': ['energy', 'efficiency'],
                        'author': 'European Parliament',
                        'form': 'legislative'
                    }
                }
            ]
        }
        
        # Configure the mock index to return the sample results
        self.index_mock.query.return_value = self.sample_results
        
        # Configure the mock Pinecone client to return the mock index
        self.pinecone_mock.Index.return_value = self.index_mock
        
        # Initialize the feature processor
        self.feature_processor = FeatureProcessor(
            feature_weights={
                'document_type': 0.4,
                'subject_matters': 0.3,
                'author': 0.2,
                'form': 0.1
            },
            multi_valued_features=['subject_matters']
        )
        
        # Mock the pinecone module and Index class
        self.pinecone_patcher = patch('pinecone.Pinecone')
        self.mock_pinecone = self.pinecone_patcher.start()
        self.mock_pinecone.return_value = self.pinecone_mock
        
        # Mock the index existence check
        mock_list_indexes = MagicMock()
        mock_list_indexes.names.return_value = ['test-index']
        self.pinecone_mock.list_indexes.return_value = mock_list_indexes
        self.pinecone_mock.Index.return_value = self.index_mock
        
        # Initialize the recommender with the mock Pinecone client
        self.recommender = PineconeRecommender(
            api_key='fake_api_key',
            index_name='test-index',
            embedder_model='sentence-transformers/all-MiniLM-L6-v2',  # Smaller model for testing
            feature_processor=self.feature_processor,
            text_weight=SIMILARITY['text_weight'],
            categorical_weight=SIMILARITY['categorical_weight']
        )
        
        # Replace the real embedder with a mock
        self.recommender.embedder = MagicMock()
        # Configure the mock embedder to return a random vector
        self.recommender.embedder.generate_embeddings.return_value = np.random.rand(384)
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop the patcher
        self.pinecone_patcher.stop()
    
    def test_initialization(self):
        """Test that the recommender initializes correctly."""
        # Check that the index name is set correctly
        self.assertEqual(self.recommender.index_name, 'test-index')
        # Check that the weights are set correctly
        self.assertEqual(self.recommender.text_weight, SIMILARITY['text_weight'])
        self.assertEqual(self.recommender.categorical_weight, SIMILARITY['categorical_weight'])
    
    def test_get_recommendations(self):
        """Test getting recommendations based on a text query."""
        # Get recommendations
        recommendations = self.recommender.get_recommendations(
            query_text="renewable energy policy",
            top_k=3
        )
        
        # Check that the correct number of recommendations is returned
        self.assertEqual(len(recommendations), 3)
        
        # Check that the recommendations are sorted by score
        self.assertGreaterEqual(recommendations[0]['score'], recommendations[1]['score'])
        self.assertGreaterEqual(recommendations[1]['score'], recommendations[2]['score'])
        
        # Check that the recommendations have the correct structure
        for rec in recommendations:
            self.assertIn('id', rec)
            self.assertIn('score', rec)
            self.assertIn('metadata', rec)
    
    def test_get_recommendations_with_filter(self):
        """Test getting recommendations with a document type filter."""
        # Get recommendations with filter
        recommendations = self.recommender.get_recommendations(
            query_text="renewable energy policy",
            top_k=3,
            filter_document_type="directive"
        )
        
        # Check that all recommendations have the correct document type
        for rec in recommendations:
            if 'metadata' in rec and 'document_type' in rec['metadata']:
                self.assertEqual(rec['metadata']['document_type'], "directive")
    
    def test_get_similar_documents(self):
        """Test getting similar documents based on a document ID."""
        # Configure the mock embedder to return a document embedding
        self.recommender.get_document_embedding = MagicMock(return_value=np.random.rand(768))
        
        # Get similar documents
        similar_docs = self.recommender.get_similar_documents(
            document_id="32018L2001",
            top_k=3
        )
        
        # Check that the correct number of similar documents is returned
        self.assertEqual(len(similar_docs), 3)
        
        # Check that the similar documents are sorted by score
        self.assertGreaterEqual(similar_docs[0]['score'], similar_docs[1]['score'])
        self.assertGreaterEqual(similar_docs[1]['score'], similar_docs[2]['score'])
        
        # Check that the similar documents have the correct structure
        for doc in similar_docs:
            self.assertIn('id', doc)
            self.assertIn('score', doc)
            self.assertIn('metadata', doc)
    
    def test_combine_text_and_categorical_scores(self):
        """Test combining text and categorical scores."""
        # Sample scores
        text_scores = {
            'doc1': 0.9,
            'doc2': 0.8,
            'doc3': 0.7
        }
        categorical_scores = {
            'doc1': 0.6,
            'doc2': 0.7,
            'doc3': 0.8
        }
        
        # Combine scores
        combined_scores = self.recommender._combine_text_and_categorical_scores(
            text_scores=text_scores,
            categorical_scores=categorical_scores
        )
        
        # Check that the combined scores are calculated correctly
        for doc_id in combined_scores:
            expected_score = (
                self.recommender.text_weight * text_scores[doc_id] +
                self.recommender.categorical_weight * categorical_scores[doc_id]
            )
            self.assertAlmostEqual(combined_scores[doc_id], expected_score)

if __name__ == '__main__':
    unittest.main()
