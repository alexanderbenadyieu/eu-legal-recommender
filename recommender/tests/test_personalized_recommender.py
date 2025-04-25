"""
Tests for the PersonalizedRecommender class.
"""
import unittest
import numpy as np
from pathlib import Path
import sys
import os
import json
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.personalized_recommender import PersonalizedRecommender
from src.models.user_profile import UserProfile
from src.models.features import FeatureProcessor
from src.config import USER_PROFILE

class TestPersonalizedRecommender(unittest.TestCase):
    """Test cases for the PersonalizedRecommender class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the Pinecone client and index
        self.pinecone_mock = MagicMock()
        self.index_mock = MagicMock()
        
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
        
        # Initialize the recommender with the mock Pinecone client
        with patch('pinecone.Pinecone', return_value=self.pinecone_mock):
            self.recommender = PersonalizedRecommender(
                api_key='fake_api_key',
                index_name='test-index',
                embedder_model='sentence-transformers/all-MiniLM-L6-v2',  # Smaller model for testing
                feature_processor=self.feature_processor,
                profile_weight=USER_PROFILE['profile_weight'],
                query_weight=USER_PROFILE['query_weight'],
                expert_weight=USER_PROFILE['expert_weight'],
                historical_weight=USER_PROFILE['historical_weight'],
                categorical_weight=USER_PROFILE['categorical_weight']
            )
            
            # Replace the real embedder with a mock
            self.recommender.embedder = MagicMock()
            # Configure the mock embedder to return a random vector
            self.recommender.embedder.generate_embeddings.return_value = np.random.rand(768)
            
            # Create a test user profile
            self.user_id = "test_user"
            self.expert_description = """
            The user is interested in EU regulations related to renewable energy, 
            particularly wind and solar power. They focus on policy frameworks, 
            subsidies, and integration of renewable sources into the energy grid.
            """
            self.historical_docs = [
                "32018L2001",  # Renewable Energy Directive
                "32019R0943",  # Electricity Market Regulation
                "32012L0027"   # Energy Efficiency Directive
            ]
            self.categorical_prefs = {
                "document_type": {
                    "regulation": 0.4,
                    "directive": 0.6
                },
                "subject_matters": {
                    "energy": 0.5,
                    "environment": 0.3,
                    "climate change": 0.2
                },
                "author": {
                    "European Commission": 0.7,
                    "European Parliament": 0.3
                }
            }
    
    def test_initialization(self):
        """Test that the personalized recommender initializes correctly."""
        # Check that the index name is set correctly
        self.assertEqual(self.recommender.index_name, 'test-index')
        # Check that the weights are set correctly
        self.assertEqual(self.recommender.profile_weight, USER_PROFILE['profile_weight'])
        self.assertEqual(self.recommender.query_weight, USER_PROFILE['query_weight'])
        self.assertEqual(self.recommender.expert_weight, USER_PROFILE['expert_weight'])
        self.assertEqual(self.recommender.historical_weight, USER_PROFILE['historical_weight'])
        self.assertEqual(self.recommender.categorical_weight, USER_PROFILE['categorical_weight'])
    
    def test_create_expert_profile(self):
        """Test creating an expert profile."""
        # Mock the user profile
        user_profile = MagicMock()
        user_profile.create_expert_profile.return_value = np.random.rand(768)
        
        # Mock the get_user_profile method to return the mock user profile
        self.recommender.get_user_profile = MagicMock(return_value=user_profile)
        
        # Create the expert profile
        self.recommender.create_expert_profile(self.user_id, self.expert_description)
        
        # Check that the user profile's create_expert_profile method was called
        user_profile.create_expert_profile.assert_called_once_with(self.expert_description)
    
    def test_add_historical_document(self):
        """Test adding a historical document."""
        # Mock the user profile
        user_profile = MagicMock()
        
        # Mock the get_user_profile method to return the mock user profile
        self.recommender.get_user_profile = MagicMock(return_value=user_profile)
        
        # Add a historical document
        self.recommender.add_historical_document(self.user_id, self.historical_docs[0])
        
        # Check that the user profile's add_historical_document method was called
        user_profile.add_historical_document.assert_called_once_with(self.historical_docs[0])
    
    def test_set_categorical_preferences(self):
        """Test setting categorical preferences."""
        # Mock the user profile
        user_profile = MagicMock()
        
        # Mock the get_user_profile method to return the mock user profile
        self.recommender.get_user_profile = MagicMock(return_value=user_profile)
        
        # Set categorical preferences
        self.recommender.set_categorical_preferences(self.user_id, self.categorical_prefs)
        
        # Check that the user profile's set_categorical_preferences method was called
        user_profile.set_categorical_preferences.assert_called_once_with(self.categorical_prefs)
    
    def test_get_personalized_recommendations(self):
        """Test getting personalized recommendations."""
        # Mock the user profile
        user_profile = MagicMock()
        user_profile.expert_profile = np.random.rand(768)
        user_profile.historical_documents = self.historical_docs
        user_profile.categorical_preferences = self.categorical_prefs
        
        # Mock the get_user_profile method to return the mock user profile
        self.recommender.get_user_profile = MagicMock(return_value=user_profile)
        
        # Mock the get_recommendations method
        self.recommender.get_recommendations = MagicMock(return_value=self.sample_results['matches'])
        
        # Get personalized recommendations
        recommendations = self.recommender.get_personalized_recommendations(
            user_id=self.user_id,
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
    
    def test_load_renewable_energy_profile(self):
        """Test loading the renewable energy client profile."""
        # Path to the renewable energy client profile
        profile_path = Path(project_root) / "profiles" / "renewable_energy_client.json"
        
        # Skip the test if the profile doesn't exist
        if not profile_path.exists():
            self.skipTest("Renewable energy client profile not found")
        
        # Load the profile
        with open(profile_path, 'r') as f:
            profile_data = json.load(f)
        
        # Extract profile components
        user_id = profile_data['user_id']
        profile = profile_data['profile']
        
        # Mock the user profile
        user_profile = MagicMock()
        
        # Mock the get_user_profile method to return the mock user profile
        self.recommender.get_user_profile = MagicMock(return_value=user_profile)
        
        # Create expert profile if available
        if 'expert_profile' in profile and 'description' in profile['expert_profile']:
            expert_description = profile['expert_profile']['description']
            self.recommender.create_expert_profile(user_id, expert_description)
            user_profile.create_expert_profile.assert_called_once_with(expert_description)
        
        # Add historical documents if available
        if 'historical_documents' in profile:
            historical_docs = profile['historical_documents']
            for doc_id in historical_docs:
                self.recommender.add_historical_document(user_id, doc_id)
                user_profile.add_historical_document.assert_any_call(doc_id)
        
        # Set categorical preferences if available
        if 'categorical_preferences' in profile:
            categorical_prefs = profile['categorical_preferences']
            self.recommender.set_categorical_preferences(user_id, categorical_prefs)
            user_profile.set_categorical_preferences.assert_called_once_with(categorical_prefs)

if __name__ == '__main__':
    unittest.main()
