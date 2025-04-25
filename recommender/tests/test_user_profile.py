"""
Tests for the UserProfile class.
"""
import unittest
import numpy as np
from pathlib import Path
import sys
import os
import json
import tempfile
import shutil

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.user_profile import UserProfile
from src.models.embeddings import BERTEmbedder
from src.config import EMBEDDER

class TestUserProfile(unittest.TestCase):
    """Test cases for the UserProfile class."""
    
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
        # Initialize the user profile
        self.user_profile = UserProfile(
            user_id="test_user",
            embedder=self.embedder
        )
        
        # Sample expert profile description
        self.expert_description = """
        The user is interested in EU regulations related to renewable energy, 
        particularly wind and solar power. They focus on policy frameworks, 
        subsidies, and integration of renewable sources into the energy grid.
        """
        
        # Sample historical documents
        self.historical_docs = [
            "32018L2001",  # Renewable Energy Directive
            "32019R0943",  # Electricity Market Regulation
            "32012L0027"   # Energy Efficiency Directive
        ]
        
        # Sample categorical preferences
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
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test that the user profile initializes correctly."""
        # Check that the user ID is set correctly
        self.assertEqual(self.user_profile.user_id, "test_user")
        # Check that the embedder is set correctly
        self.assertEqual(self.user_profile.embedder, self.embedder)
        # Check that the profile is initially empty
        self.assertIsNone(self.user_profile.expert_profile)
        self.assertEqual(self.user_profile.historical_documents, [])
        self.assertEqual(self.user_profile.categorical_preferences, {})
    
    def test_create_expert_profile(self):
        """Test creating an expert profile."""
        # Create the expert profile
        embedding = self.user_profile.create_expert_profile(self.expert_description)
        
        # Check that the expert profile is set correctly
        self.assertIsNotNone(self.user_profile.expert_profile)
        self.assertEqual(self.user_profile.expert_profile.shape, (self.actual_embedding_dim,))
        self.assertEqual(self.user_profile.expert_profile_text, self.expert_description)
        
        # Check that the returned embedding is the same as the stored one
        np.testing.assert_array_equal(embedding, self.user_profile.expert_profile)
    
    def test_add_historical_document(self):
        """Test adding historical documents."""
        # Add historical documents
        for doc_id in self.historical_docs:
            # Create a random embedding for the document
            doc_embedding = np.random.rand(self.actual_embedding_dim)
            doc_embedding = doc_embedding / np.linalg.norm(doc_embedding)  # Normalize
            self.user_profile.add_historical_document(doc_id, doc_embedding)
        
        # Check that the historical documents are set correctly
        self.assertEqual(len(self.user_profile.historical_documents), len(self.historical_docs))
        for doc_id in self.historical_docs:
            self.assertIn(doc_id, self.user_profile.historical_documents)
    
    def test_set_categorical_preferences(self):
        """Test setting categorical preferences."""
        # Set categorical preferences
        self.user_profile.set_categorical_preferences(self.categorical_prefs)
        
        # Check that the categorical preferences are set correctly
        self.assertEqual(self.user_profile.categorical_preferences, self.categorical_prefs)
        
        # Check specific preferences
        self.assertEqual(
            self.user_profile.categorical_preferences["document_type"]["regulation"],
            0.4
        )
        self.assertEqual(
            self.user_profile.categorical_preferences["subject_matters"]["energy"],
            0.5
        )
    
    def test_to_dict(self):
        """Test converting the user profile to a dictionary."""
        # Set up a complete profile
        self.user_profile.create_expert_profile(self.expert_description)
        for doc_id in self.historical_docs:
            # Create a random embedding for the document
            doc_embedding = np.random.rand(self.actual_embedding_dim)
            doc_embedding = doc_embedding / np.linalg.norm(doc_embedding)  # Normalize
            self.user_profile.add_historical_document(doc_id, doc_embedding)
        self.user_profile.set_categorical_preferences(self.categorical_prefs)
        
        # Convert to dictionary
        profile_dict = self.user_profile.to_dict()
        
        # Check that the dictionary has the correct structure
        self.assertEqual(profile_dict["user_id"], "test_user")
        self.assertEqual(profile_dict["expert_profile_text"], self.expert_description)
        self.assertEqual(profile_dict["historical_documents"], self.historical_docs)
        self.assertEqual(profile_dict["categorical_preferences"], self.categorical_prefs)
    
    def test_from_dict(self):
        """Test creating a user profile from a dictionary."""
        # Set up a complete profile
        self.user_profile.create_expert_profile(self.expert_description)
        for doc_id in self.historical_docs:
            # Create a random embedding for the document
            doc_embedding = np.random.rand(self.actual_embedding_dim)
            doc_embedding = doc_embedding / np.linalg.norm(doc_embedding)  # Normalize
            self.user_profile.add_historical_document(doc_id, doc_embedding)
        self.user_profile.set_categorical_preferences(self.categorical_prefs)
        
        # Convert to dictionary
        profile_dict = self.user_profile.to_dict()
        
        # Create a new profile from the dictionary
        new_profile = UserProfile.from_dict(profile_dict, self.embedder)
        
        # Check that the new profile has the correct values
        self.assertEqual(new_profile.user_id, "test_user")
        self.assertEqual(new_profile.expert_profile_text, self.expert_description)
        self.assertEqual(new_profile.historical_documents, self.historical_docs)
        self.assertEqual(new_profile.categorical_preferences, self.categorical_prefs)
    
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
        
        # Create a user profile from the data
        user_id = profile_data['user_id']
        profile = UserProfile(user_id=user_id, embedder=self.embedder)
        
        # Extract profile components
        profile_components = profile_data['profile']
        
        # Create expert profile if available
        if 'expert_profile' in profile_components and 'description' in profile_components['expert_profile']:
            expert_description = profile_components['expert_profile']['description']
            profile.create_expert_profile(expert_description)
        
        # Add historical documents if available
        if 'historical_documents' in profile_components:
            historical_docs = profile_components['historical_documents']
            for doc_id in historical_docs:
                profile.add_historical_document(doc_id)
        
        # Set categorical preferences if available
        if 'categorical_preferences' in profile_components:
            categorical_prefs = profile_components['categorical_preferences']
            profile.set_categorical_preferences(categorical_prefs)
        
        # Check that the profile is loaded correctly
        self.assertEqual(profile.user_id, user_id)
        if 'expert_profile' in profile_components and 'description' in profile_components['expert_profile']:
            self.assertEqual(profile.expert_profile_text, expert_description)
        if 'historical_documents' in profile_components:
            self.assertEqual(len(profile.historical_documents), len(historical_docs))
        if 'categorical_preferences' in profile_components:
            self.assertEqual(profile.categorical_preferences, categorical_prefs)

if __name__ == '__main__':
    unittest.main()
