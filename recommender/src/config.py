"""
Configuration module for the EU Legal Document Recommender System.

This module centralizes all configuration settings for the recommender system,
including model parameters, similarity weights, and default values.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
PROFILES_DIR = PROJECT_ROOT / "profiles"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
PROFILES_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# API Keys
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'gcp-starter')

# Model Configuration
EMBEDDER = {
    'model_name': 'nlpaueb/legal-bert-base-uncased',
    'dimension': 768,
    'device': os.getenv('DEVICE', 'cpu'),  # 'cpu' or 'cuda'
    'batch_size': 32
}

# Pinecone Configuration
PINECONE = {
    'index_name': 'eu-legal-documents-legal-bert',
    'metric': 'cosine',
    'dimension': EMBEDDER['dimension']
}

# Similarity Weights
SIMILARITY = {
    'text_weight': 0.7,
    'categorical_weight': 0.3
}

# User Profile Configuration
USER_PROFILE = {
    'expert_weight': 0.4,
    'historical_weight': 0.4,
    'categorical_weight': 0.2,
    'profile_weight': 0.7,
    'query_weight': 0.3
}

# Feature Configuration
FEATURES = {
    'weights': {
        'document_type': 0.4,
        'subject_matters': 0.3,
        'author': 0.2,
        'form': 0.1
    },
    'multi_valued_features': ['subject_matters']
}

# Database Configuration
DATABASE = {
    'db_type': os.getenv('DB_TYPE', 'consolidated')
}

# Logging Configuration
LOGGING = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': LOGS_DIR / 'recommender.log'
}
