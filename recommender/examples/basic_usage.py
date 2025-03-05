"""
Example script demonstrating basic usage of the recommender system.
"""
import sys
from pathlib import Path
import logging

# Add parent directory to path to import recommender
sys.path.append(str(Path(__file__).parents[1]))

from src.models.embeddings import BERTEmbedder
from src.models.features import FeatureProcessor
from src.models.similarity import SimilarityComputer
from src.models.ranker import DocumentRanker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Sample documents
    documents = [
        {
            'id': '32024L0123',
            'summary': 'This directive establishes a framework for reducing greenhouse gas emissions and promoting renewable energy sources.',
            'keywords': ['climate change', 'renewable energy', 'emissions', 'environment'],
            'features': {
                'type': 'directive',
                'subject': 'environment',
                'scope': 'EU-wide',
                'legal_basis': 'TFEU_192'
            }
        },
        {
            'id': '32024R0456',
            'summary': 'This regulation sets standards for vehicle emissions and establishes testing procedures.',
            'keywords': ['vehicles', 'emissions', 'standards', 'testing'],
            'features': {
                'type': 'regulation',
                'subject': 'transport',
                'scope': 'EU-wide',
                'legal_basis': 'TFEU_114'
            }
        },
        {
            'id': '32024D0789',
            'summary': 'Decision on financial support for renewable energy projects in member states.',
            'keywords': ['renewable energy', 'funding', 'projects', 'financial support'],
            'features': {
                'type': 'decision',
                'subject': 'energy',
                'scope': 'member_states',
                'legal_basis': 'TFEU_194'
            }
        }
    ]

    # Initialize components
    logger.info("Initializing recommender components...")
    
    embedder = BERTEmbedder(
        model_name='all-MiniLM-L6-v2',
        cache_dir='./cache/embeddings'
    )
    
    feature_processor = FeatureProcessor({
        'type': ['regulation', 'directive', 'decision'],
        'subject': ['environment', 'transport', 'energy', 'finance'],
        'scope': ['EU-wide', 'member_states', 'third_countries'],
        'legal_basis': ['TFEU_114', 'TFEU_192', 'TFEU_194']
    })
    
    similarity_computer = SimilarityComputer(
        text_weight=0.7,
        categorical_weight=0.3,
        use_faiss=True
    )
    
    ranker = DocumentRanker(
        embedder=embedder,
        feature_processor=feature_processor,
        similarity_computer=similarity_computer,
        cache_dir='./cache'
    )

    # Process documents
    logger.info("Processing documents...")
    ranker.process_documents(documents)

    # Example query profiles
    query_profiles = [
        {
            'name': 'Environmental Focus',
            'interests': 'Climate change and environmental protection measures',
            'keywords': ['climate', 'environment', 'protection', 'emissions'],
            'features': {
                'type': 'any',
                'subject': 'environment',
                'scope': 'EU-wide',
                'legal_basis': 'TFEU_192'
            }
        },
        {
            'name': 'Energy Projects',
            'interests': 'Renewable energy project funding and implementation',
            'keywords': ['renewable', 'energy', 'funding', 'projects'],
            'features': {
                'type': 'decision',
                'subject': 'energy',
                'scope': 'member_states',
                'legal_basis': 'TFEU_194'
            }
        }
    ]

    # Get recommendations for each profile
    for profile in query_profiles:
        logger.info(f"\nGetting recommendations for profile: {profile['name']}")
        recommendations = ranker.rank_documents(
            query_profile=profile,
            top_k=2,
            min_similarity=0.3
        )
        
        logger.info("Top recommendations:")
        for doc_id, similarity in recommendations:
            logger.info(f"Document {doc_id}: {similarity:.3f} similarity score")

if __name__ == "__main__":
    main()
