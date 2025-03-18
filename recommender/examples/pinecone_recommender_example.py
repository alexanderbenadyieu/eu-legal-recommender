"""
Example script demonstrating usage of the Pinecone-based recommender system.
"""
import sys
import os
from pathlib import Path
import logging
import argparse
from pprint import pprint

# Add parent directory to path to import recommender
sys.path.append(str(Path(__file__).parents[1]))

from src.models.pinecone_recommender import PineconeRecommender
from src.models.features import FeatureProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Legal document recommendation example")
    
    parser.add_argument("--pinecone-api-key", type=str, 
                        default=os.environ.get("PINECONE_API_KEY"),
                        help="Pinecone API key (or set PINECONE_API_KEY env var)")
    
    parser.add_argument("--index-name", type=str, 
                        default="eu-legal-documents-legal-bert",
                        help="Pinecone index name")
    
    parser.add_argument("--query", type=str,
                        default="Climate change and environmental protection measures",
                        help="Query text for recommendations")
    
    parser.add_argument("--document-id", type=str,
                        help="Document ID to find similar documents (alternative to query)")
    
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of recommendations to return")
    
    parser.add_argument("--filter-type", type=str, choices=["regulation", "directive", "decision"],
                        help="Filter results by document type")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Ensure API key is provided
    if not args.pinecone_api_key:
        raise ValueError("Pinecone API key must be provided via --pinecone-api-key or PINECONE_API_KEY env var")
    
    # Initialize feature processor with example feature schema
    # This should match the schema used when generating the embeddings
    feature_processor = FeatureProcessor({
        'type': ['regulation', 'directive', 'decision', 'other'],
        'subject': ['environment', 'transport', 'energy', 'finance', 'other'],
        'scope': ['EU-wide', 'member_states', 'third_countries', 'other'],
        'legal_basis': ['TFEU_114', 'TFEU_192', 'TFEU_194', 'other']
    })
    
    # Initialize recommender
    logger.info("Initializing Pinecone recommender...")
    recommender = PineconeRecommender(
        api_key=args.pinecone_api_key,
        index_name=args.index_name,
        embedder_model="nlpaueb/legal-bert-small-uncased",
        feature_processor=feature_processor,
        text_weight=0.8,
        categorical_weight=0.2
    )
    
    # Prepare filter if needed
    filter_dict = None
    if args.filter_type:
        filter_dict = {"metadata": {"type": args.filter_type}}
    
    # Example query features
    query_features = {
        'type': 'directive' if args.filter_type == 'directive' else 'regulation',
        'subject': 'environment',  # Example subject for environmental query
        'scope': 'EU-wide',
        'legal_basis': 'TFEU_192'  # Environment legal basis
    }
    
    # Get recommendations
    if args.document_id:
        logger.info(f"Finding documents similar to ID: {args.document_id}")
        recommendations = recommender.get_recommendations_by_id(
            document_id=args.document_id,
            top_k=args.top_k,
            filter=filter_dict
        )
    else:
        logger.info(f"Getting recommendations for query: '{args.query}'")
        logger.info(f"Using categorical features: {query_features}")
        recommendations = recommender.get_recommendations(
            query_text=args.query,
            query_keywords=["climate", "emissions", "environment", "regulation"],
            query_features=query_features,
            top_k=args.top_k,
            filter=filter_dict
        )
    
    # Display results
    logger.info(f"Top {len(recommendations)} recommendations:")
    for i, rec in enumerate(recommendations):
        print(f"\n{i+1}. Document ID: {rec['id']} (Score: {rec['score']:.4f})")
        
        # Show component scores if available
        if rec.get('text_score') is not None:
            print(f"   Text Similarity: {rec['text_score']:.4f}")
        if rec.get('categorical_score') is not None:
            print(f"   Categorical Similarity: {rec['categorical_score']:.4f}")
            
        if rec.get('metadata'):
            print("   Metadata:")
            for key, value in rec['metadata'].items():
                if key != 'categorical_features':  # Skip showing raw categorical features
                    print(f"   - {key}: {value}")

if __name__ == "__main__":
    main()
