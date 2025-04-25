#!/usr/bin/env python3
"""
Test the weight optimizer with our improved document retrieval method.
"""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Import dictionary patching solution
from patch_all_dictionaries import safe_merge_dicts, main as apply_dict_patches

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.models.pinecone_recommender import PineconeRecommender
from src.models.personalized_recommender import PersonalizedRecommender
from src.utils.recommender_evaluation import RecommenderEvaluator
from src.utils.weight_optimizer import WeightOptimizer

def main():
    # Apply comprehensive dictionary patching solution
    logger.info("Applying dictionary patches to prevent addition errors")
    apply_dict_patches()
    
    # Load environment variables
    load_dotenv()
    
    # Load Pinecone API key
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "eu-legal-documents-legal-bert")
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
    
    if not pinecone_api_key:
        logger.error("PINECONE_API_KEY environment variable not set")
        sys.exit(1)
        
    # Print API key (masked for security) and environment for debugging
    if pinecone_api_key:
        masked_key = pinecone_api_key[:8] + "*" * (len(pinecone_api_key) - 12) + pinecone_api_key[-4:]
        logger.info(f"Using Pinecone API key: {masked_key}")
        logger.info(f"Using Pinecone environment: {pinecone_environment}")
        logger.info(f"Using Pinecone index: {pinecone_index_name}")
    
    # Initialize personalized recommender directly (it will create its own PineconeRecommender internally)
    print("\nInitializing PersonalizedRecommender...")
    recommender = PersonalizedRecommender(
        api_key=pinecone_api_key,
        index_name=pinecone_index_name,
        embedder_model="nlpaueb/legal-bert-base-uncased"
    )
    
    # Path to test data
    test_data_path = str(Path(__file__).parent.parent / "evaluation" / "renewable_energy_client_test_data.json")
    
    # Initialize evaluator
    print(f"\nInitializing evaluator with test data: {test_data_path}")
    evaluator = RecommenderEvaluator(
        recommender=recommender,
        test_data_path=test_data_path
    )
    
    # Initialize weight optimizer
    print("\nInitializing weight optimizer...")
    optimizer = WeightOptimizer(
        recommender=recommender,
        evaluator=evaluator,
        metric='ndcg@10'
    )
    
    # Define weight parameters to optimize
    weight_params = {
        'expert_profile': [0.2, 0.3, 0.4, 0.5],
        'historical_documents': [0.2, 0.3, 0.4, 0.5],
        'categorical_preferences': [0.2, 0.3, 0.4, 0.5]
    }
    
    # Define k values for evaluation
    k_values = [5, 10]
    
    # Define metrics to compute
    metrics = ['precision', 'recall', 'ndcg', 'map']
    
    print("\n=== TESTING WEIGHT OPTIMIZER WITH IMPROVED DOCUMENT RETRIEVAL ===\n")
    print(f"Running optimization with {len(weight_params['expert_profile'])} x {len(weight_params['historical_documents'])} x {len(weight_params['categorical_preferences'])} = {len(weight_params['expert_profile']) * len(weight_params['historical_documents']) * len(weight_params['categorical_preferences'])} weight combinations")
    print(f"Using k values: {k_values}")
    print(f"Computing metrics: {metrics}")
    
    try:
        # Run optimization
        print("\nStarting optimization...")
        results = optimizer.optimize(
            weight_params=weight_params,
            k_values=k_values,
            metrics=metrics
        )
        
        # Print results
        print("\n=== OPTIMIZATION RESULTS ===")
        print(f"Best metric score ({optimizer.metric}): {results['best_score']}")
        print(f"Best weights: {results['best_weights']}")
        
        # Save results
        output_path = str(Path(__file__).parent.parent / "results" / "optimized_weights.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        optimizer.save_weights(output_path)
        print(f"\nSaved optimized weights to: {output_path}")
        
        print("\n=== WEIGHT OPTIMIZATION SUCCESSFUL! ===")
        print("The improved document retrieval method works with the weight optimizer.")
        
    except Exception as e:
        print(f"\n❌ ERROR: Weight optimization failed: {str(e)}")
        logger.error(f"Weight optimization failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
