#!/usr/bin/env python
"""
Weight Adjustment Example for EU Legal Recommender

This example demonstrates how to dynamically adjust weights in different parts of the
recommender system to fine-tune recommendation quality for different use cases.

The script shows:
1. How to create and load weight configurations
2. How to adjust weights at different levels (similarity, personalization, features)
3. How to compare recommendation results with different weight settings
"""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt

# Path to project root and add to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import recommender components
from src.models.personalized_recommender import PersonalizedRecommender
from src.models.similarity import SimilarityComputer
from src.utils.weight_config import WeightConfig
from src.utils.evaluation import evaluate_recommendations
from src.utils.logging import get_logger
from src.config import PINECONE

# Set up logging
logger = get_logger(__name__)
logger.setLevel(logging.INFO)

def create_weight_configurations() -> Dict[str, str]:
    """
    Create different weight configurations for testing.
    
    Returns:
        Dict[str, str]: Dictionary mapping configuration names to file paths
    """
    # Create output directory
    output_dir = Path("./output/weight_configs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration 1: Text-focused
    text_focused = WeightConfig()
    text_focused.set_weights({
        "text_weight": 0.9,
        "categorical_weight": 0.1
    }, "similarity")
    text_focused.set_weights({
        "profile_weight": 0.4,
        "query_weight": 0.6
    }, "personalization")
    text_focused_path = output_dir / "text_focused.json"
    text_focused.save_config(str(text_focused_path))
    
    # Configuration 2: Categorical-focused
    categorical_focused = WeightConfig()
    categorical_focused.set_weights({
        "text_weight": 0.3,
        "categorical_weight": 0.7
    }, "similarity")
    categorical_focused.set_weights({
        "profile_weight": 0.6,
        "query_weight": 0.4
    }, "personalization")
    categorical_focused_path = output_dir / "categorical_focused.json"
    categorical_focused.save_config(str(categorical_focused_path))
    
    # Configuration 3: Balanced
    balanced = WeightConfig()
    balanced.set_weights({
        "text_weight": 0.5,
        "categorical_weight": 0.5
    }, "similarity")
    balanced.set_weights({
        "profile_weight": 0.5,
        "query_weight": 0.5
    }, "personalization")
    balanced.set_weights({
        "document_type_weight": 0.25,
        "subject_matter_weight": 0.35,
        "date_weight": 0.15,
        "author_weight": 0.25
    }, "features")
    balanced_path = output_dir / "balanced.json"
    balanced.save_config(str(balanced_path))
    
    # Configuration 4: Custom feature weights
    custom_features = WeightConfig()
    custom_features.set_weights({
        "text_weight": 0.6,
        "categorical_weight": 0.4
    }, "similarity")
    custom_features.set_weights({
        "document_type_weight": 0.4,
        "subject_matter_weight": 0.4,
        "date_weight": 0.1,
        "author_weight": 0.1
    }, "features")
    custom_features_path = output_dir / "custom_features.json"
    custom_features.save_config(str(custom_features_path))
    
    logger.info(f"Created 4 weight configurations in {output_dir}")
    
    return {
        "text_focused": str(text_focused_path),
        "categorical_focused": str(categorical_focused_path),
        "balanced": str(balanced_path),
        "custom_features": str(custom_features_path)
    }

def run_recommendations_with_config(api_key: str, config_path: str, query: str, user_id: str = "test_user") -> List[Dict[str, Any]]:
    """
    Run recommendations using a specific weight configuration.
    
    Args:
        api_key: Pinecone API key
        config_path: Path to weight configuration file
        query: Query text to search for
        user_id: User ID for personalization
        
    Returns:
        List of recommendation results
    """
    # Initialize recommender with weight configuration
    recommender = PersonalizedRecommender(
        api_key=api_key,
        weight_config_path=config_path
    )
    
    # Log current weights
    logger.info(f"Using configuration from {config_path}")
    logger.info(f"Text weight: {recommender.text_weight:.2f}, Categorical weight: {recommender.categorical_weight:.2f}")
    logger.info(f"Profile weight: {recommender.profile_weight:.2f}, Query weight: {recommender.query_weight:.2f}")
    
    # Get recommendations
    results = recommender.recommend(
        query=query,
        user_id=user_id,
        top_k=10,
        include_scores=True
    )
    
    return results

def compare_configurations(api_key: str, config_paths: Dict[str, str], query: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compare recommendation results with different weight configurations.
    
    Args:
        api_key: Pinecone API key
        config_paths: Dictionary mapping configuration names to file paths
        query: Query text to search for
        
    Returns:
        Dictionary mapping configuration names to recommendation results
    """
    results = {}
    
    for config_name, config_path in config_paths.items():
        logger.info(f"\n--- Testing configuration: {config_name} ---")
        config_results = run_recommendations_with_config(api_key, config_path, query)
        results[config_name] = config_results
        
        # Print top 3 results
        logger.info(f"Top 3 results for {config_name}:")
        for i, result in enumerate(config_results[:3]):
            logger.info(f"  {i+1}. {result['title']} (Score: {result['score']:.4f})")
            
    return results

def dynamic_weight_adjustment_example(api_key: str, query: str):
    """
    Demonstrate dynamic weight adjustment during runtime.
    
    Args:
        api_key: Pinecone API key
        query: Query text to search for
    """
    
    logger.info("\n=== Dynamic Weight Adjustment Example ===")
    
    # Initialize recommender with default weights
    recommender = PersonalizedRecommender(api_key=api_key)
    
    # Get recommendations with default weights
    logger.info("\n--- Default Weights ---")
    logger.info(f"Text weight: {recommender.text_weight:.2f}, Categorical weight: {recommender.categorical_weight:.2f}")
    default_results = recommender.recommend(query=query, top_k=5, include_scores=True)
    
    # Print results
    logger.info("Top 5 results with default weights:")
    for i, result in enumerate(default_results):
        logger.info(f"  {i+1}. {result['title']} (Score: {result['score']:.4f})")
    
    # Adjust weights to focus more on text similarity
    logger.info("\n--- Adjusting Weights to Focus on Text ---")
    recommender.set_similarity_weights(text_weight=0.9, categorical_weight=0.1)
    
    # Get recommendations with adjusted weights
    text_focused_results = recommender.recommend(query=query, top_k=5, include_scores=True)
    
    # Print results
    logger.info("Top 5 results with text-focused weights:")
    for i, result in enumerate(text_focused_results):
        logger.info(f"  {i+1}. {result['title']} (Score: {result['score']:.4f})")
    
    # Adjust weights to focus more on categorical features
    logger.info("\n--- Adjusting Weights to Focus on Categorical Features ---")
    recommender.set_similarity_weights(text_weight=0.3, categorical_weight=0.7)
    
    # Get recommendations with adjusted weights
    categorical_focused_results = recommender.recommend(query=query, top_k=5, include_scores=True)
    
    # Print results
    logger.info("Top 5 results with categorical-focused weights:")
    for i, result in enumerate(categorical_focused_results):
        logger.info(f"  {i+1}. {result['title']} (Score: {result['score']:.4f})")
    
    # Compare results
    compare_result_overlap(default_results, text_focused_results, categorical_focused_results)

def compare_result_overlap(default_results, text_focused_results, categorical_focused_results):
    """
    Compare the overlap between different result sets.
    
    Args:
        default_results: Results with default weights
        text_focused_results: Results with text-focused weights
        categorical_focused_results: Results with categorical-focused weights
    """
    # Extract document IDs
    default_ids = [r['id'] for r in default_results]
    text_ids = [r['id'] for r in text_focused_results]
    cat_ids = [r['id'] for r in categorical_focused_results]
    
    # Calculate overlap
    text_overlap = len(set(default_ids) & set(text_ids))
    cat_overlap = len(set(default_ids) & set(cat_ids))
    text_cat_overlap = len(set(text_ids) & set(cat_ids))
    
    logger.info("\n--- Result Overlap Analysis ---")
    logger.info(f"Default vs Text-focused: {text_overlap} documents in common")
    logger.info(f"Default vs Categorical-focused: {cat_overlap} documents in common")
    logger.info(f"Text-focused vs Categorical-focused: {text_cat_overlap} documents in common")
    
    # Calculate Jaccard similarity
    jaccard_default_text = len(set(default_ids) & set(text_ids)) / len(set(default_ids) | set(text_ids))
    jaccard_default_cat = len(set(default_ids) & set(cat_ids)) / len(set(default_ids) | set(cat_ids))
    jaccard_text_cat = len(set(text_ids) & set(cat_ids)) / len(set(text_ids) | set(cat_ids))
    
    logger.info(f"Jaccard similarity (Default vs Text-focused): {jaccard_default_text:.2f}")
    logger.info(f"Jaccard similarity (Default vs Categorical-focused): {jaccard_default_cat:.2f}")
    logger.info(f"Jaccard similarity (Text-focused vs Categorical-focused): {jaccard_text_cat:.2f}")

def document_ranker_weight_adjustment_example(api_key: str, query: str):
    """
    Demonstrate the enhanced weight adjustment capabilities in DocumentRanker.
    
    This example shows how to:
    1. Initialize a DocumentRanker with weight change tracking
    2. Adjust similarity weights and observe cache invalidation
    3. Set feature-level weights for individual queries
    4. Get detailed similarity breakdowns in results
    
    Args:
        api_key: Pinecone API key
        query: Query text to search for
    """
    logger.info("\n=== DocumentRanker Enhanced Weight Adjustment Example ===")
    
    # Import necessary components for direct DocumentRanker usage
    from recommender.src.models.embeddings import BERTEmbedder
    from recommender.src.models.features import FeatureProcessor
    from recommender.src.models.ranker import DocumentRanker
    from recommender.src.models.similarity import SimilarityComputer
    
    logger.info("\n--- Initializing DocumentRanker with weight change tracking ---")
    
    # Create components
    embedder = BERTEmbedder()
    feature_processor = FeatureProcessor()
    similarity_computer = SimilarityComputer(text_weight=0.6, categorical_weight=0.4)
    
    # Initialize DocumentRanker with weight change tracking enabled
    ranker = DocumentRanker(
        embedder=embedder,
        feature_processor=feature_processor,
        similarity_computer=similarity_computer,
        invalidate_cache_on_weight_change=True  # This will clear cache when weights change
    )
    
    # Process some sample documents
    sample_docs = [
        {
            "id": "doc1",
            "summary": "Regulation on environmental protection measures in agriculture",
            "keywords": ["environment", "agriculture", "regulation"],
            "features": {
                "document_type": "regulation",
                "subject_matter": "environment",
                "date": "2022",
                "author": "Commission"
            }
        },
        {
            "id": "doc2",
            "summary": "Directive on renewable energy sources and sustainability criteria",
            "keywords": ["renewable", "energy", "sustainability"],
            "features": {
                "document_type": "directive",
                "subject_matter": "energy",
                "date": "2021",
                "author": "Parliament"
            }
        },
        {
            "id": "doc3",
            "summary": "Decision on funding for agricultural development programs",
            "keywords": ["funding", "agriculture", "development"],
            "features": {
                "document_type": "decision",
                "subject_matter": "agriculture",
                "date": "2022",
                "author": "Council"
            }
        }
    ]
    
    # Process documents
    ranker.process_documents(sample_docs)
    
    # Create a sample query profile
    query_profile = {
        "interests": "Environmental protection in agriculture",
        "keywords": ["environment", "agriculture", "protection"],
        "features": {
            "document_type": "regulation",
            "subject_matter": "environment",
            "date": "2022",
            "author": "Commission"
        }
    }
    
    # Get initial weights
    initial_weights = ranker.get_weights()
    logger.info(f"Initial weights configuration: {json.dumps(initial_weights, indent=2)}")
    
    # Get recommendations with default weights
    logger.info("\n--- Getting recommendations with default weights ---")
    default_results = ranker.rank_documents(query_profile, top_k=3)
    
    # Print results
    logger.info("Results with default weights:")
    for i, result in enumerate(default_results):
        logger.info(f"  {i+1}. Document {result['id']} (Score: {result['similarity']:.4f})")
        if result['feature_details']:
            logger.info(f"     Feature details: {result['feature_details']}")
    
    # Adjust weights to focus more on text similarity
    logger.info("\n--- Adjusting weights to focus on text similarity ---")
    ranker.set_similarity_weights(text_weight=0.9, categorical_weight=0.1)
    
    # Get updated weights
    updated_weights = ranker.get_weights()
    logger.info(f"Updated weights: {json.dumps(updated_weights['similarity'], indent=2)}")
    
    # Get recommendations with new weights
    text_focused_results = ranker.rank_documents(query_profile, top_k=3)
    
    # Print results
    logger.info("Results with text-focused weights:")
    for i, result in enumerate(text_focused_results):
        logger.info(f"  {i+1}. Document {result['id']} (Score: {result['similarity']:.4f})")
    
    # Demonstrate per-query feature weighting without changing global weights
    logger.info("\n--- Using per-query feature weights without changing global settings ---")
    
    # Define feature weights just for this query
    query_feature_weights = {
        "document_type": 0.6,  # Prioritize document type
        "subject_matter": 0.3,
        "date": 0.05,
        "author": 0.05
    }
    
    logger.info(f"Per-query feature weights: {query_feature_weights}")
    
    # Get recommendations with per-query feature weights
    feature_weighted_results = ranker.rank_documents(
        query_profile, 
        top_k=3,
        feature_weights=query_feature_weights  # Apply weights just for this query
    )
    
    # Print results
    logger.info("Results with per-query feature weights:")
    for i, result in enumerate(feature_weighted_results):
        logger.info(f"  {i+1}. Document {result['id']} (Score: {result['similarity']:.4f})")
    
    # Verify that global weights haven't changed
    current_weights = ranker.get_weights()
    logger.info(f"\nGlobal weights after per-query weighting: {json.dumps(current_weights['similarity'], indent=2)}")
    
    # Compare results across different weight configurations
    logger.info("\n--- Result comparison across weight configurations ---")
    
    # Extract document IDs from each result set
    default_ids = [r['id'] for r in default_results]
    text_focused_ids = [r['id'] for r in text_focused_results]
    feature_weighted_ids = [r['id'] for r in feature_weighted_results]
    
    logger.info(f"Default weights results: {default_ids}")
    logger.info(f"Text-focused results: {text_focused_ids}")
    logger.info(f"Feature-weighted results: {feature_weighted_ids}")

def main():
    """Main function to run the weight adjustment examples."""
    # Check for Pinecone API key
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        logger.error("PINECONE_API_KEY environment variable not set")
        logger.info("Please set your Pinecone API key using:")
        logger.info("export PINECONE_API_KEY=your-api-key")
        return
    
    # Example query
    query = "Data protection regulations in the European Union"
    
    # Create weight configurations
    config_paths = create_weight_configurations()
    
    # Compare different configurations
    compare_configurations(api_key, config_paths, query)
    
    # Demonstrate dynamic weight adjustment
    dynamic_weight_adjustment_example(api_key, query)
    
    # Demonstrate enhanced DocumentRanker weight adjustment capabilities
    document_ranker_weight_adjustment_example(api_key, query)

if __name__ == "__main__":
    main()
