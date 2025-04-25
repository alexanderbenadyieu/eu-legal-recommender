#!/usr/bin/env python
"""
Example script demonstrating weight optimization and evaluation for the EU Legal Recommender.

This script shows how to:
1. Create a test dataset with ground truth relevance judgments
2. Evaluate recommender performance using standard IR metrics
3. Optimize weights for similarity computation
4. Visualize evaluation results
"""

import os
import json
import argparse
from pathlib import Path

from src.models.personalized_recommender import PersonalizedRecommender
from src.utils.recommender_evaluation import RecommenderEvaluator
from src.utils.weight_optimizer import WeightOptimizer
from src.utils.logging import setup_logger

# Set up logger
logger = setup_logger('weight_optimization_example')

def main():
    """Run the weight optimization example."""
    parser = argparse.ArgumentParser(description='Weight Optimization Example for EU Legal Recommender')
    
    # Recommender configuration
    parser.add_argument('--api-key', type=str, help='Pinecone API key (or use PINECONE_API_KEY env var)')
    parser.add_argument('--index-name', type=str, default='eu-legal-docs', help='Pinecone index name')
    parser.add_argument('--model', type=str, default='legal-bert', help='Embedding model name')
    
    # Test data options
    parser.add_argument('--create-test-data', action='store_true', help='Create new test dataset')
    parser.add_argument('--test-data-path', type=str, default='data/test_data.json', 
                      help='Path to test data file')
    parser.add_argument('--num-queries', type=int, default=10, 
                      help='Number of queries for test dataset')
    
    # Weight optimization options
    parser.add_argument('--skip-optimization', action='store_true', 
                      help='Skip weight optimization')
    parser.add_argument('--optimization-metric', type=str, default='ndcg@10',
                      help='Metric to optimize for')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='optimization_results',
                      help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get API key from args or environment
    api_key = args.api_key or os.getenv('PINECONE_API_KEY')
    if not api_key:
        raise ValueError('Pinecone API key must be provided via --api-key or PINECONE_API_KEY env var')
    
    # Initialize recommender
    logger.info(f'Initializing recommender with index {args.index_name} and model {args.model}')
    recommender = PersonalizedRecommender(
        api_key=api_key,
        index_name=args.index_name,
        embedder_model=args.model
    )
    
    # Create test data if requested
    if args.create_test_data:
        logger.info(f'Creating test dataset with {args.num_queries} queries')
        
        # Sample EU legal queries
        eu_legal_queries = [
            {'id': 'q1', 'text': 'GDPR compliance requirements for businesses'},
            {'id': 'q2', 'text': 'EU environmental protection directives'},
            {'id': 'q3', 'text': 'Consumer rights in digital markets'},
            {'id': 'q4', 'text': 'EU competition law and antitrust regulations'},
            {'id': 'q5', 'text': 'Renewable energy targets and policies'},
            {'id': 'q6', 'text': 'EU banking regulations after financial crisis'},
            {'id': 'q7', 'text': 'Cross-border taxation in the European Union'},
            {'id': 'q8', 'text': 'EU labor law and worker protection'},
            {'id': 'q9', 'text': 'Digital services act implementation'},
            {'id': 'q10', 'text': 'Agricultural subsidies and rural development'}
        ]
        
        # Use only the requested number of queries
        queries = eu_legal_queries[:args.num_queries]
        
        # Create optimizer for test dataset creation
        optimizer = WeightOptimizer(recommender=recommender)
        
        # Create test dataset
        optimizer.create_test_dataset(
            output_path=args.test_data_path,
            queries=queries,
            num_relevant_per_query=5
        )
    
    # Ensure test data exists
    if not os.path.exists(args.test_data_path):
        raise FileNotFoundError(f'Test data file not found at {args.test_data_path}. '
                              f'Use --create-test-data to create it.')
    
    # Create evaluator
    evaluator = RecommenderEvaluator(recommender, args.test_data_path)
    
    # Run initial evaluation
    logger.info('Running initial evaluation')
    initial_results = evaluator.evaluate()
    
    # Save initial results
    with open(f'{output_dir}/initial_evaluation.json', 'w') as f:
        json.dump(initial_results, f, indent=2)
    
    # Print initial results summary
    print('\nInitial Evaluation Results:')
    for metric, value in initial_results['average'].items():
        print(f'  {metric}: {value:.4f}')
    
    # Run weight optimization if not skipped
    if not args.skip_optimization:
        logger.info(f'Running weight optimization for {args.optimization_metric}')
        
        # Define weight parameters to try
        weight_params = {
            'text_weight': [0.5, 0.6, 0.7, 0.8, 0.9],
            'categorical_weight': [0.5, 0.4, 0.3, 0.2, 0.1]
        }
        
        # Create optimizer
        optimizer = WeightOptimizer(
            recommender=recommender,
            evaluator=evaluator,
            metric=args.optimization_metric,
            config_path=f'{output_dir}/optimal_weights.json'
        )
        
        # Run optimization
        optimization_results = optimizer.optimize(weight_params=weight_params)
        
        # Save optimization results
        with open(f'{output_dir}/optimization_results.json', 'w') as f:
            json.dump(optimization_results, f, indent=2)
        
        # Apply best weights
        best_weights = optimization_results['best_weights']
        logger.info(f'Best weights found: {best_weights}')
        
        # Run evaluation with best weights
        logger.info('Running evaluation with best weights')
        optimized_results = evaluator.evaluate()
        
        # Save optimized results
        with open(f'{output_dir}/optimized_evaluation.json', 'w') as f:
            json.dump(optimized_results, f, indent=2)
        
        # Print comparison
        print('\nWeight Optimization Results:')
        print(f'  Best weights: {best_weights}')
        print(f'  Best {args.optimization_metric}: {optimization_results["best_score"]:.4f}')
        
        print('\nComparison of Metrics:')
        print(f'{"Metric":<15} {"Initial":<10} {"Optimized":<10} {"Improvement":<10}')
        print('-' * 45)
        
        for metric in sorted(initial_results['average'].keys()):
            initial = initial_results['average'].get(metric, 0.0)
            optimized = optimized_results['average'].get(metric, 0.0)
            improvement = optimized - initial
            improvement_pct = (improvement / initial * 100) if initial > 0 else 0
            
            print(f'{metric:<15} {initial:.4f}{"":>5} {optimized:.4f}{"":>5} '
                 f'{improvement_pct:+.1f}%')
    
    logger.info(f'Example completed. Results saved to {output_dir}')

if __name__ == '__main__':
    main()
