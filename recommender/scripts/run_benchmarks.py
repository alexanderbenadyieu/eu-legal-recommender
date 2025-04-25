#!/usr/bin/env python
"""
Benchmark script for the EU Legal Recommender system.

This script measures the performance of the recommender system on various metrics:
- Query response time
- Embedding generation time
- Memory usage
- Recommendation quality (if ground truth is available)
- Weight optimization for similarity components
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from tqdm import tqdm

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import memory_profiler

from src.models.pinecone_recommender import PineconeRecommender
from src.models.personalized_recommender import PersonalizedRecommender
from src.utils.logging import setup_logger
from src.utils.recommender_evaluation import RecommenderEvaluator
from src.utils.weight_optimizer import WeightOptimizer

# Set up logger
logger = setup_logger('benchmarks', log_file='logs/benchmarks.log')


def run_performance_benchmark(recommender, queries: List[str], 
                             num_recommendations: int = 10) -> Dict[str, Any]:
    """
    Run performance benchmarks for the recommender.
    
    Args:
        recommender: Recommender instance to benchmark
        queries: List of query strings to test
        num_recommendations: Number of recommendations to retrieve
        
    Returns:
        Dictionary of benchmark results
    """
    results = {
        "query_times": [],
        "embedding_times": [],
        "memory_usage": []
    }
    
    logger.info(f"Running performance benchmark with {len(queries)} queries")
    
    for query in tqdm(queries, desc="Benchmarking queries"):
        # Measure embedding generation time
        start_time = time.time()
        if hasattr(recommender, 'embedder'):
            _ = recommender.embedder.get_embedding(query)
        embedding_time = time.time() - start_time
        results["embedding_times"].append(embedding_time)
        
        # Measure query response time
        start_time = time.time()
        _ = recommender.get_recommendations(query=query, limit=num_recommendations)
        query_time = time.time() - start_time
        results["query_times"].append(query_time)
        
        # Measure memory usage
        memory_usage = memory_profiler.memory_usage()[0]
        results["memory_usage"].append(memory_usage)
    
    # Calculate statistics
    for key in results:
        if results[key]:
            results[f"{key}_avg"] = np.mean(results[key])
            results[f"{key}_std"] = np.std(results[key])
            results[f"{key}_min"] = np.min(results[key])
            results[f"{key}_max"] = np.max(results[key])
    
    return results

def run_quality_benchmark(recommender, test_data_path: str, 
                         metrics: Optional[List[str]] = None,
                         k_values: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Run quality benchmarks for the recommender.
    
    Args:
        recommender: Recommender instance to benchmark
        test_data_path: Path to test data file
        metrics: List of metrics to compute
        k_values: List of k values for top-k metrics
        
    Returns:
        Dictionary of benchmark results
    """
    metrics = metrics or ['precision', 'recall', 'ndcg', 'map', 'diversity']
    k_values = k_values or [5, 10, 20]
    
    logger.info(f"Running quality benchmark with metrics: {metrics}")
    
    # Create evaluator
    evaluator = RecommenderEvaluator(recommender, test_data_path)
    
    # Run evaluation
    results = evaluator.evaluate(metrics=metrics, k_values=k_values)
    
    return results

def optimize_weights(recommender, test_data_path: str, 
                    weight_params: Dict[str, List[float]],
                    output_path: Optional[str] = None,
                    metric: str = 'ndcg@10') -> Dict[str, Any]:
    """
    Run weight optimization for the recommender.
    
    Args:
        recommender: Recommender instance to optimize
        test_data_path: Path to test data file
        weight_params: Dictionary mapping weight names to lists of values to try
        output_path: Path to save optimization results
        metric: Metric to optimize for
        
    Returns:
        Dictionary of optimization results
    """
    logger.info(f"Running weight optimization for {metric}")
    
    # Create optimizer
    optimizer = WeightOptimizer(
        recommender=recommender,
        test_data_path=test_data_path,
        metric=metric,
        config_path=output_path
    )
    
    # Run optimization
    results = optimizer.optimize(weight_params=weight_params)
    
    return results

def create_test_dataset(recommender, output_path: str, 
                       num_queries: int = 20,
                       num_relevant_per_query: int = 5,
                       query_file: Optional[str] = None) -> None:
    """
    Create a test dataset for evaluation.
    
    Args:
        recommender: Recommender instance to use
        output_path: Path to save test dataset
        num_queries: Number of queries to generate
        num_relevant_per_query: Number of relevant documents per query
        query_file: Optional file with predefined queries
    """
    logger.info(f"Creating test dataset with {num_queries} queries")
    
    # Load queries if file provided
    if query_file and os.path.exists(query_file):
        with open(query_file, 'r') as f:
            queries = json.load(f)
    else:
        # Generate sample queries
        # This is a placeholder - actual implementation would need domain knowledge
        queries = [
            {"id": f"q{i}", "text": f"Sample query {i}"}
            for i in range(1, num_queries + 1)
        ]
    
    # Create optimizer for test dataset creation
    optimizer = WeightOptimizer(recommender=recommender)
    
    # Create test dataset
    optimizer.create_test_dataset(
        output_path=output_path,
        queries=queries,
        num_relevant_per_query=num_relevant_per_query
    )


def visualize_results(results: Dict[str, Any], output_dir: str) -> None:
    """
    Visualize benchmark results.
    
    Args:
        results: Dictionary of benchmark results
        output_dir: Directory to save visualizations
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up plotting style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Visualize performance metrics if available
    if "query_times" in results:
        # Query time distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(results["query_times"], kde=True)
        plt.title("Query Response Time Distribution")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency")
        plt.savefig(f"{output_dir}/query_time_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # Memory usage over queries
        plt.figure(figsize=(10, 6))
        plt.plot(results["memory_usage"])
        plt.title("Memory Usage Over Queries")
        plt.xlabel("Query Number")
        plt.ylabel("Memory Usage (MB)")
        plt.savefig(f"{output_dir}/memory_usage.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    # Visualize quality metrics if available
    if "average" in results:
        # Extract metrics at different k values
        metrics_data = []
        for metric, value in results["average"].items():
            if '@' in metric:
                metric_name, k = metric.split('@')
                metrics_data.append({
                    "Metric": metric_name,
                    "K": int(k),
                    "Value": value
                })
        
        if metrics_data:
            # Create DataFrame
            df = pd.DataFrame(metrics_data)
            
            # Plot metrics by k value
            plt.figure(figsize=(12, 8))
            chart = sns.barplot(x="K", y="Value", hue="Metric", data=df)
            plt.title("Recommendation Quality Metrics at Different K Values")
            plt.xlabel("K (Number of Recommendations)")
            plt.ylabel("Metric Value")
            plt.legend(title="Metric")
            plt.savefig(f"{output_dir}/quality_metrics.png", dpi=300, bbox_inches="tight")
            plt.close()
    
    # Visualize weight optimization results if available
    if "all_results" in results:
        # Extract weight combinations and scores
        weights_data = []
        for result in results["all_results"]:
            weights = result["weights"]
            score = result["score"]
            
            # Add to data
            weights_data.append({
                **weights,
                "Score": score
            })
        
        if weights_data:
            # Create DataFrame
            df = pd.DataFrame(weights_data)
            
            # Plot top weight combinations
            top_n = min(10, len(df))
            plt.figure(figsize=(12, 8))
            df_top = df.sort_values("Score", ascending=False).head(top_n)
            
            # Melt DataFrame for easier plotting
            weight_cols = [col for col in df.columns if col != "Score"]
            df_melted = pd.melt(df_top.reset_index(), 
                               id_vars=["index", "Score"], 
                               value_vars=weight_cols,
                               var_name="Weight", 
                               value_name="Value")
            
            # Plot
            chart = sns.barplot(x="index", y="Value", hue="Weight", data=df_melted)
            plt.title(f"Top {top_n} Weight Combinations by Score")
            plt.xlabel("Combination Rank")
            plt.ylabel("Weight Value")
            plt.legend(title="Weight")
            
            # Add score labels
            for i, score in enumerate(df_top["Score"]):
                plt.text(i, 1.05, f"Score: {score:.4f}", ha="center")
            
            plt.savefig(f"{output_dir}/weight_optimization.png", dpi=300, bbox_inches="tight")
            plt.close()
    
    logger.info(f"Saved visualizations to {output_dir}")


def benchmark_personalized_recommender(recommender, queries, profile_path):
    """
    Benchmark a personalized recommender with a user profile.
    
    Args:
        recommender: PersonalizedRecommender instance
        queries: List of query strings
        profile_path: Path to user profile JSON file
        
    Returns:
        Dictionary with benchmark results
    """
    # Load user profile
    with open(profile_path, 'r') as f:
        profile = json.load(f)
        
    logger.info(f"Running personalized benchmark with profile from {profile_path}")
    
    query_times = []
    for query in tqdm(queries, desc="Benchmarking personalized queries"):
        # Measure query response time
        start_time = time.time()
        _ = recommender.get_personalized_recommendations(
            query=query,
            user_profile=profile,
            limit=10
        )
        query_time = time.time() - start_time
        query_times.append(query_time)
    
    # Calculate statistics
    results = {
        "avg": np.mean(query_times),
        "std": np.std(query_times),
        "min": np.min(query_times),
        "max": np.max(query_times)
    }
    
    return results


def run_benchmarks(api_key, index_name, queries_file=None, profile_path=None, output_file=None):
    """
    Run comprehensive benchmarks for the EU Legal Recommender system.
    
    Args:
        api_key: Pinecone API key
        index_name: Pinecone index name
        queries_file: Path to file with benchmark queries
        profile_path: Path to user profile for personalized benchmarks
        output_file: Path to output file for results
        
    Returns:
        Dictionary with benchmark results
    """
    # Initialize recommender
    logger.info(f"Initializing PineconeRecommender with index {index_name}")
    recommender = PineconeRecommender(
        api_key=api_key,
        index_name=index_name
    )
    
    # Load queries
    if queries_file and os.path.exists(queries_file):
        with open(queries_file, 'r') as f:
            queries_data = json.load(f)
            if isinstance(queries_data, list):
                if all(isinstance(q, str) for q in queries_data):
                    queries = queries_data
                elif all(isinstance(q, dict) and 'text' in q for q in queries_data):
                    queries = [q['text'] for q in queries_data]
                else:
                    logger.warning(f"Invalid queries format in {queries_file}. Using default queries.")
                    queries = [f"Sample query {i}" for i in range(1, 21)]
            else:
                logger.warning(f"Invalid queries format in {queries_file}. Using default queries.")
                queries = [f"Sample query {i}" for i in range(1, 21)]
    else:
        logger.info("No queries file provided. Using default queries.")
        queries = [f"Sample query {i}" for i in range(1, 21)]
    
    # Run performance benchmark
    logger.info("Running performance benchmark...")
    performance_results = run_performance_benchmark(
        recommender=recommender,
        queries=queries,
        num_recommendations=10
    )
    
    results = {
        "performance": performance_results
    }
    
    # Run personalized benchmark if profile provided
    if profile_path:
        personalized_recommender = PersonalizedRecommender(
            recommender=recommender
        )
        results['personalized'] = benchmark_personalized_recommender(
            recommender=personalized_recommender,
            queries=queries,
            profile_path=profile_path
        )
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create visualizations
        output_dir = os.path.dirname(output_file)
        visualize_results(results, output_dir)
    
    return results


def main():
    """Run the benchmark script from command line."""
    parser = argparse.ArgumentParser(description='Benchmark the EU Legal Recommender system')
    parser.add_argument('--api-key', required=True, help='Pinecone API key')
    parser.add_argument('--index-name', default='eu-legal-documents-legal-bert', help='Pinecone index name')
    parser.add_argument('--queries-file', type=Path, help='Path to file with benchmark queries')
    parser.add_argument('--profile-path', type=Path, help='Path to user profile for personalized benchmarks')
    parser.add_argument('--output-file', type=Path, help='Path to output file for results')
    
    args = parser.parse_args()
    
    results = run_benchmarks(
        api_key=args.api_key,
        index_name=args.index_name,
        queries_file=args.queries_file,
        profile_path=args.profile_path,
        output_file=args.output_file
    )
    
    # Print summary
    print("\nBenchmark Results:")
    for benchmark, stats in results.items():
        print(f"\n{benchmark.replace('_', ' ').title()}:")
        for stat, value in stats.items():
            print(f"  {stat}: {value:.4f}s")


if __name__ == '__main__':
    main()
