"""
Evaluation metrics for recommender systems.

This module provides functions to evaluate the performance of recommender systems
using standard information retrieval metrics such as precision, recall, NDCG,
and diversity.
"""

from typing import List, Dict, Tuple, Set, Optional, Union, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.logging import get_logger

# Set up logger
logger = get_logger(__name__)

def precision_at_k(recommendations: List[str], 
                  relevant_docs: Set[str], 
                  k: int) -> float:
    """
    Calculate precision@k for a single query.
    
    Args:
        recommendations: List of recommended document IDs
        relevant_docs: Set of relevant document IDs
        k: Number of top recommendations to consider
        
    Returns:
        Precision@k value between 0 and 1
    """
    if not recommendations or k <= 0:
        return 0.0
        
    # Get top-k recommendations
    top_k = recommendations[:min(k, len(recommendations))]
    
    # Count relevant items in top-k
    relevant_count = sum(1 for doc_id in top_k if doc_id in relevant_docs)
    
    # Calculate precision
    return relevant_count / len(top_k)

def recall_at_k(recommendations: List[str], 
               relevant_docs: Set[str], 
               k: int) -> float:
    """
    Calculate recall@k for a single query.
    
    Args:
        recommendations: List of recommended document IDs
        relevant_docs: Set of relevant document IDs
        k: Number of top recommendations to consider
        
    Returns:
        Recall@k value between 0 and 1
    """
    if not relevant_docs or k <= 0:
        return 0.0
        
    # Get top-k recommendations
    top_k = recommendations[:min(k, len(recommendations))]
    
    # Count relevant items in top-k
    relevant_count = sum(1 for doc_id in top_k if doc_id in relevant_docs)
    
    # Calculate recall
    return relevant_count / len(relevant_docs)

def ndcg_at_k(recommendations: List[str], 
             relevant_docs: Set[str], 
             k: int,
             relevance_scores: Optional[Dict[str, float]] = None) -> float:
    """
    Calculate NDCG@k (Normalized Discounted Cumulative Gain) for a single query.
    
    Args:
        recommendations: List of recommended document IDs
        relevant_docs: Set of relevant document IDs
        k: Number of top recommendations to consider
        relevance_scores: Optional dictionary mapping doc_ids to relevance scores.
                         If not provided, binary relevance is assumed (1 if relevant, 0 if not)
        
    Returns:
        NDCG@k value between 0 and 1
    """
    if not relevant_docs or k <= 0:
        return 0.0
        
    # Get top-k recommendations
    top_k = recommendations[:min(k, len(recommendations))]
    
    # Calculate DCG
    dcg = 0.0
    for i, doc_id in enumerate(top_k):
        # Get relevance score (1 if relevant and no scores provided)
        if relevance_scores:
            rel = relevance_scores.get(doc_id, 0.0)
        else:
            rel = 1.0 if doc_id in relevant_docs else 0.0
            
        # Add to DCG with log2(i+2) discount (i+2 because i is 0-indexed)
        dcg += rel / np.log2(i + 2)
    
    # Calculate ideal DCG
    # Sort relevant documents by relevance score (if provided)
    if relevance_scores:
        # Get scores for relevant documents
        rel_scores = [(doc_id, relevance_scores.get(doc_id, 0.0)) 
                     for doc_id in relevant_docs]
        # Sort by score in descending order
        rel_scores.sort(key=lambda x: x[1], reverse=True)
        # Take top-k
        ideal_relevance = [score for _, score in rel_scores[:k]]
    else:
        # Binary relevance - all relevant docs have score 1.0
        ideal_relevance = [1.0] * min(len(relevant_docs), k)
    
    # Calculate IDCG
    idcg = 0.0
    for i, rel in enumerate(ideal_relevance):
        idcg += rel / np.log2(i + 2)
    
    # Return NDCG
    if idcg == 0:
        return 0.0
    return dcg / idcg

def diversity(recommendations: List[str], 
             feature_vectors: Dict[str, np.ndarray],
             k: int) -> float:
    """
    Calculate diversity of recommendations based on feature dissimilarity.
    
    Higher values indicate more diverse recommendations.
    
    Args:
        recommendations: List of recommended document IDs
        feature_vectors: Dictionary mapping doc_ids to feature vectors
        k: Number of top recommendations to consider
        
    Returns:
        Diversity score between 0 and 1
    """
    if not recommendations or k <= 1:
        return 1.0  # Maximum diversity for 0 or 1 recommendations
        
    # Get top-k recommendations
    top_k = recommendations[:min(k, len(recommendations))]
    
    # Get feature vectors for top-k
    vectors = []
    for doc_id in top_k:
        if doc_id in feature_vectors:
            vectors.append(feature_vectors[doc_id])
    
    # If we don't have enough vectors, return maximum diversity
    if len(vectors) <= 1:
        return 1.0
        
    # Calculate pairwise similarities
    similarities = []
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            # Reshape vectors for cosine_similarity
            v1 = vectors[i].reshape(1, -1)
            v2 = vectors[j].reshape(1, -1)
            sim = cosine_similarity(v1, v2)[0][0]
            similarities.append(sim)
    
    # Calculate average similarity
    avg_similarity = sum(similarities) / len(similarities)
    
    # Diversity is inverse of similarity
    return 1.0 - avg_similarity

def calculate_all_metrics(recommendations: List[str],
                         relevant_docs: Set[str],
                         feature_vectors: Optional[Dict[str, np.ndarray]] = None,
                         relevance_scores: Optional[Dict[str, float]] = None,
                         k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    Calculate all evaluation metrics for a single query.
    
    Args:
        recommendations: List of recommended document IDs
        relevant_docs: Set of relevant document IDs
        feature_vectors: Optional dictionary mapping doc_ids to feature vectors (for diversity)
        relevance_scores: Optional dictionary mapping doc_ids to relevance scores
        k_values: List of k values for which to calculate metrics
        
    Returns:
        Dictionary of metric names to values
    """
    results = {}
    
    for k in k_values:
        # Calculate precision, recall, and NDCG for each k
        results[f'precision@{k}'] = precision_at_k(recommendations, relevant_docs, k)
        results[f'recall@{k}'] = recall_at_k(recommendations, relevant_docs, k)
        results[f'ndcg@{k}'] = ndcg_at_k(recommendations, relevant_docs, k, relevance_scores)
        
        # Calculate diversity if feature vectors are provided
        if feature_vectors:
            results[f'diversity@{k}'] = diversity(recommendations, feature_vectors, k)
    
    return results

def average_metrics(per_query_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate average metrics across multiple queries.
    
    Args:
        per_query_results: Dictionary mapping query IDs to metric results
        
    Returns:
        Dictionary of averaged metric values
    """
    if not per_query_results:
        return {}
        
    # Initialize averages
    avg_results = {}
    
    # Get all metric keys
    all_metrics = set()
    for query_results in per_query_results.values():
        all_metrics.update(query_results.keys())
    
    # Calculate averages
    for metric in all_metrics:
        values = [results.get(metric, 0.0) for results in per_query_results.values()]
        avg_results[metric] = sum(values) / len(values)
    
    return avg_results
