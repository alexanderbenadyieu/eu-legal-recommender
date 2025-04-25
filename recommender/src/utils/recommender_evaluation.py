"""
Evaluation metrics for recommender systems using the Microsoft Recommenders library.

This module provides a wrapper around the Microsoft Recommenders library's evaluation
metrics for assessing the performance of the EU Legal Recommender system.
"""

from typing import List, Dict, Set, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Import evaluation metrics from Recommenders
from recommenders.evaluation.python_evaluation import (
    precision_at_k, 
    recall_at_k, 
    ndcg_at_k,
    map_at_k,
    diversity
)

# Additional metrics
from sklearn.metrics import mean_squared_error
from math import sqrt

from src.utils.logging import get_logger
from src.models.personalized_recommender import PersonalizedRecommender

# Set up logger
logger = get_logger(__name__)

class RecommenderEvaluator:
    """
    Evaluate recommender system performance using Microsoft Recommenders metrics.
    
    This class provides methods to evaluate a recommender system using standard
    information retrieval metrics like precision, recall, NDCG, MAP, and diversity.
    It also includes methods for temporal evaluation and performance benchmarking.
    """
    
    def __init__(self, recommender, test_data_path: Optional[str] = None, default_metric: str = 'ndcg@10'):
        """
        Initialize evaluator.
        
        Args:
            recommender: Recommender instance to evaluate
            test_data_path: Optional path to test data file
            default_metric: Default metric to use for reporting (default: ndcg@10)
        """
        self.recommender = recommender
        self.test_data = {}
        self.performance_metrics = {}
        self.metric = default_metric
        
        if test_data_path:
            self.load_test_data(test_data_path)
            
        logger.info(f"Initialized RecommenderEvaluator for {type(recommender).__name__}")
    
    def load_test_data(self, file_path: str) -> None:
        """
        Load test data from file.
        
        Args:
            file_path: Path to test data file (JSON)
        """
        try:
            with open(file_path, 'r') as f:
                self.test_data = json.load(f)
                
            logger.info(f"Loaded test data from {file_path} with {len(self.test_data)} queries")
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            raise ValueError(f"Failed to load test data: {str(e)}")
    
    def save_test_data(self, file_path: str) -> None:
        """
        Save test data to file.
        
        Args:
            file_path: Path to save test data (JSON)
        """
        try:
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(self.test_data, f, indent=2)
                
            logger.info(f"Saved test data to {file_path}")
        except Exception as e:
            logger.error(f"Error saving test data: {str(e)}")
            raise ValueError(f"Failed to save test data: {str(e)}")
    
    def add_test_query(self, query_id: str, query: str, relevant_docs: List[str], 
                     relevance_scores: Optional[Dict[str, float]] = None) -> None:
        """
        Add a test query to the test data.
        
        Args:
            query_id: Unique identifier for the query
            query: Query text
            relevant_docs: List of relevant document IDs
            relevance_scores: Optional dictionary mapping doc_ids to relevance scores
        """
        self.test_data[query_id] = {
            "query": query,
            "relevant_docs": relevant_docs,
            "relevance_scores": relevance_scores or {}
        }
        
        logger.info(f"Added test query {query_id} with {len(relevant_docs)} relevant documents")
    
    def evaluate(self, metrics: Optional[List[str]] = None, 
                k_values: Optional[List[int]] = None,
                evaluation_mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate recommender on test data.
        
        Args:
            metrics: List of metrics to compute
            k_values: List of k values for top-k metrics
            evaluation_mode: Optional mode for profile-based evaluation:
                - 'expert_only': Use only expert profile component
                - 'historical_only': Use only historical documents component
                - 'categorical_only': Use only categorical preferences component
                - 'full_profile': Use all profile components (default for profile-based evaluation)
            
        Returns:
            Dictionary of evaluation results
        """
        if not self.test_data:
            logger.error("No test data available for evaluation")
            raise ValueError("Test data must be loaded before evaluation")
            
        metrics = metrics or ['precision', 'recall', 'ndcg', 'map']
        k_values = k_values or [5, 10, 20]
        
        results = {}
        per_query_results = []
        
        # Get recommender type
        recommender_type = type(self.recommender).__name__
        logger.info(f"Evaluating {recommender_type} on {len(self.test_data)} test queries with mode={evaluation_mode or 'default'}")
        
        # Process each query in test data
        for query_id, query_data in self.test_data.items():
            try:
                # Get query text and relevant documents
                query_text = query_data.get('query', '')
                relevant_docs = set(query_data.get('relevant_docs', []))
                relevance_scores = query_data.get('relevance_scores', {})
                
                if not relevant_docs:
                    logger.warning(f"No relevant documents for query {query_id}, skipping")
                    continue
                
                # Add evaluation_mode to get_recommendations call if specified
                extra_args = {}
                if evaluation_mode:
                    extra_args['evaluation_mode'] = evaluation_mode
                    
                # Detect if this is a client profile evaluation (client ID ending with _client)
                is_profile_eval = '_client' in query_id
                
                # Special handling for personalized recommender in profile evaluation
                if is_profile_eval and isinstance(self.recommender, PersonalizedRecommender):
                    # For profile-based evaluation, we need to ensure the profile is loaded
                    client_id = query_id
                    
                    # Check if we have profile components in the test data
                    profile_components = query_data.get('profile_components', {})
                    
                    # First ensure profile exists for this client
                    # If we're in evaluation mode, we need to initialize the right profile components
                    if evaluation_mode == 'expert_only' and 'expert_profile' in profile_components:
                        # Create expert profile if testing expert-only mode
                        expert_desc = profile_components.get('expert_profile', {}).get('description', '')
                        if expert_desc:
                            try:
                                self.recommender.create_expert_profile(client_id, expert_desc)
                                logger.info(f"Created expert profile for {client_id} in expert_only mode")
                            except Exception as e:
                                logger.error(f"Error creating expert profile: {e}")
                    elif evaluation_mode == 'categorical_only' and 'categorical_preferences' in profile_components:
                        # Set categorical preferences if testing categorical-only mode
                        cat_prefs = profile_components.get('categorical_preferences', {})
                        if cat_prefs:
                            try:
                                self.recommender.set_categorical_preferences(client_id, cat_prefs)
                                logger.info(f"Set categorical preferences for {client_id} in categorical_only mode")
                            except Exception as e:
                                logger.error(f"Error setting categorical preferences: {e}")
                    elif evaluation_mode == 'historical_only' and 'historical_documents' in profile_components:
                        # Add historical documents if testing historical-only mode
                        hist_docs = profile_components.get('historical_documents', [])
                        for doc_id in hist_docs:
                            try:
                                # Handle potential parameter mismatch with add_historical_document
                                try:
                                    self.recommender.add_historical_document(client_id, doc_id)
                                    logger.debug(f"Added historical document {doc_id} to {client_id} profile")
                                except TypeError as te:
                                    # If temporal_boost error, retry without it
                                    if "temporal_boost" in str(te) or "reference_date" in str(te):
                                        # This is likely a PersonalizedRecommender.add_historical_document method
                                        # which doesn't accept temporal_boost or reference_date parameters
                                        logger.warning(f"Parameter mismatch when adding historical document {doc_id}, using alternative method")
                                        
                                        # Get the profile object if it's a PersonalizedRecommender
                                        if hasattr(self.recommender, 'get_user_profile'):
                                            profile = self.recommender.get_user_profile(client_id)
                                            
                                            # Get document data directly from Pinecone
                                            if hasattr(self.recommender, 'recommender'):
                                                doc_data = self.recommender.recommender.get_document_by_id(doc_id)
                                                if doc_data and hasattr(profile, 'add_historical_document'):
                                                    profile.add_historical_document(doc_id, doc_data)
                                                    logger.debug(f"Added historical document {doc_id} to {client_id} profile using alternative method")
                                                else:
                                                    logger.error(f"Could not retrieve document data for {doc_id}")
                                    else:
                                        raise
                            except Exception as e:
                                logger.error(f"Error adding historical document {doc_id}: {e}")
                        logger.info(f"Added {len(hist_docs)} historical documents for {client_id} in historical_only mode")
                    elif not evaluation_mode or evaluation_mode == 'full_profile':
                        # Full profile mode - initialize all components
                        # 1. Expert profile
                        expert_desc = profile_components.get('expert_profile', {}).get('description', '')
                        if expert_desc:
                            try:
                                self.recommender.create_expert_profile(client_id, expert_desc)
                                logger.info(f"Created expert profile for {client_id} in full_profile mode")
                            except Exception as e:
                                logger.error(f"Error creating expert profile: {e}")
                        
                        # 2. Categorical preferences
                        cat_prefs = profile_components.get('categorical_preferences', {})
                        if cat_prefs:
                            try:
                                self.recommender.set_categorical_preferences(client_id, cat_prefs)
                                logger.info(f"Set categorical preferences for {client_id} in full_profile mode")
                            except Exception as e:
                                logger.error(f"Error setting categorical preferences: {e}")
                                
                        # 3. Historical documents
                        hist_docs = profile_components.get('historical_documents', [])
                        for doc_id in hist_docs:
                            try:
                                # Handle potential parameter mismatch with add_historical_document
                                try:
                                    self.recommender.add_historical_document(client_id, doc_id)
                                    logger.debug(f"Added historical document {doc_id} to {client_id} profile")
                                except TypeError as te:
                                    # If temporal_boost error, retry without it
                                    if "temporal_boost" in str(te) or "reference_date" in str(te):
                                        # This is likely a PersonalizedRecommender.add_historical_document method
                                        # which doesn't accept temporal_boost or reference_date parameters
                                        logger.warning(f"Parameter mismatch when adding historical document {doc_id}, using alternative method")
                                        
                                        # Get the profile object if it's a PersonalizedRecommender
                                        if hasattr(self.recommender, 'get_user_profile'):
                                            profile = self.recommender.get_user_profile(client_id)
                                            
                                            # Get document data directly from Pinecone
                                            if hasattr(self.recommender, 'recommender'):
                                                doc_data = self.recommender.recommender.get_document_by_id(doc_id)
                                                if doc_data and hasattr(profile, 'add_historical_document'):
                                                    profile.add_historical_document(doc_id, doc_data)
                                                    logger.debug(f"Added historical document {doc_id} to {client_id} profile using alternative method")
                                                else:
                                                    logger.error(f"Could not retrieve document data for {doc_id}")
                                    else:
                                        raise
                            except Exception as e:
                                logger.error(f"Error adding historical document {doc_id}: {e}")
                        if hist_docs:
                            logger.info(f"Added {len(hist_docs)} historical documents for {client_id} in full_profile mode")
                    
                    # For profile-based evaluation, pass client_id through the query parameter
                    # The PersonalizedRecommender.get_recommendations method will detect this as a client ID
                    recommendations = self.recommender.get_recommendations(
                        query=client_id,  # Pass client_id as the query - the recommender will use this as user_id
                        limit=max(k_values),
                        **extra_args
                    )
                else:
                    # Standard query-based evaluation
                    recommendations = self.recommender.get_recommendations(
                        query=query_text,
                        limit=max(k_values),
                        **extra_args
                    )
                
                # Extract document IDs from recommendations
                if recommendations and isinstance(recommendations[0], dict):
                    rec_ids = [rec.get('id') for rec in recommendations]
                else:
                    rec_ids = recommendations
                    
                # Calculate evaluation metrics
                query_results = {
                    'query_id': query_id,
                    'metrics': self._calculate_metrics(
                        recommendations=rec_ids,
                        relevant_docs=relevant_docs,
                        relevance_scores=relevance_scores,
                        metrics=metrics,
                        k_values=k_values
                    )
                }
                
                per_query_results.append(query_results['metrics'])
                results[query_id] = query_results
                
            except Exception as e:
                logger.error(f"Error evaluating query {query_id}: {str(e)}")
                logger.info(f"Skipping query {query_id} due to error")
        
        # Calculate average metrics across all queries
        if per_query_results:
            results['average'] = self._average_results(per_query_results)
            logger.info(f"Completed evaluation with average {self.metric} = "
                       f"{results['average'].get(self.metric, 'N/A')}")
        else:
            logger.warning("No valid results to average")
            results['average'] = {}
            
        return results
    
    def _calculate_metrics(self, recommendations: List[str], 
                          relevant_docs: Set[str],
                          relevance_scores: Dict[str, float],
                          metrics: List[str],
                          k_values: List[int]) -> Dict[str, float]:
        """
        Calculate metrics for a single query using Recommenders library.
        
        Args:
            recommendations: List of recommended document IDs
            relevant_docs: Set of relevant document IDs
            relevance_scores: Dictionary mapping doc_ids to relevance scores
            metrics: List of metrics to compute
            k_values: List of k values
            
        Returns:
            Dictionary of metric values
        """
        results = {}
        
        # Convert to format expected by Recommenders
        # Create dataframes for ground truth and predictions
        gt_df = pd.DataFrame({
            'userID': [1] * len(relevant_docs),  # Dummy user ID for each ground truth
            'itemID': list(relevant_docs),
            'rating': [relevance_scores.get(doc_id, 1.0) for doc_id in relevant_docs]
        })
        
        pred_df = pd.DataFrame({
            'userID': [1] * len(recommendations),  # Dummy user ID
            'itemID': recommendations,
            'prediction': list(range(len(recommendations), 0, -1))  # Descending scores
        })
        
        # Calculate metrics
        for k in k_values:
            for metric in metrics:
                if metric == 'precision':
                    # Precision@k
                    value = precision_at_k(gt_df, pred_df, k=k)
                    results[f'precision@{k}'] = value
                    
                elif metric == 'recall':
                    # Recall@k
                    value = recall_at_k(gt_df, pred_df, k=k)
                    results[f'recall@{k}'] = value
                    
                elif metric == 'ndcg':
                    # NDCG@k
                    value = ndcg_at_k(gt_df, pred_df, k=k)
                    results[f'ndcg@{k}'] = value
                    
                elif metric == 'map':
                    # MAP@k
                    value = map_at_k(gt_df, pred_df, k=k)
                    results[f'map@{k}'] = value
                    
                elif metric == 'diversity':
                    # For diversity, we need item features
                    # This is a placeholder - actual implementation would need item features
                    # We'll use a dummy implementation for now
                    results[f'diversity@{k}'] = self._calculate_diversity(recommendations[:k])
        
        return results
    
    def _calculate_diversity(self, recommendations: List[str]) -> float:
        """
        Calculate diversity of recommendations.
        
        This is a placeholder implementation. For actual diversity calculation,
        we would need item features from the recommender.
        
        Args:
            recommendations: List of recommended document IDs
            
        Returns:
            Diversity score between 0 and 1
        """
        try:
            # Try to get document features from recommender
            feature_vectors = {}
            for doc_id in recommendations:
                # This assumes the recommender has a method to get document features
                # Adjust based on your actual recommender implementation
                try:
                    if hasattr(self.recommender, 'get_document_features'):
                        feature_vectors[doc_id] = self.recommender.get_document_features(doc_id)
                    elif hasattr(self.recommender, 'get_document_embedding'):
                        feature_vectors[doc_id] = self.recommender.get_document_embedding(doc_id)
                except:
                    pass
            
            # If we have features, calculate diversity
            if feature_vectors and len(feature_vectors) > 1:
                # Calculate pairwise similarities
                similarities = []
                doc_ids = list(feature_vectors.keys())
                for i in range(len(doc_ids)):
                    for j in range(i+1, len(doc_ids)):
                        v1 = feature_vectors[doc_ids[i]].reshape(1, -1)
                        v2 = feature_vectors[doc_ids[j]].reshape(1, -1)
                        sim = np.dot(v1, v2.T)[0][0] / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        similarities.append(sim)
                
                # Diversity is inverse of average similarity
                avg_similarity = sum(similarities) / len(similarities)
                return 1.0 - avg_similarity
            
            # Default diversity if we can't calculate
            return 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating diversity: {str(e)}")
            return 0.5
    
    def _average_results(self, per_query_results: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate average metrics across multiple queries.
        
        Args:
            per_query_results: List of per-query metric results
            
        Returns:
            Dictionary of averaged metric values
        """
        if not per_query_results:
            return {}
            
        # Initialize averages
        avg_results = {}
        
        # Get all metric keys
        all_metrics = set()
        for query_results in per_query_results:
            all_metrics.update(query_results.keys())
        
        # Calculate averages
        for metric in all_metrics:
            values = [results.get(metric, 0.0) for results in per_query_results]
            avg_results[metric] = sum(values) / len(values)
        
        return avg_results
    
    def evaluate_temporal_boost(self, query: str, date_range_days: int = 365, 
                           boost_weights: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                           k: int = 10, visualize: bool = True) -> Dict[str, Any]:
        """
        Evaluate the impact of temporal boosting on recommendation results.
        
        This method tests different temporal boost weights and analyzes how they
        affect the recency distribution of recommendations and ranking changes.
        
        Args:
            query: Query string to test
            date_range_days: Number of days to consider for date analysis
            boost_weights: List of temporal boost weights to test
            k: Number of recommendations to retrieve
            visualize: Whether to generate visualizations
            
        Returns:
            Dictionary of evaluation results
        """
        results = {
            "query": query,
            "boost_weights": boost_weights,
            "date_range_days": date_range_days,
            "k": k,
            "recommendations": {},
            "metrics": {}
        }
        
        # Get current date for reference
        ref_date = datetime.now().date()
        start_date = ref_date - timedelta(days=date_range_days)
        
        # Get recommendations with different boost weights
        for weight in boost_weights:
            try:
                # Check if recommender has temporal boost capability
                if hasattr(self.recommender, 'get_recommendations_with_temporal_boost'):
                    recs = self.recommender.get_recommendations_with_temporal_boost(
                        query=query,
                        limit=k,
                        temporal_boost_weight=weight,
                        ref_date=ref_date
                    )
                else:
                    # Fall back to standard recommendations if temporal boost not available
                    logger.warning("Recommender does not support temporal boost directly")
                    recs = self.recommender.get_recommendations(query=query, limit=k)
                    
                results["recommendations"][str(weight)] = recs
                
                # Calculate date-based metrics
                date_metrics = self._analyze_date_distribution(recs, start_date, ref_date)
                results["metrics"][str(weight)] = date_metrics
                
            except Exception as e:
                logger.error(f"Error evaluating with boost weight {weight}: {str(e)}")
                results["metrics"][str(weight)] = {"error": str(e)}
        
        # Compare rankings across different boost weights
        if len(boost_weights) > 1:
            results["ranking_changes"] = self._analyze_ranking_changes(
                {str(w): results["recommendations"][str(w)] for w in boost_weights if str(w) in results["recommendations"]}
            )
        
        # Generate visualizations
        if visualize and len(results["metrics"]) > 0:
            visualization_path = self._visualize_temporal_results(results)
            results["visualization_path"] = visualization_path
        
        return results
    
    def _analyze_date_distribution(self, recommendations: List[Dict[str, Any]], 
                                 start_date, ref_date) -> Dict[str, Any]:
        """
        Analyze the date distribution of recommendations.
        
        Args:
            recommendations: List of recommendation results
            start_date: Start date for analysis
            ref_date: Reference date (usually current date)
            
        Returns:
            Dictionary of date distribution metrics
        """
        metrics = {
            "avg_days_from_ref": None,
            "median_days_from_ref": None,
            "max_days_from_ref": None,
            "min_days_from_ref": None,
            "docs_within_30_days": 0,
            "docs_within_90_days": 0,
            "docs_within_180_days": 0,
            "docs_within_365_days": 0,
            "date_distribution": {}
        }
        
        # Extract dates from recommendations
        dates = []
        date_counts = {}
        
        for rec in recommendations:
            if "date" in rec and rec["date"]:
                try:
                    # Parse date string (format may vary)
                    if isinstance(rec["date"], str):
                        doc_date = datetime.strptime(rec["date"], "%Y-%m-%d").date()
                    elif isinstance(rec["date"], datetime):
                        doc_date = rec["date"].date()
                    else:
                        continue
                        
                    # Calculate days from reference date
                    days_diff = (ref_date - doc_date).days
                    
                    if days_diff >= 0:  # Only consider past dates
                        dates.append(days_diff)
                        
                        # Update period counts
                        if days_diff <= 30:
                            metrics["docs_within_30_days"] += 1
                        if days_diff <= 90:
                            metrics["docs_within_90_days"] += 1
                        if days_diff <= 180:
                            metrics["docs_within_180_days"] += 1
                        if days_diff <= 365:
                            metrics["docs_within_365_days"] += 1
                            
                        # Update date distribution
                        year_month = doc_date.strftime("%Y-%m")
                        date_counts[year_month] = date_counts.get(year_month, 0) + 1
                        
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error parsing date: {rec.get('date')}, {str(e)}")
        
        # Calculate statistics if dates are available
        if dates:
            metrics["avg_days_from_ref"] = np.mean(dates)
            metrics["median_days_from_ref"] = np.median(dates)
            metrics["max_days_from_ref"] = np.max(dates)
            metrics["min_days_from_ref"] = np.min(dates)
            
            # Sort date distribution by year-month
            metrics["date_distribution"] = {k: date_counts[k] for k in sorted(date_counts.keys())}
        
        return metrics
    
    def _analyze_ranking_changes(self, recommendations_by_weight: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Analyze how rankings change with different boost weights.
        
        Args:
            recommendations_by_weight: Dictionary mapping boost weights to recommendation lists
            
        Returns:
            Dictionary of ranking change metrics
        """
        weights = sorted(recommendations_by_weight.keys())
        if len(weights) < 2:
            return {}
            
        base_weight = weights[0]  # Usually 0.0 (no boost)
        base_recs = recommendations_by_weight[base_weight]
        
        # Create mapping of document IDs to ranks for base recommendations
        base_ranks = {rec["id"]: idx for idx, rec in enumerate(base_recs)}
        
        changes = {}
        for weight in weights[1:]:  # Skip base weight
            boosted_recs = recommendations_by_weight[weight]
            
            # Track position changes and new/removed documents
            position_changes = []
            new_docs = []
            removed_docs = []
            
            # Check for new documents that weren't in base recommendations
            for idx, rec in enumerate(boosted_recs):
                doc_id = rec["id"]
                if doc_id in base_ranks:
                    # Document exists in both - calculate position change
                    base_pos = base_ranks[doc_id]
                    change = base_pos - idx  # Positive means moved up, negative means moved down
                    position_changes.append({
                        "id": doc_id,
                        "base_position": base_pos,
                        "new_position": idx,
                        "change": change
                    })
                else:
                    # New document that wasn't in base recommendations
                    new_docs.append({
                        "id": doc_id,
                        "position": idx
                    })
            
            # Check for documents that were removed from base recommendations
            boosted_ids = {rec["id"] for rec in boosted_recs}
            for doc_id, base_pos in base_ranks.items():
                if doc_id not in boosted_ids:
                    removed_docs.append({
                        "id": doc_id,
                        "base_position": base_pos
                    })
            
            # Calculate summary statistics
            avg_change = np.mean([c["change"] for c in position_changes]) if position_changes else 0
            max_up = max([c["change"] for c in position_changes], default=0)
            max_down = min([c["change"] for c in position_changes], default=0)
            
            changes[weight] = {
                "position_changes": position_changes,
                "new_documents": new_docs,
                "removed_documents": removed_docs,
                "summary": {
                    "avg_position_change": avg_change,
                    "max_position_up": max_up,
                    "max_position_down": max_down,
                    "num_new_docs": len(new_docs),
                    "num_removed_docs": len(removed_docs),
                    "total_changes": len(position_changes) + len(new_docs) + len(removed_docs)
                }
            }
        
        return changes
    
    def _visualize_temporal_results(self, results: Dict[str, Any]) -> str:
        """
        Generate visualizations for temporal boost evaluation results.
        
        Args:
            results: Results from evaluate_temporal_boost
            
        Returns:
            Path to saved visualization file
        """
        # Create output directory if it doesn't exist
        output_dir = Path("output/evaluations")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plot style
        sns.set(style="whitegrid")
        plt.rcParams.update({"font.size": 12})
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Average document age by boost weight
        weights = sorted([float(w) for w in results["metrics"].keys()])
        avg_ages = [results["metrics"][str(w)]["avg_days_from_ref"] 
                   for w in weights if "avg_days_from_ref" in results["metrics"][str(w)]]
        
        if len(weights) == len(avg_ages) and len(weights) > 0:
            axes[0, 0].plot(weights, avg_ages, marker='o', linestyle='-')
            axes[0, 0].set_xlabel("Temporal Boost Weight")
            axes[0, 0].set_ylabel("Average Document Age (days)")
            axes[0, 0].set_title("Impact of Temporal Boost on Average Document Age")
            axes[0, 0].grid(True)
        
        # Plot 2: Document age distribution for different boost weights
        for weight in [weights[0], weights[-1]] if len(weights) > 1 else weights:
            weight_str = str(weight)
            if weight_str in results["metrics"] and "date_distribution" in results["metrics"][weight_str]:
                dist = results["metrics"][weight_str]["date_distribution"]
                if dist:
                    dates = list(dist.keys())[-12:]  # Last 12 months
                    counts = [dist.get(d, 0) for d in dates]
                    
                    axes[0, 1].bar(
                        [i + (0.2 if weight == weights[0] else 0) for i in range(len(dates))],
                        counts,
                        width=0.4,
                        alpha=0.7,
                        label=f"Boost = {weight}"
                    )
        
        axes[0, 1].set_xlabel("Month")
        axes[0, 1].set_ylabel("Number of Documents")
        axes[0, 1].set_title("Document Date Distribution")
        if len(weights) > 1:
            axes[0, 1].legend()
        if len(weights) > 0 and "date_distribution" in results["metrics"][str(weights[0])]:
            dates = list(results["metrics"][str(weights[0])]["date_distribution"].keys())[-12:]
            axes[0, 1].set_xticks(range(len(dates)))
            axes[0, 1].set_xticklabels(dates, rotation=45)
        
        # Plot 3: Recency metrics by boost weight
        if len(weights) > 0:
            metrics_30 = [results["metrics"][str(w)].get("docs_within_30_days", 0) for w in weights]
            metrics_90 = [results["metrics"][str(w)].get("docs_within_90_days", 0) for w in weights]
            metrics_365 = [results["metrics"][str(w)].get("docs_within_365_days", 0) for w in weights]
            
            axes[1, 0].plot(weights, metrics_30, marker='o', linestyle='-', label="Within 30 days")
            axes[1, 0].plot(weights, metrics_90, marker='s', linestyle='-', label="Within 90 days")
            axes[1, 0].plot(weights, metrics_365, marker='^', linestyle='-', label="Within 365 days")
            axes[1, 0].set_xlabel("Temporal Boost Weight")
            axes[1, 0].set_ylabel("Number of Documents")
            axes[1, 0].set_title("Documents Within Time Periods")
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot 4: Ranking changes summary
        if "ranking_changes" in results and len(results["ranking_changes"]) > 0:
            change_weights = sorted(results["ranking_changes"].keys())
            total_changes = [results["ranking_changes"][w]["summary"]["total_changes"] for w in change_weights]
            new_docs = [results["ranking_changes"][w]["summary"]["num_new_docs"] for w in change_weights]
            
            x = np.arange(len(change_weights))
            width = 0.35
            
            axes[1, 1].bar(x - width/2, total_changes, width, label="Total Changes")
            axes[1, 1].bar(x + width/2, new_docs, width, label="New Documents")
            
            axes[1, 1].set_xlabel("Temporal Boost Weight")
            axes[1, 1].set_ylabel("Count")
            axes[1, 1].set_title("Ranking Changes by Boost Weight")
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(change_weights)
            axes[1, 1].legend()
        
        # Add overall title
        plt.suptitle(f"Temporal Boost Evaluation for Query: {results['query']}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"temporal_eval_{timestamp}.png"
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Saved temporal evaluation visualization to {output_path}")
        return str(output_path)
    
    def benchmark_performance(self, queries: List[str], iterations: int = 3, 
                           profile_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Benchmark the performance of the recommender system.
        
        This method measures query response time, memory usage, and other
        performance metrics for a set of benchmark queries.
        
        Args:
            queries: List of query strings to benchmark
            iterations: Number of iterations to run for each query
            profile_path: Optional path to user profile for personalized recommendations
            
        Returns:
            Dictionary of benchmark results
        """
        results = {
            "query_times": [],
            "embedding_times": [],
            "memory_usage": [],
            "per_query": {}
        }
        
        # Load profile if provided
        profile = None
        if profile_path and hasattr(self.recommender, 'set_user_profile'):
            try:
                with open(profile_path, 'r') as f:
                    profile = json.load(f)
                self.recommender.set_user_profile(profile)
                logger.info(f"Loaded user profile from {profile_path}")
            except Exception as e:
                logger.error(f"Error loading profile: {str(e)}")
        
        # Run benchmarks for each query
        for query in queries:
            query_results = {
                "query": query,
                "iterations": []
            }
            
            for i in range(iterations):
                iteration_metrics = {}
                
                # Measure embedding generation time if applicable
                if hasattr(self.recommender, 'embedder'):
                    start_time = time.time()
                    _ = self.recommender.embedder.get_embedding(query)
                    embedding_time = time.time() - start_time
                    iteration_metrics["embedding_time"] = embedding_time
                    results["embedding_times"].append(embedding_time)
                
                # Measure query response time
                start_time = time.time()
                _ = self.recommender.get_recommendations(query=query, limit=10)
                query_time = time.time() - start_time
                iteration_metrics["query_time"] = query_time
                results["query_times"].append(query_time)
                
                # Add iteration results
                query_results["iterations"].append(iteration_metrics)
            
            # Calculate statistics for this query
            query_results["avg_query_time"] = np.mean([it["query_time"] for it in query_results["iterations"]])
            if "embedding_time" in query_results["iterations"][0]:
                query_results["avg_embedding_time"] = np.mean([it["embedding_time"] for it in query_results["iterations"]])
            
            results["per_query"][query] = query_results
        
        # Calculate overall statistics
        results["avg_query_time"] = np.mean(results["query_times"])
        results["std_query_time"] = np.std(results["query_times"])
        results["min_query_time"] = np.min(results["query_times"])
        results["max_query_time"] = np.max(results["query_times"])
        
        if results["embedding_times"]:
            results["avg_embedding_time"] = np.mean(results["embedding_times"])
            results["std_embedding_time"] = np.std(results["embedding_times"])
        
        return results
