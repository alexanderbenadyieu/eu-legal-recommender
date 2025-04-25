"""
Weight optimization module for the EU Legal Recommender system.

This module provides tools to find optimal weights for different components
of the similarity computation in the recommender system.
"""

from typing import List, Dict, Tuple, Optional, Any, Callable
import numpy as np
import json
from pathlib import Path
import itertools
import time
from tqdm import tqdm

from src.utils.recommender_evaluation import RecommenderEvaluator
from src.utils.logging import get_logger

# Set up logger
logger = get_logger(__name__)

class WeightOptimizer:
    """
    Optimize weights for similarity computation in the recommender system.
    
    This class implements a grid search to find optimal weights for different
    components of the similarity computation (e.g., text similarity vs. categorical
    feature similarity).
    """
    
    def __init__(self, recommender, evaluator: Optional[RecommenderEvaluator] = None,
                test_data_path: Optional[str] = None, 
                metric: str = 'ndcg@10',
                config_path: Optional[str] = None):
        """
        Initialize weight optimizer.
        
        Args:
            recommender: Recommender instance to optimize
            evaluator: Optional RecommenderEvaluator instance
            test_data_path: Path to test data file (if evaluator not provided)
            metric: Metric to optimize for (default: ndcg@10)
            config_path: Path to save/load configuration
        """
        self.recommender = recommender
        
        # Set up evaluator if not provided
        if evaluator:
            self.evaluator = evaluator
        elif test_data_path:
            self.evaluator = RecommenderEvaluator(recommender, test_data_path)
        else:
            raise ValueError("Either evaluator or test_data_path must be provided")
        
        self.metric = metric
        self.config_path = config_path
        self.results = {}
        
        logger.info(f"Initialized WeightOptimizer for {type(recommender).__name__}, "
                   f"optimizing for {metric}")
    
    def optimize(self, weight_params: Dict[str, List[float]], 
                k_values: Optional[List[int]] = None,
                metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run grid search to find optimal weights.
        
        Args:
            weight_params: Dictionary mapping weight names to lists of values to try
            k_values: List of k values for evaluation
            metrics: List of metrics to compute
            
        Returns:
            Dictionary with optimization results
        """
        # Default values
        k_values = k_values or [5, 10, 20]
        metrics = metrics or ['precision', 'recall', 'ndcg', 'map']
        
        # Generate all weight combinations
        param_names = list(weight_params.keys())
        param_values = list(weight_params.values())
        combinations = list(itertools.product(*param_values))
        
        logger.info(f"Running grid search with {len(combinations)} weight combinations")
        
        # Track results
        all_results = []
        best_score = -1
        best_weights = None
        
        # Run grid search
        for combo in tqdm(combinations, desc="Evaluating weight combinations"):
            # Create weight dictionary
            weights = {name: value for name, value in zip(param_names, combo)}
            
            # Skip combos where all weights are zero and normalize otherwise
            total = sum(weights.values())
            if total == 0:
                continue
            if len(weights) > 1 and total != 1.0:
                # Normalize weights to sum to 1.0
                weights = {k: v/total for k, v in weights.items()}
            
            # Apply weights to recommender - create a copy to avoid modifying the original
            weights_copy = weights.copy()
            self._apply_weights(weights_copy)
            
            # Evaluate recommender, guard against evaluation errors
            try:
                # Pass the evaluation mode if we detected a specific profile configuration
                eval_results = self.evaluator.evaluate(
                    metrics=metrics, 
                    k_values=k_values,
                    evaluation_mode=self.evaluation_mode
                )
                avg_results = eval_results.get("average", {})
            except Exception as e:
                logger.warning(f"Skipping weights {weights} due to evaluation error: {str(e)}")
                # assign zero scores for all requested metrics/k
                avg_results = {f"{m}@{k}": 0.0 for m in metrics for k in k_values}
            
            # Get score for target metric
            score = avg_results.get(self.metric, 0.0)
            
            # Track result
            result = {
                "weights": weights,
                "score": score,
                "metrics": avg_results
            }
            all_results.append(result)
            
            # Update best if better
            if score > best_score:
                best_score = score
                best_weights = weights
                logger.info(f"New best weights found: {weights} with {self.metric}={score:.4f}")
        
        # Sort results by score
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Compile results
        self.results = {
            "best_weights": best_weights,
            "best_score": best_score,
            "metric": self.metric,
            "all_results": all_results
        }
        
        # Save results if config path provided
        if self.config_path:
            self.save_weights(self.config_path)
        
        return self.results
    
    def _apply_weights(self, weights: Dict[str, float]) -> None:
        """
        Apply weights to recommender.
        
        This method needs to be adapted based on the specific recommender implementation.
        
        Args:
            weights: Dictionary of weight values
        """
        try:
            # Store the weights for later use in evaluation
            self.current_weights = weights
            
            # Determine if we're using a specific evaluation mode
            self.evaluation_mode = None
            
            # Expert-only mode check for profile components
            if all(k in weights for k in ['expert_weight', 'historical_weight', 'categorical_weight']):
                # Determine which profile components to use based on weights
                if abs(weights['expert_weight'] - 1.0) < 0.001 and abs(weights['historical_weight']) < 0.001 and abs(weights['categorical_weight']) < 0.001:
                    # Only expert profile (within small epsilon of exact values)
                    self.evaluation_mode = 'expert_only'
                    logger.info("Using expert-only profile configuration for evaluation")
                elif abs(weights['expert_weight']) < 0.001 and abs(weights['historical_weight'] - 1.0) < 0.001 and abs(weights['categorical_weight']) < 0.001:
                    # Only historical documents
                    self.evaluation_mode = 'historical_only'
                    logger.info("Using historical-only profile configuration for evaluation")
                elif abs(weights['expert_weight']) < 0.001 and abs(weights['historical_weight']) < 0.001 and abs(weights['categorical_weight'] - 1.0) < 0.001:
                    # Only categorical preferences
                    self.evaluation_mode = 'categorical_only'
                    logger.info("Using categorical-only profile configuration for evaluation")
                else:
                    # Mixed profile (using multiple components with their weights)
                    self.evaluation_mode = 'full_profile'
                    logger.info(f"Using full profile configuration with weights: expert={weights['expert_weight']:.2f}, " 
                              f"historical={weights['historical_weight']:.2f}, categorical={weights['categorical_weight']:.2f}")
            
            # Apply weights based on which recommender we're using
            if hasattr(self.recommender, 'set_weights'):
                # PersonalizedRecommender 
                try:
                    # Filter weights to only include ones applicable to the recommender
                    filtered_weights = {}
                    for key, value in weights.items():
                        # Only include weights that the recommender expects
                        if hasattr(self.recommender, f"_{key}") or key in ['expert_weight', 'historical_weight', 
                                                                          'categorical_weight', 'text_weight', 
                                                                          'summary_weight', 'keyword_weight']:
                            filtered_weights[key] = value
                    
                    # Apply filtered weights
                    self.recommender.set_weights(**filtered_weights)
                    logger.info(f"Applied weights to PersonalizedRecommender: {filtered_weights}")
                except Exception as e:
                    logger.error(f"Error applying weights to PersonalizedRecommender: {str(e)}")
                
            elif hasattr(self.recommender, 'set_similarity_weights'):
                # PineconeRecommender - text vs categorical weights
                if 'text_weight' in weights and 'categorical_weight' in weights:
                    try:
                        self.recommender.set_similarity_weights(
                            text_weight=weights['text_weight'],
                            categorical_weight=weights['categorical_weight']
                        )
                        logger.info(f"Applied similarity weights to PineconeRecommender: "
                                  f"text={weights['text_weight']:.2f}, "
                                  f"categorical={weights['categorical_weight']:.2f}")
                    except Exception as e:
                        logger.error(f"Error applying similarity weights to PineconeRecommender: {str(e)}")
                    
            else:
                # Generic recommender - no weight configuration
                logger.warning("Recommender does not support weight configuration")
                
        except Exception as e:
            logger.error(f"Error applying weights: {str(e)}")
            raise ValueError(f"Failed to apply weights: {str(e)}")
    
    def save_weights(self, file_path: Optional[str] = None) -> None:
        """
        Save optimization results to file.
        
        Args:
            file_path: Path to save results (JSON)
        """
        file_path = file_path or self.config_path
        
        if not file_path:
            logger.warning("No file path provided for saving weights")
            return
        
        try:
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save results
            with open(file_path, 'w') as f:
                json.dump(self.results, f, indent=2)
                
            logger.info(f"Saved weight optimization results to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving weights: {str(e)}")
            raise ValueError(f"Failed to save weights: {str(e)}")
    
    def load_weights(self, file_path: Optional[str] = None) -> Dict[str, float]:
        """
        Load weights from file and apply to recommender.
        
        Args:
            file_path: Path to load weights from (JSON)
            
        Returns:
            Dictionary of loaded weights
        """
        file_path = file_path or self.config_path
        
        if not file_path:
            logger.warning("No file path provided for loading weights")
            return {}
        
        try:
            # Load results
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Get best weights
            weights = data.get("best_weights", {})
            
            # Apply weights
            if weights:
                self._apply_weights(weights)
                logger.info(f"Loaded and applied weights from {file_path}")
            
            return weights
            
        except Exception as e:
            logger.error(f"Error loading weights: {str(e)}")
            raise ValueError(f"Failed to load weights: {str(e)}")
    
    def create_test_dataset(self, output_path: str, 
                          queries: List[Dict[str, Any]],
                          num_relevant_per_query: int = 5) -> None:
        """
        Create a test dataset with ground truth relevance judgments.
        
        Args:
            output_path: Path to save test dataset
            queries: List of query dictionaries with 'id' and 'text' keys
            num_relevant_per_query: Number of relevant documents to include per query
        """
        test_data = {}
        
        for query_info in tqdm(queries, desc="Creating test dataset"):
            query_id = query_info['id']
            query_text = query_info['text']
            
            # Get recommendations for query
            try:
                recommendations = self.recommender.get_recommendations(
                    query=query_text,
                    limit=num_relevant_per_query
                )
                
                # Convert to list of IDs if recommendations are more complex
                if recommendations and isinstance(recommendations[0], dict):
                    rec_ids = [rec.get('id') for rec in recommendations]
                else:
                    rec_ids = recommendations
                
                # Add to test data
                test_data[query_id] = {
                    "query": query_text,
                    "relevant_docs": rec_ids,
                    "relevance_scores": {doc_id: 1.0 for doc_id in rec_ids}
                }
                
            except Exception as e:
                logger.error(f"Error processing query {query_id}: {str(e)}")
        
        # Save test data
        try:
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(test_data, f, indent=2)
                
            logger.info(f"Created test dataset with {len(test_data)} queries at {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving test dataset: {str(e)}")
            raise ValueError(f"Failed to save test dataset: {str(e)}")
    
    def import_expert_judgments(self, file_path: str) -> None:
        """
        Import expert-created relevance judgments.
        
        Args:
            file_path: Path to expert judgments file (JSON)
        """
        try:
            with open(file_path, 'r') as f:
                expert_data = json.load(f)
            
            # Add to evaluator's test data
            for query_id, query_data in expert_data.items():
                self.evaluator.add_test_query(
                    query_id=query_id,
                    query=query_data.get("query", ""),
                    relevant_docs=query_data.get("relevant_docs", []),
                    relevance_scores=query_data.get("relevance_scores", {})
                )
            
            logger.info(f"Imported {len(expert_data)} expert judgments from {file_path}")
            
        except Exception as e:
            logger.error(f"Error importing expert judgments: {str(e)}")
            raise ValueError(f"Failed to import expert judgments: {str(e)}")
