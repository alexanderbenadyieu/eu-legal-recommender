"""
User feedback collection and processing for the EU Legal Recommender system.

This module provides functionality to collect, store, and process user feedback
on document recommendations, which can be used to improve future recommendations.
"""
from typing import Dict, List, Optional, Union, Any
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.logging import get_logger
from src.utils.exceptions import ValidationError, ProcessingError

# Set up logger for this module
logger = get_logger(__name__)

class FeedbackManager:
    """Manage user feedback on document recommendations.
    
    This class provides methods for collecting, storing, and analyzing user feedback
    on document recommendations. The feedback can be used to improve future recommendations
    through various mechanisms like boosting frequently accessed documents or adjusting
    similarity weights.
    
    Attributes:
        feedback_store_path (Path): Path to the feedback storage file
        feedback_data (pd.DataFrame): DataFrame containing feedback records
    """
    
    def __init__(self, feedback_store_path: Union[str, Path]):
        """
        Initialize feedback manager.
        
        Args:
            feedback_store_path (Union[str, Path]): Path to the feedback storage file
        """
        # Convert string path to Path object if necessary
        if isinstance(feedback_store_path, str):
            feedback_store_path = Path(feedback_store_path)
            
        self.feedback_store_path = feedback_store_path
        
        # Create parent directory if it doesn't exist
        self.feedback_store_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load feedback data
        self.feedback_data = self._load_feedback_data()
        
        logger.info(f"Initialized FeedbackManager with store at {feedback_store_path}")
        
    def _load_feedback_data(self) -> pd.DataFrame:
        """
        Load feedback data from storage file.
        
        Returns:
            pd.DataFrame: DataFrame containing feedback records
        """
        if self.feedback_store_path.exists():
            try:
                # Load existing data
                feedback_data = pd.read_csv(self.feedback_store_path)
                logger.info(f"Loaded {len(feedback_data)} feedback records from {self.feedback_store_path}")
                return feedback_data
            except Exception as e:
                logger.warning(f"Error loading feedback data: {str(e)}. Creating new feedback store.")
        
        # Create new feedback store with required columns
        feedback_data = pd.DataFrame(columns=[
            'user_id',
            'document_id',
            'query',
            'rating',
            'feedback_type',
            'timestamp',
            'additional_data'
        ])
        
        return feedback_data
    
    def _save_feedback_data(self) -> None:
        """
        Save feedback data to storage file.
        
        Raises:
            ProcessingError: If there's an error saving the feedback data
        """
        try:
            self.feedback_data.to_csv(self.feedback_store_path, index=False)
            logger.info(f"Saved {len(self.feedback_data)} feedback records to {self.feedback_store_path}")
        except Exception as e:
            logger.error(f"Error saving feedback data: {str(e)}")
            raise ProcessingError(f"Failed to save feedback data: {str(e)}")
    
    def add_feedback(self,
                    user_id: str,
                    document_id: str,
                    rating: int,
                    query: Optional[str] = None,
                    feedback_type: str = 'rating',
                    additional_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a new feedback record.
        
        Args:
            user_id (str): ID of the user providing feedback
            document_id (str): ID of the document being rated
            rating (int): Numeric rating (typically 1-5)
            query (Optional[str], optional): Query that led to this recommendation. Defaults to None.
            feedback_type (str, optional): Type of feedback (e.g., 'rating', 'click', 'save'). Defaults to 'rating'.
            additional_data (Optional[Dict[str, Any]], optional): Additional feedback data. Defaults to None.
            
        Raises:
            ValidationError: If required parameters are invalid
        """
        # Validate inputs
        if not user_id or not isinstance(user_id, str):
            logger.error(f"Invalid user_id: {user_id}")
            raise ValidationError(f"user_id must be a non-empty string, got {user_id}")
            
        if not document_id or not isinstance(document_id, str):
            logger.error(f"Invalid document_id: {document_id}")
            raise ValidationError(f"document_id must be a non-empty string, got {document_id}")
            
        if not isinstance(rating, int) or rating < 1 or rating > 5:
            logger.error(f"Invalid rating: {rating}")
            raise ValidationError(f"rating must be an integer between 1 and 5, got {rating}")
        
        # Create new feedback record
        new_feedback = {
            'user_id': user_id,
            'document_id': document_id,
            'query': query,
            'rating': rating,
            'feedback_type': feedback_type,
            'timestamp': datetime.now().isoformat(),
            'additional_data': json.dumps(additional_data) if additional_data else None
        }
        
        # Add to feedback data
        self.feedback_data = pd.concat([self.feedback_data, pd.DataFrame([new_feedback])], ignore_index=True)
        
        # Save updated feedback data
        self._save_feedback_data()
        
        logger.info(f"Added feedback: user={user_id}, document={document_id}, rating={rating}")
    
    def get_document_ratings(self, document_id: str) -> Dict[str, Any]:
        """
        Get ratings statistics for a specific document.
        
        Args:
            document_id (str): ID of the document
            
        Returns:
            Dict[str, Any]: Dictionary containing rating statistics
        """
        # Filter feedback data for the specified document
        doc_feedback = self.feedback_data[self.feedback_data['document_id'] == document_id]
        
        if len(doc_feedback) == 0:
            return {
                'document_id': document_id,
                'count': 0,
                'average_rating': None,
                'rating_distribution': {}
            }
        
        # Calculate statistics
        ratings = doc_feedback['rating'].values
        rating_counts = doc_feedback['rating'].value_counts().to_dict()
        
        # Sort rating distribution by rating value
        rating_distribution = {str(i): rating_counts.get(i, 0) for i in range(1, 6)}
        
        return {
            'document_id': document_id,
            'count': len(doc_feedback),
            'average_rating': float(np.mean(ratings)),
            'rating_distribution': rating_distribution
        }
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get preference statistics for a specific user.
        
        Args:
            user_id (str): ID of the user
            
        Returns:
            Dict[str, Any]: Dictionary containing user preference statistics
        """
        # Filter feedback data for the specified user
        user_feedback = self.feedback_data[self.feedback_data['user_id'] == user_id]
        
        if len(user_feedback) == 0:
            return {
                'user_id': user_id,
                'feedback_count': 0,
                'average_rating': None,
                'highly_rated_documents': []
            }
        
        # Calculate statistics
        avg_rating = float(user_feedback['rating'].mean())
        
        # Get highly rated documents (rating >= 4)
        highly_rated = user_feedback[user_feedback['rating'] >= 4]
        highly_rated_docs = highly_rated['document_id'].unique().tolist()
        
        return {
            'user_id': user_id,
            'feedback_count': len(user_feedback),
            'average_rating': avg_rating,
            'highly_rated_documents': highly_rated_docs
        }
    
    def apply_feedback_to_results(self, 
                                results: List[Dict[str, Any]], 
                                user_id: Optional[str] = None,
                                boost_factor: float = 0.2) -> List[Dict[str, Any]]:
        """
        Apply feedback data to adjust recommendation results.
        
        This method boosts the scores of documents that have received positive feedback
        from the specified user or from all users if no user_id is provided.
        
        Args:
            results (List[Dict[str, Any]]): List of recommendation results
            user_id (Optional[str], optional): ID of the user. Defaults to None.
            boost_factor (float, optional): Factor to boost scores by. Defaults to 0.2.
            
        Returns:
            List[Dict[str, Any]]: Adjusted recommendation results
        """
        if len(results) == 0:
            return results
            
        # Create a copy of results to avoid modifying the original
        adjusted_results = results.copy()
        
        # Filter feedback data
        if user_id:
            relevant_feedback = self.feedback_data[self.feedback_data['user_id'] == user_id]
        else:
            relevant_feedback = self.feedback_data
            
        if len(relevant_feedback) == 0:
            return adjusted_results
            
        # Create a mapping of document_id to average rating
        doc_ratings = {}
        for doc_id, group in relevant_feedback.groupby('document_id'):
            doc_ratings[doc_id] = group['rating'].mean()
            
        # Apply boost based on ratings
        for i, result in enumerate(adjusted_results):
            doc_id = result.get('id')
            if doc_id in doc_ratings:
                avg_rating = doc_ratings[doc_id]
                # Normalize rating to [0, 1] range and apply boost
                rating_boost = ((avg_rating - 1) / 4) * boost_factor
                
                # Apply boost to score
                if 'score' in result:
                    original_score = result['score']
                    boosted_score = original_score * (1 + rating_boost)
                    adjusted_results[i]['score'] = boosted_score
                    adjusted_results[i]['original_score'] = original_score
                    adjusted_results[i]['feedback_boost'] = rating_boost
                    
        # Re-sort results by adjusted score
        adjusted_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        logger.info(f"Applied feedback adjustments to {len(adjusted_results)} results")
        return adjusted_results
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """
        Get overall feedback statistics.
        
        Returns:
            Dict[str, Any]: Dictionary containing feedback statistics
        """
        if len(self.feedback_data) == 0:
            return {
                'total_feedback_count': 0,
                'user_count': 0,
                'document_count': 0,
                'average_rating': None
            }
            
        # Calculate statistics
        total_count = len(self.feedback_data)
        user_count = self.feedback_data['user_id'].nunique()
        document_count = self.feedback_data['document_id'].nunique()
        avg_rating = float(self.feedback_data['rating'].mean())
        
        # Rating distribution
        rating_counts = self.feedback_data['rating'].value_counts().to_dict()
        rating_distribution = {str(i): rating_counts.get(i, 0) for i in range(1, 6)}
        
        return {
            'total_feedback_count': total_count,
            'user_count': user_count,
            'document_count': document_count,
            'average_rating': avg_rating,
            'rating_distribution': rating_distribution
        }
