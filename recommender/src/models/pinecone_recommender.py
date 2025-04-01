"""
Recommender system using Pinecone vector database for legal document recommendations.

This module provides the core recommender system functionality, using Pinecone
for vector similarity search and BERT embeddings for semantic understanding.
"""
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import pinecone
import sys
import os

# Add the parent directory to the path so we can import database_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import database_utils

from .embeddings import BERTEmbedder
from .features import FeatureProcessor
from src.utils.logging import get_logger
from src.utils.exceptions import (
    PineconeError, 
    ValidationError, 
    EmbeddingError,
    RecommendationError
)

# Set up logger for this module
logger = get_logger(__name__)

class PineconeRecommender:
    """Recommend legal documents using Pinecone vector database."""
    
    def __init__(
        self,
        api_key: str,
        index_name: str = "eu-legal-documents-legal-bert",
        embedder_model: str = "nlpaueb/legal-bert-base-uncased",
        feature_processor: Optional[FeatureProcessor] = None,
        text_weight: float = 0.7,
        categorical_weight: float = 0.3,
    ):
        """
        Initialize Pinecone recommender.
        
        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index to use
            embedder_model: Name of the BERT model to use for query embedding
            feature_processor: Optional FeatureProcessor for categorical features
            text_weight: Weight for text embedding similarity (0-1)
            categorical_weight: Weight for categorical feature similarity (0-1)
            
        Raises:
            ValidationError: If weights don't sum to 1.0
            EmbeddingError: If there's an error initializing the embedder
            PineconeError: If there's an error connecting to Pinecone
        """
        logger.info(f"Initializing PineconeRecommender with index {index_name} and model {embedder_model}")
        
        # Validate weights
        if text_weight + categorical_weight != 1.0:
            logger.error(f"Invalid weights: text_weight={text_weight}, categorical_weight={categorical_weight}")
            raise ValidationError("Weights must sum to 1.0", code="INVALID_WEIGHTS")
            
        self.api_key = api_key
        self.index_name = index_name
        self.text_weight = text_weight
        self.categorical_weight = categorical_weight
        self.client_preferences = None  # Initialize client preferences to None
        
        # Initialize embedder with the same model used for the Pinecone index
        self.embedder = BERTEmbedder(model_name=embedder_model)
        
        # Initialize feature processor if provided
        self.feature_processor = feature_processor
        
        # Connect to Pinecone
        self._init_pinecone()
        
    def _init_pinecone(self) -> None:
        """
        Initialize connection to Pinecone.
        
        Raises:
            PineconeError: If there's an error connecting to Pinecone or the index doesn't exist
        """
        logger.info(f"Connecting to Pinecone index: {self.index_name}")
        
        try:
            # Initialize Pinecone with API key
            self.pc = pinecone.Pinecone(api_key=self.api_key)
            
            # Check if index exists
            if self.index_name not in self.pc.list_indexes().names():
                logger.error(f"Pinecone index '{self.index_name}' does not exist")
                raise PineconeError(f"Index '{self.index_name}' does not exist", code="INDEX_NOT_FOUND")
                
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            
            # Get index stats
            stats = self.index.describe_index_stats()
            logger.info(f"Connected to Pinecone index with {stats.total_vector_count} vectors")
            logger.debug(f"Vector dimension: {stats.dimension}")
            
        except pinecone.core.client.exceptions.ApiException as e:
            logger.error(f"Pinecone API error: {str(e)}")
            raise PineconeError(f"API error: {str(e)}", code="API_ERROR")
        except Exception as e:
            logger.error(f"Unexpected error connecting to Pinecone: {str(e)}")
            raise PineconeError(f"Unexpected error: {str(e)}", code="UNEXPECTED_ERROR")
        
    def get_recommendations_with_embedding(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        query_keywords: Optional[List[str]] = None,
        query_features: Optional[Dict[str, Union[str, List[str]]]] = None,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_categorical: bool = True,
        client_preferences: Optional[Dict[str, float]] = None,
        embedding_type: str = 'combined',
        temporal_boost: Optional[float] = None,
        reference_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Get document recommendations using a pre-computed query embedding.
        
        Args:
            query_embedding: Pre-computed embedding vector for the query
            query_text: Query text (for logging purposes)
            query_keywords: Optional list of keywords to enhance the query
            query_features: Optional dictionary of categorical features
            top_k: Number of recommendations to return
            filter: Optional Pinecone metadata filter
            include_categorical: Whether to include categorical features in scoring
            client_preferences: Optional dictionary mapping client-weighted features
                             (e.g., 'form', 'author') to their preference weights
            embedding_type: Type of embedding to use for the query ('combined', 'summary', or 'keyword')
            
        Returns:
            List of recommended documents with scores and metadata
        """
        # Process categorical features if provided
        if include_categorical and query_features:
            logger.info(f"Processing categorical features for query: {query_features}")
            try:
                # Store the categorical features directly as a dictionary
                # We'll use this for matching against document categorical features
                self._query_categorical_features = query_features
                logger.debug(f"Query categorical features: {self._query_categorical_features}")
            except Exception as e:
                logger.warning(f"Error processing categorical features: {e}")
                self._query_categorical_features = None
                # We continue execution despite this error, as it's non-critical
        else:
            logger.debug("No categorical features to process or feature processing disabled")
            self._query_categorical_features = None
        
        # Prepare filter to match the embedding type if specified
        if embedding_type in ['summary', 'keyword'] and filter is None:
            # Only filter by embedding_type if no other filter is specified
            filter = {'embedding_type': embedding_type}
        elif embedding_type in ['summary', 'keyword'] and isinstance(filter, dict):
            # Add embedding_type to existing filter
            filter = {**filter, 'embedding_type': embedding_type}
        
        # Query Pinecone
        logger.info(f"Querying Pinecone with pre-computed embedding: '{query_text[:50]}...'")
        try:
            results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k * 2 if self._query_categorical_features is not None else top_k,  # Get more results for reranking
                include_metadata=True,
                filter=filter
            )
            logger.debug(f"Received {len(results.matches)} matches from Pinecone")
        except pinecone.core.client.exceptions.ApiException as e:
            logger.error(f"Pinecone query error: {str(e)}")
            raise PineconeError(f"Query error: {str(e)}", code="QUERY_ERROR")
        except Exception as e:
            logger.error(f"Unexpected error querying Pinecone: {str(e)}")
            raise RecommendationError(f"Failed to query vector database: {str(e)}", query=query_text[:50])
        
        # Process results
        recommendations = []
        for match in results.matches:
            # Extract the document's categorical features from metadata if available
            doc_categorical_features = None
            cat_sim = None
            
            if self._query_categorical_features is not None and match.metadata and 'categorical_features' in match.metadata:
                try:
                    # Parse the JSON string of categorical features
                    feature_str = match.metadata['categorical_features']
                    if isinstance(feature_str, str):
                        doc_features_dict = json.loads(feature_str)
                        
                        # Log the categorical features for debugging
                        logger.debug(f"Document {match.id} categorical features: {doc_features_dict}")
                        
                        # Use weighted feature matching for better similarity calculation
                        weighted_matches = 0.0
                        weighted_total = 0.0
                        
                        # Process each feature type with its appropriate weight
                        for feature_name, query_values in self._query_categorical_features.items():
                            if feature_name in doc_features_dict and feature_name in self.feature_processor.feature_weights:
                                doc_values = doc_features_dict[feature_name]
                                feature_weight = self.feature_processor.feature_weights[feature_name]
                                
                                # Handle different feature types appropriately
                                if feature_name in self.feature_processor.multi_valued_features:
                                    # For multi-valued features, calculate Jaccard similarity
                                    query_set = set(query_values)
                                    doc_set = set(doc_values)
                                    
                                    if len(query_set) > 0:
                                        # Calculate Jaccard similarity: intersection / union
                                        intersection = len(query_set.intersection(doc_set))
                                        union = len(query_set.union(doc_set))
                                        similarity = intersection / union if union > 0 else 0
                                        
                                        weighted_matches += similarity * feature_weight
                                        weighted_total += feature_weight
                                        
                                        logger.debug(f"Feature '{feature_name}' similarity: {similarity} (weight: {feature_weight})")
                                else:
                                    # For single-valued features, exact match
                                    query_value = query_values[0] if isinstance(query_values, list) else query_values
                                    doc_value = doc_values[0] if isinstance(doc_values, list) else doc_values
                                    
                                    if query_value == doc_value:
                                        weighted_matches += feature_weight
                                    
                                    weighted_total += feature_weight
                                    
                                    logger.debug(f"Feature '{feature_name}' match: {query_value == doc_value} (weight: {feature_weight})")
                        
                        # Calculate weighted categorical similarity
                        if weighted_total > 0:
                            cat_sim = weighted_matches / weighted_total
                            logger.info(f"Categorical similarity for {match.id}: {cat_sim:.4f} (weighted: {weighted_matches:.2f}/{weighted_total:.2f})")
                except Exception as e:
                    logger.warning(f"Error processing categorical features for document {match.id}: {e}")
            
            # Calculate combined score
            score = match.score  # Default to text similarity score
            
            # If we have a categorical similarity score, incorporate it
            if cat_sim is not None:
                # Combine scores with weights
                score = self.text_weight * match.score + self.categorical_weight * cat_sim
                logger.info(f"Combined score for {match.id}: {score:.4f} (text: {match.score:.4f}, cat: {cat_sim:.4f})")
                
                # Apply client preferences for specific features if available
                if client_preferences and match.metadata and 'categorical_features' in match.metadata:
                    preference_bonus = 0.0
                    feature_str = match.metadata['categorical_features']
                    
                    if isinstance(feature_str, str):
                        try:
                            doc_features = json.loads(feature_str)
                            
                            # Apply preferences for all client-weighted features
                            for feature_name, preference_weight in client_preferences.items():
                                if feature_name in doc_features:
                                    doc_values = doc_features[feature_name]
                                    
                                    # For multi-valued features, check if any preferred values are present
                                    if isinstance(doc_values, list) and len(doc_values) > 0:
                                        preference_bonus += preference_weight
                                        logger.info(f"Applied {feature_name} preference bonus: {preference_weight}")
                                    # For single-valued features, check exact match
                                    elif doc_values and preference_weight > 0:
                                        preference_bonus += preference_weight
                                        logger.info(f"Applied {feature_name} preference bonus for '{doc_values}': {preference_weight}")
                            
                            # Add preference bonus to score
                            if preference_bonus > 0:
                                score += preference_bonus
                                logger.info(f"Applied total preference bonus: {preference_bonus:.4f}")
                        except Exception as e:
                            logger.warning(f"Error applying client preferences: {e}")
            
            recommendations.append({
                'id': match.id,
                'score': score,
                'text_score': match.score,  # Include original text similarity score
                'categorical_score': cat_sim if 'cat_sim' in locals() else None,  # Include categorical score if calculated
                'metadata': match.metadata
            })
        
        # Apply temporal boosting if requested
        if temporal_boost is not None and temporal_boost > 0:
            logger.info(f"Applying temporal boosting with weight {temporal_boost}")
            self._apply_temporal_boosting(recommendations, temporal_boost, reference_date)
        
        # Sort by combined score and limit to top_k
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        recommendations = recommendations[:top_k]
            
        logger.info(f"Found {len(recommendations)} recommendations")
        return recommendations

    def get_recommendations(
        self,
        query_text: str,
        query_keywords: Optional[List[str]] = None,
        query_features: Optional[Dict[str, Union[str, List[str]]]] = None,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_categorical: bool = True,
        client_preferences: Optional[Dict[str, float]] = None,
        embedding_type: str = 'combined',
        temporal_boost: Optional[float] = None,
        reference_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Get document recommendations based on query.
        
        Args:
            query_text: Query text (e.g., document summary or user query)
            query_keywords: Optional list of keywords to enhance the query
            query_features: Optional dictionary of categorical features
            top_k: Number of recommendations to return
            filter: Optional Pinecone metadata filter
            include_categorical: Whether to include categorical features in scoring
            client_preferences: Optional dictionary mapping client-weighted features
                             (e.g., 'form', 'author') to their preference weights
            embedding_type: Type of embedding to use for the query ('combined', 'summary', or 'keyword')
            
        Returns:
            List of recommended documents with scores and metadata
            
        Raises:
            ValidationError: If input parameters are invalid
            EmbeddingError: If there's an error generating the embedding
            RecommendationError: If there's an error generating recommendations
        """
        # Validate inputs
        if not query_text or not query_text.strip():
            logger.error("Empty query text provided")
            raise ValidationError("Query text cannot be empty", field="query_text")
            
        if embedding_type not in ['combined', 'summary', 'keyword']:
            logger.warning(f"Invalid embedding_type '{embedding_type}', defaulting to 'combined'")
            embedding_type = 'combined'
            
        # Store client preferences if provided
        if client_preferences:
            logger.info(f"Using client preferences: {client_preferences}")
            self.client_preferences = client_preferences
            
        # Generate query embedding based on the specified embedding type
        try:
            if embedding_type == 'combined' and query_keywords:
                # Generate combined embedding with dynamic weighting
                query_embedding = self.embedder.combine_text_features(
                    summary=query_text,
                    keywords=query_keywords
                )
                logger.info(f"Using combined embedding with {len(query_keywords)} keywords")
            elif embedding_type == 'keyword' and query_keywords:
                # Generate keyword-only embedding
                keyword_text = ' '.join(query_keywords)
                query_embedding = self.embedder.generate_embeddings([keyword_text], show_progress=False)[0]
                logger.info(f"Using keyword-only embedding with {len(query_keywords)} keywords")
            else:
                # Default to summary/text-only embedding
                query_embedding = self.embedder.generate_embeddings([query_text], show_progress=False)[0]
                logger.info(f"Using {'summary' if embedding_type == 'summary' else 'text-only'} embedding")
                
            logger.debug(f"Generated embedding with shape {query_embedding.shape}")
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}")
        
        # Use the get_recommendations_with_embedding method to get recommendations
        try:
            return self.get_recommendations_with_embedding(
                query_embedding=query_embedding,
                query_text=query_text,
                query_keywords=query_keywords,
                query_features=query_features,
                top_k=top_k,
                filter=filter,
                include_categorical=include_categorical,
                client_preferences=client_preferences,
                embedding_type=embedding_type,
                temporal_boost=temporal_boost,
                reference_date=reference_date
            )
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            if isinstance(e, (PineconeError, ValidationError, EmbeddingError, RecommendationError)):
                raise
            else:
                raise RecommendationError(f"Unexpected error: {str(e)}", query=query_text[:50])
    
    def _apply_temporal_boosting(self, recommendations, temporal_boost, reference_date=None):
        """
        Apply temporal boosting to a list of recommendations.
        
        Args:
            recommendations: List of recommendation dictionaries to boost
            temporal_boost: Weight for temporal boosting (0.0-1.0)
            reference_date: Optional reference date for temporal calculations in format YYYY-MM-DD
        """
        # Store original similarity scores
        for rec in recommendations:
            rec['original_similarity'] = rec['score']
        
        # Parse reference date or use current date
        ref_date = None
        if reference_date:
            try:
                ref_date = datetime.strptime(reference_date, '%Y-%m-%d').date()
                logger.info(f"Using reference date: {ref_date}")
            except ValueError:
                logger.warning(f"Invalid reference date format: {reference_date}, using current date")
                ref_date = datetime.now().date()
        else:
            ref_date = datetime.now().date()
            logger.info(f"Using current date as reference: {ref_date}")
        
        # Apply temporal boosting to each recommendation
        for rec in recommendations:
            # Extract document date from the consolidated database
            doc_date = None
            doc_id = rec['id']
            
            try:
                # Fetch document info from the database using the document ID (celex_number)
                doc_info = database_utils.get_document_by_celex(doc_id)
                
                # Check if we got a valid document and it has a date
                if doc_info and 'date_of_document' in doc_info and doc_info['date_of_document']:
                    try:
                        # Parse the date string from the database
                        date_str = doc_info['date_of_document']
                        # Handle different date formats
                        for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%Y/%m/%d']:
                            try:
                                doc_date = datetime.strptime(date_str, fmt).date()
                                break
                            except ValueError:
                                continue
                    except Exception as e:
                        logger.warning(f"Error parsing date {date_str}: {e}")
            except Exception as e:
                logger.warning(f"Error fetching document {doc_id} from database: {e}")
                
            # Fallback to metadata if database lookup failed
            if not doc_date and 'metadata' in rec and rec['metadata']:
                # Try different date fields that might be present in metadata
                date_fields = ['date', 'publication_date', 'document_date']
                for field in date_fields:
                    if field in rec['metadata'] and rec['metadata'][field]:
                        try:
                            # Try to parse the date string
                            date_str = rec['metadata'][field]
                            # Handle different date formats
                            for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%Y/%m/%d']:
                                try:
                                    doc_date = datetime.strptime(date_str, fmt).date()
                                    break
                                except ValueError:
                                    continue
                            if doc_date:
                                break
                        except Exception as e:
                            logger.warning(f"Error parsing date {rec['metadata'][field]}: {e}")
            
            # Apply temporal boost if we have a valid document date
            if doc_date:
                # Calculate days difference
                days_diff = (ref_date - doc_date).days
                
                # Apply non-linear decay function (more recent documents get higher boost)
                if days_diff >= 0:  # Only boost documents from the past
                    # Calculate temporal score (1.0 for most recent, decreasing for older)
                    # Use exponential decay: e^(-days_diff/decay_factor)
                    import math
                    decay_factor = 365 * 2  # 2 years half-life
                    temporal_score = math.exp(-days_diff / decay_factor)
                    
                    # Store temporal score for reference
                    rec['temporal_score'] = temporal_score
                    
                    # Apply weighted combination of original similarity and temporal score
                    rec['score'] = (1 - temporal_boost) * rec['original_similarity'] + temporal_boost * temporal_score
                    
                    logger.debug(f"Document {rec['id']} date: {doc_date}, temporal score: {temporal_score:.4f}, new score: {rec['score']:.4f}")
                else:
                    # Future documents don't get a boost
                    rec['temporal_score'] = 0.0
                    logger.debug(f"Document {rec['id']} has future date {doc_date}, no temporal boost applied")
            else:
                # No date available, can't apply temporal boost
                rec['temporal_score'] = 0.0
                logger.warning(f"No valid date found for document {rec['id']}, temporal boost not applied")
    
    def get_recommendations_by_id(
        self,
        document_id: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_categorical: bool = True,
        client_preferences: Optional[Dict[str, float]] = None,
        temporal_boost: Optional[float] = None,
        reference_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Get similar documents based on an existing document ID.
        
        Args:
            document_id: ID of the reference document
            top_k: Number of recommendations to return
            filter: Optional Pinecone metadata filter
            include_categorical: Whether to include categorical features in scoring
            client_preferences: Optional dictionary mapping client-weighted features
            temporal_boost: Optional weight for temporal boosting (0.0-1.0) to favor more recent documents
            reference_date: Optional reference date for temporal calculations in format YYYY-MM-DD
            
        Returns:
            List of recommended documents with scores and metadata
            
        Raises:
            ValidationError: If document_id is invalid or not found
            PineconeError: If there's an error querying Pinecone
            RecommendationError: If there's an error generating recommendations
        """
        # Validate input
        if not document_id or not document_id.strip():
            logger.error("Empty document ID provided")
            raise ValidationError("Document ID cannot be empty", field="document_id")
            
        logger.info(f"Finding documents similar to ID: {document_id}")
        
        try:
            # Get the document vector from Pinecone
            fetch_response = self.index.fetch([document_id])
            
            if document_id not in fetch_response.vectors:
                logger.error(f"Document ID '{document_id}' not found in Pinecone index")
                raise ValidationError(f"Document ID '{document_id}' not found in Pinecone index", 
                                     field="document_id", 
                                     code="DOC_NOT_FOUND")
                
            # Get the vector
            vector = fetch_response.vectors[document_id].values
            
            # Get the document metadata to extract categorical features
            doc_metadata = fetch_response.vectors[document_id].metadata
            logger.debug(f"Retrieved document metadata: {list(doc_metadata.keys()) if doc_metadata else 'None'}")
            
        except pinecone.core.client.exceptions.ApiException as e:
            logger.error(f"Pinecone API error while fetching document: {str(e)}")
            raise PineconeError(f"Error fetching document: {str(e)}", code="FETCH_ERROR")
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching document: {str(e)}")
            raise RecommendationError(f"Failed to fetch document: {str(e)}")
        
        # Store client preferences if provided
        if client_preferences:
            logger.info(f"Using client preferences: {client_preferences}")
            self.client_preferences = client_preferences
        else:
            self.client_preferences = None
        
        # Extract categorical features from the document if available
        if include_categorical and doc_metadata and 'categorical_features' in doc_metadata:
            try:
                feature_str = doc_metadata['categorical_features']
                if isinstance(feature_str, str):
                    self._query_categorical_features = json.loads(feature_str)
                    logger.info(f"Using categorical features from document: {list(self._query_categorical_features.keys())}")
                else:
                    logger.warning("Categorical features not in expected string format")
                    self._query_categorical_features = None
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing categorical features JSON: {e}")
                self._query_categorical_features = None
            except Exception as e:
                logger.warning(f"Error extracting categorical features from document: {e}")
                self._query_categorical_features = None
        else:
            logger.debug("No categorical features available or feature processing disabled")
            self._query_categorical_features = None
            
        # Query Pinecone with the vector
        try:
            results = self.index.query(
                vector=vector,
                top_k=top_k * 2 if self._query_categorical_features is not None else top_k + 1,  # Get more results for reranking
                include_metadata=True,
                filter=filter
            )
            logger.debug(f"Received {len(results.matches)} matches from Pinecone")
            
        except pinecone.core.client.exceptions.ApiException as e:
            logger.error(f"Pinecone query error: {str(e)}")
            raise PineconeError(f"Query error: {str(e)}", code="QUERY_ERROR")
        except Exception as e:
            logger.error(f"Unexpected error querying Pinecone: {str(e)}")
            raise RecommendationError(f"Failed to query vector database: {str(e)}")
        
        # Process results, excluding the query document itself
        recommendations = []
        for match in results.matches:
            if match.id != document_id:  # Exclude the query document
                logger.debug(f"Processing match {match.id} with score {match.score}")
                # Extract the document's categorical features from metadata if available
                doc_categorical_features = None
                cat_sim = None
                
                if self._query_categorical_features is not None and match.metadata and 'categorical_features' in match.metadata:
                    try:
                        # Parse the JSON string of categorical features
                        feature_str = match.metadata['categorical_features']
                        if isinstance(feature_str, str):
                            doc_features_dict = json.loads(feature_str)
                            
                            # Use weighted feature matching for better similarity calculation
                            weighted_matches = 0.0
                            weighted_total = 0.0
                            
                            # Process each feature type with its appropriate weight
                            for feature_name, query_values in self._query_categorical_features.items():
                                if feature_name in doc_features_dict and feature_name in self.feature_processor.feature_weights:
                                    doc_values = doc_features_dict[feature_name]
                                    feature_weight = self.feature_processor.feature_weights[feature_name]
                                    
                                    # Handle different feature types appropriately
                                    if feature_name in self.feature_processor.multi_valued_features:
                                        # For multi-valued features, calculate Jaccard similarity
                                        query_set = set(query_values)
                                        doc_set = set(doc_values)
                                        
                                        if len(query_set) > 0:
                                            # Calculate Jaccard similarity: intersection / union
                                            intersection = len(query_set.intersection(doc_set))
                                            union = len(query_set.union(doc_set))
                                            similarity = intersection / union if union > 0 else 0
                                            
                                            weighted_matches += similarity * feature_weight
                                            weighted_total += feature_weight
                                    else:
                                        # For single-valued features, exact match
                                        query_value = query_values[0] if isinstance(query_values, list) else query_values
                                        doc_value = doc_values[0] if isinstance(doc_values, list) else doc_values
                                        
                                        if query_value == doc_value:
                                            weighted_matches += feature_weight
                                        
                                        weighted_total += feature_weight
                            
                            # Calculate weighted categorical similarity
                            if weighted_total > 0:
                                cat_sim = weighted_matches / weighted_total
                    except Exception as e:
                        logger.warning(f"Error processing categorical features for document {match.id}: {e}")
                
                # Calculate combined score
                score = match.score  # Default to text similarity score
                
                # If we have a categorical similarity score, incorporate it
                if cat_sim is not None:
                    # Combine scores with weights
                    score = self.text_weight * match.score + self.categorical_weight * cat_sim
                    
                    # Apply client preferences if available
                    if self.client_preferences and match.metadata and 'categorical_features' in match.metadata:
                        preference_bonus = 0.0
                        feature_str = match.metadata['categorical_features']
                        
                        if isinstance(feature_str, str):
                            try:
                                doc_features = json.loads(feature_str)
                                
                                # Apply preferences for all client-weighted features
                                for feature_name, preference_values in self.client_preferences.items():
                                    if feature_name in doc_features:
                                        doc_values = doc_features[feature_name]
                                        
                                        # Handle nested preference dictionaries (e.g., document_type: {regulation: 0.8})
                                        if isinstance(preference_values, dict):
                                            # For multi-valued features
                                            if isinstance(doc_values, list):
                                                for doc_value in doc_values:
                                                    if doc_value in preference_values:
                                                        preference_bonus += preference_values[doc_value]
                                            # For single-valued features
                                            elif doc_values in preference_values:
                                                preference_bonus += preference_values[doc_values]
                                        # Handle direct weight values
                                        elif isinstance(preference_values, (int, float)):
                                            # For multi-valued features, check if any preferred values are present
                                            if isinstance(doc_values, list) and len(doc_values) > 0:
                                                preference_bonus += preference_values
                                            # For single-valued features, check exact match
                                            elif doc_values and preference_values > 0:
                                                preference_bonus += preference_values
                                
                                # Add preference bonus to score
                                if preference_bonus > 0:
                                    score += preference_bonus
                            except Exception as e:
                                logger.warning(f"Error applying client preferences: {e}")
                
                recommendations.append({
                    'id': match.id,
                    'score': score,
                    'text_score': match.score,
                    'categorical_score': cat_sim,
                    'metadata': match.metadata
                })
            
        # Apply temporal boosting if requested
        if temporal_boost is not None and temporal_boost > 0:
            logger.info(f"Applying temporal boosting with weight {temporal_boost}")
            self._apply_temporal_boosting(recommendations, temporal_boost, reference_date)
            
            # Sort by combined score
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            recommendations = recommendations[:top_k]
        
        logger.info(f"Found {len(recommendations)} similar documents")
        return recommendations
