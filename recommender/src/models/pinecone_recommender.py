"""
Recommender system using Pinecone vector database for legal document recommendations.
"""
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import logging
from pathlib import Path
import json
from datetime import datetime
import pinecone

from .embeddings import BERTEmbedder
from .features import FeatureProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        """
        if text_weight + categorical_weight != 1.0:
            raise ValueError("Weights must sum to 1.0")
            
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
        """Initialize connection to Pinecone."""
        logger.info(f"Connecting to Pinecone index: {self.index_name}")
        # Initialize Pinecone with API key
        self.pc = pinecone.Pinecone(api_key=self.api_key)
        
        # Check if index exists
        if self.index_name not in self.pc.list_indexes().names():
            raise ValueError(f"Pinecone index '{self.index_name}' does not exist")
            
        # Connect to index
        self.index = self.pc.Index(self.index_name)
        
        # Get index stats
        stats = self.index.describe_index_stats()
        logger.info(f"Connected to Pinecone index with {stats.total_vector_count} vectors")
        logger.info(f"Vector dimension: {stats.dimension}")
        
    def get_recommendations(
        self,
        query_text: str,
        query_keywords: Optional[List[str]] = None,
        query_features: Optional[Dict[str, Union[str, List[str]]]] = None,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_categorical: bool = True,
        client_preferences: Optional[Dict[str, float]] = None,
        embedding_type: str = 'combined'
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
        """
        # Generate query embedding
        if query_keywords:
            # If keywords are provided, combine them with the query text
            combined_text = query_text + " " + " ".join(query_keywords)
        else:
            combined_text = query_text
            
        # Generate embedding for the query text based on the specified embedding type
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
        
        # Process categorical features if provided
        if include_categorical and query_features:
            logger.info(f"Processing categorical features for query: {query_features}")
            try:
                # Store the categorical features directly as a dictionary
                # We'll use this for matching against document categorical features
                self._query_categorical_features = query_features
                logger.info(f"Query categorical features: {self._query_categorical_features}")
            except Exception as e:
                logger.warning(f"Error processing categorical features: {e}")
                self._query_categorical_features = None
        else:
            self._query_categorical_features = None
        
        # Prepare filter to match the embedding type if specified
        if embedding_type in ['summary', 'keyword'] and filter is None:
            # Only filter by embedding_type if no other filter is specified
            filter = {'embedding_type': embedding_type}
        elif embedding_type in ['summary', 'keyword'] and isinstance(filter, dict):
            # Add embedding_type to existing filter
            filter = {**filter, 'embedding_type': embedding_type}
        
        # Query Pinecone
        logger.info(f"Querying Pinecone with embedding type '{embedding_type}': '{query_text[:50]}...'")
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k * 2 if self._query_categorical_features is not None else top_k,  # Get more results for reranking
            include_metadata=True,
            filter=filter
        )
        
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
                if self.client_preferences and match.metadata and 'categorical_features' in match.metadata:
                    preference_bonus = 0.0
                    feature_str = match.metadata['categorical_features']
                    
                    if isinstance(feature_str, str):
                        try:
                            doc_features = json.loads(feature_str)
                            
                            # Apply preferences for all client-weighted features
                            for feature_name, preference_weight in self.client_preferences.items():
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
        
        # Sort by combined score and limit to top_k
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        recommendations = recommendations[:top_k]
            
        logger.info(f"Found {len(recommendations)} recommendations")
        return recommendations
    
    def get_recommendations_by_id(
        self,
        document_id: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_categorical: bool = True,
        client_preferences: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """
        Get similar documents based on an existing document ID.
        
        Args:
            document_id: ID of the reference document
            top_k: Number of recommendations to return
            filter: Optional Pinecone metadata filter
            
        Returns:
            List of recommended documents with scores and metadata
        """
        logger.info(f"Finding documents similar to ID: {document_id}")
        
        # Get the document vector from Pinecone
        fetch_response = self.index.fetch([document_id])
        
        if document_id not in fetch_response.vectors:
            raise ValueError(f"Document ID '{document_id}' not found in Pinecone index")
            
        # Get the vector
        vector = fetch_response.vectors[document_id].values
        
        # Get the document metadata to extract categorical features
        doc_metadata = fetch_response.vectors[document_id].metadata
        
        # Store client preferences if provided
        self.client_preferences = client_preferences
        
        # Extract categorical features from the document if available
        if include_categorical and doc_metadata and 'categorical_features' in doc_metadata:
            try:
                feature_str = doc_metadata['categorical_features']
                if isinstance(feature_str, str):
                    self._query_categorical_features = json.loads(feature_str)
                    logger.info(f"Using categorical features from document: {list(self._query_categorical_features.keys())}")
            except Exception as e:
                logger.warning(f"Error extracting categorical features from document: {e}")
                self._query_categorical_features = None
        else:
            self._query_categorical_features = None
            
        # Query Pinecone with the vector
        results = self.index.query(
            vector=vector,
            top_k=top_k * 2 if self._query_categorical_features is not None else top_k + 1,  # Get more results for reranking
            include_metadata=True,
            filter=filter
        )
        
        # Process results, excluding the query document itself
        recommendations = []
        for match in results.matches:
            if match.id != document_id:  # Exclude the query document
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
                                for feature_name, preference_weight in self.client_preferences.items():
                                    if feature_name in doc_features:
                                        doc_values = doc_features[feature_name]
                                        
                                        # For multi-valued features, check if any preferred values are present
                                        if isinstance(doc_values, list) and len(doc_values) > 0:
                                            preference_bonus += preference_weight
                                        # For single-valued features, check exact match
                                        elif doc_values and preference_weight > 0:
                                            preference_bonus += preference_weight
                                
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
            
        logger.info(f"Found {len(recommendations)} similar documents")
        return recommendations
