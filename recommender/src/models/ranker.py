"""
Rank documents based on combined similarity to query profiles.

This module provides the DocumentRanker class for ranking documents based on their
similarity to user query profiles. It combines text embeddings and categorical features
to compute similarity scores and return the most relevant documents.
"""
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
from pathlib import Path
import json
from datetime import datetime, date

# Import components
from .embeddings import BERTEmbedder
from .features import FeatureProcessor
from .similarity import SimilarityComputer

# Import utilities
from ..utils.logging import get_logger
from ..utils.exceptions import ValidationError, ProcessingError, ConfigurationError, EmbeddingError

# Set up logger for this module
logger = get_logger(__name__)

class DocumentRanker:
    """Rank documents based on similarity to query profiles.
    
    This class provides functionality for ranking documents based on their similarity
    to user query profiles. It processes documents into text embeddings and categorical
    features, builds a similarity index, and ranks documents based on combined similarity.
    
    Attributes:
        embedder (BERTEmbedder): Instance for generating text embeddings
        feature_processor (FeatureProcessor): Instance for processing categorical features
        similarity_computer (SimilarityComputer): Instance for computing similarities
        cache_dir (Path, optional): Directory for caching document vectors
        document_vectors (Dict): Cache of processed document vectors
    """
    
    def __init__(self,
                 embedder: BERTEmbedder,
                 feature_processor: FeatureProcessor,
                 similarity_computer: SimilarityComputer,
                 cache_dir: Optional[Union[str, Path]] = None,
                 invalidate_cache_on_weight_change: bool = True):
        """
        Initialize document ranker.
        
        Args:
            embedder (BERTEmbedder): Instance for generating text embeddings
            feature_processor (FeatureProcessor): Instance for processing categorical features
            similarity_computer (SimilarityComputer): Instance for computing similarities
            cache_dir (Optional[Union[str, Path]], optional): Directory for caching document vectors.
                Defaults to None.
                
        Raises:
            ValidationError: If any of the required components are None
            ConfigurationError: If there's an issue with the configuration
        """
        # Validate inputs
        if embedder is None:
            logger.error("embedder cannot be None")
            raise ValidationError("embedder cannot be None")
            
        if feature_processor is None:
            logger.error("feature_processor cannot be None")
            raise ValidationError("feature_processor cannot be None")
            
        if similarity_computer is None:
            logger.error("similarity_computer cannot be None")
            raise ValidationError("similarity_computer cannot be None")
        
        self.embedder = embedder
        self.feature_processor = feature_processor
        self.similarity_computer = similarity_computer
        
        # Set up cache directory if provided
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Cache directory set to {self.cache_dir}")
            except Exception as e:
                logger.error(f"Failed to create cache directory: {str(e)}")
                raise ConfigurationError(f"Failed to create cache directory: {str(e)}")
        else:
            self.cache_dir = None
            logger.debug("No cache directory specified")
        
        # Initialize document cache
        self.document_vectors = {}
        
        # Flag to control cache invalidation on weight changes
        self.invalidate_cache_on_weight_change = invalidate_cache_on_weight_change
        
        # Store initial weights for change detection
        self._initial_weights = self._get_current_weights()
        
        logger.info(f"Initialized DocumentRanker with embedder={type(embedder).__name__}, "
                   f"feature_processor={type(feature_processor).__name__}, "
                   f"similarity_computer={type(similarity_computer).__name__}, "
                   f"invalidate_cache_on_weight_change={invalidate_cache_on_weight_change}")
        
    def _get_current_weights(self) -> Dict[str, Any]:
        """
        Get current weights from all components.
        
        This internal method collects the current weights from all components
        (embedder, feature processor, similarity computer) to track changes.
        
        Returns:
            Dict[str, Any]: Dictionary of current weights
        """
        weights = {}
        
        # Get similarity weights
        if hasattr(self.similarity_computer, "get_weights"):
            weights["similarity"] = self.similarity_computer.get_weights()
        else:
            weights["similarity"] = {
                "text_weight": getattr(self.similarity_computer, "text_weight", 0.7),
                "categorical_weight": getattr(self.similarity_computer, "categorical_weight", 0.3)
            }
        
        # Get embedder weights
        if hasattr(self.embedder, "get_weights"):
            weights["embedder"] = self.embedder.get_weights()
        else:
            weights["embedder"] = {
                "summary_weight": getattr(self.embedder, "summary_weight", 0.6),
                "keywords_weight": getattr(self.embedder, "keywords_weight", 0.4)
            }
        
        # Get feature processor weights
        if hasattr(self.feature_processor, "get_weights"):
            weights["features"] = self.feature_processor.get_weights()
            
        return weights
    
    def _check_weights_changed(self) -> bool:
        """
        Check if weights have changed since initialization.
        
        Returns:
            bool: True if weights have changed, False otherwise
        """
        current_weights = self._get_current_weights()
        
        # Compare with initial weights
        for component, weight_dict in current_weights.items():
            if component not in self._initial_weights:
                return True
                
            for weight_name, weight_value in weight_dict.items():
                if weight_name not in self._initial_weights[component]:
                    return True
                    
                if weight_value != self._initial_weights[component][weight_name]:
                    return True
                    
        return False
    
    def update_weights(self) -> None:
        """
        Update internal weight tracking and handle cache invalidation.
        
        This method should be called after weights have been changed in any component
        to ensure proper cache invalidation and index rebuilding.
        """
        # Check if weights have changed
        if self._check_weights_changed():
            logger.info("Weights have changed, updating internal state")
            
            # Update stored weights
            self._initial_weights = self._get_current_weights()
            
            # Invalidate cache if needed
            if self.invalidate_cache_on_weight_change:
                logger.info("Invalidating document vector cache due to weight changes")
                self.document_vectors = {}
                
            # Invalidate similarity index
            if hasattr(self.similarity_computer, "index"):
                logger.info("Invalidating similarity index due to weight changes")
                self.similarity_computer.index = None
                
            logger.debug(f"Updated weights: {self._initial_weights}")
        else:
            logger.debug("No weight changes detected")
    
    def set_similarity_weights(self, text_weight: float, categorical_weight: float) -> None:
        """
        Set weights for similarity computation.
        
        Args:
            text_weight: Weight for text similarity (0-1)
            categorical_weight: Weight for categorical similarity (0-1)
            
        Raises:
            ConfigurationError: If weights are invalid
        """
        # Update weights in similarity computer
        self.similarity_computer.set_weights(text_weight, categorical_weight)
        
        # Update internal state
        self.update_weights()
        
        logger.info(f"Set similarity weights: text={text_weight:.2f}, categorical={categorical_weight:.2f}")
    
    def set_categorical_feature_weights(self, feature_weights: Dict[str, float]) -> None:
        """
        Set weights for individual categorical features.
        
        Args:
            feature_weights: Dictionary mapping feature names to weights
            
        Raises:
            ConfigurationError: If weights are invalid
        """
        # Update weights in similarity computer
        self.similarity_computer.set_categorical_feature_weights(feature_weights)
        
        # Update internal state
        self.update_weights()
        
        logger.info(f"Set categorical feature weights: {feature_weights}")
    
    def process_document(self,
                        doc_id: str,
                        summary: str,
                        keywords: List[str],
                        features: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single document into text and categorical vectors.
        
        This method converts a document's textual content and categorical features into
        vector representations. It first generates a text embedding by combining the summary
        and keywords with dynamic weighting, then encodes the categorical features.
        Results are cached for future use.
        
        Args:
            doc_id (str): Unique document identifier
            summary (str): Document summary text
            keywords (List[str]): List of document keywords
            features (Dict[str, str]): Dictionary of categorical features
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - text_embedding: Vector representation of text content
                - categorical_features: Vector representation of categorical features
                
        Raises:
            ValidationError: If any input parameters are invalid
            ProcessingError: If there's an error processing the document
        """
        try:
            # Validate inputs
            if not doc_id or not isinstance(doc_id, str):
                logger.error(f"Invalid doc_id: {doc_id}")
                raise ValidationError("doc_id must be a non-empty string")
                
            if not summary or not isinstance(summary, str):
                logger.error(f"Invalid summary for document {doc_id}")
                raise ValidationError("summary must be a non-empty string")
                
            if keywords is None or not isinstance(keywords, list):
                logger.error(f"Invalid keywords for document {doc_id}")
                raise ValidationError("keywords must be a list")
                
            if features is None or not isinstance(features, dict):
                logger.error(f"Invalid features for document {doc_id}")
                raise ValidationError("features must be a dictionary")
            
            # Check cache first
            if doc_id in self.document_vectors:
                logger.debug(f"Retrieved document {doc_id} from cache")
                return self.document_vectors[doc_id]
                
            logger.info(f"Processing document {doc_id}")
            
            # Generate text embedding with dynamic weighting based on number of keywords
            logger.debug(f"Generating text embedding for document {doc_id} with {len(keywords)} keywords")
            text_embedding = self.embedder.combine_text_features(summary=summary, keywords=keywords)
            
            # Process categorical features
            logger.debug(f"Encoding categorical features for document {doc_id}")
            categorical_features = self.feature_processor.encode_features(features)
            
            # Cache results - store as a dictionary with embeddings and original features
            doc_vector = {
                'text_embedding': text_embedding,
                'categorical_features': categorical_features,
                'features': features  # Store original features for temporal boosting
            }
            self.document_vectors[doc_id] = doc_vector
            logger.debug(f"Cached vectors for document {doc_id}")
            
            return doc_vector
            
        except (ValidationError, EmbeddingError) as e:
            # Re-raise these exceptions with context
            logger.error(f"Error processing document {doc_id}: {str(e)}")
            raise
        except Exception as e:
            error_msg = f"Unexpected error processing document {doc_id}: {str(e)}"
            logger.error(error_msg)
            raise ProcessingError(error_msg)
        
    def process_documents(self,
                         documents: List[Dict]) -> None:
        """
        Process and index a batch of documents.
        
        This method processes multiple documents in batch, converting each document's
        text content and categorical features into vector representations. It then
        builds a similarity index for efficient similarity search.
        
        Args:
            documents (List[Dict]): List of document dictionaries, each containing:
                - id (str): Document identifier
                - summary (str): Document summary
                - keywords (List[str]): List of keywords
                - features (Dict[str, str]): Dictionary of categorical features
                
        Raises:
            ValidationError: If documents is None or empty, or if any document is invalid
            ProcessingError: If there's an error processing the documents
        """
        try:
            # Validate input
            if documents is None:
                logger.error("documents cannot be None")
                raise ValidationError("documents cannot be None")
                
            if not documents:
                logger.warning("Empty documents list provided, nothing to process")
                return
                
            logger.info(f"Processing batch of {len(documents)} documents")
            
            text_embeddings = []
            categorical_features = []
            doc_ids = []
            
            # Process each document
            for i, doc in enumerate(documents):
                try:
                    # Validate document structure
                    required_fields = ['id', 'summary', 'keywords', 'features']
                    missing_fields = [field for field in required_fields if field not in doc]
                    
                    if missing_fields:
                        logger.error(f"Document at index {i} is missing required fields: {missing_fields}")
                        raise ValidationError(f"Document is missing required fields: {missing_fields}")
                    
                    # Process document
                    doc_vector = self.process_document(
                        doc['id'],
                        doc['summary'],
                        doc['keywords'],
                        doc['features']
                    )
                    
                    # Extract embeddings from the dictionary format
                    text_emb = doc_vector['text_embedding']
                    cat_feat = doc_vector['categorical_features']
                    
                    text_embeddings.append(text_emb)
                    categorical_features.append(cat_feat)
                    doc_ids.append(doc['id'])
                    
                except Exception as e:
                    logger.error(f"Error processing document at index {i}: {str(e)}")
                    raise ValidationError(f"Error processing document at index {i}: {str(e)}")
            
            # Convert to arrays
            logger.debug(f"Converting {len(text_embeddings)} document vectors to arrays")
            text_embeddings = np.array(text_embeddings)
            categorical_features = np.array(categorical_features)
            
            # Build similarity index
            logger.info(f"Building similarity index with {len(text_embeddings)} documents")
            self.similarity_computer.build_index(text_embeddings, categorical_features)
            logger.info(f"Successfully processed and indexed {len(doc_ids)} documents")
            
        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            error_msg = f"Unexpected error processing documents: {str(e)}"
            logger.error(error_msg)
            raise ProcessingError(error_msg)
        
    def get_weights(self) -> Dict[str, Any]:
        """
        Get current weights from all components.
        
        Returns:
            Dict[str, Any]: Dictionary of current weights
        """
        return self._get_current_weights()
    
    def rank_documents(self,
                      query_profile: Dict,
                      top_k: int = 10,
                      min_similarity: float = 0.0,
                      feature_weights: Optional[Dict[str, float]] = None,
                       temporal_boost: Optional[float] = None,
                       reference_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Rank documents based on similarity to query profile.
        
        This method ranks documents based on their similarity to a user query profile.
        It processes the query profile into text embeddings and categorical features,
        then uses the similarity computer to find the most similar documents.
        
        Args:
            query_profile (Dict): Dictionary containing:
                - interests (str): Text description of user interests
                - keywords (List[str]): List of interest keywords
                - features (Dict[str, str]): Dictionary of categorical preferences
            top_k (int, optional): Number of recommendations to return. Defaults to 10.
            min_similarity (float, optional): Minimum similarity threshold. Defaults to 0.0.
            feature_weights (Dict[str, float], optional): Temporary weights for categorical features.
            temporal_boost (float, optional): Weight for temporal boosting (0.0-1.0). If provided,
                more recent documents will be boosted in the ranking. Defaults to None (no boosting).
            reference_date (str, optional): Reference date for temporal calculations in format 'YYYY-MM-DD'.
                If None, current date is used. Defaults to None.
            
        Returns:
            List[Tuple[str, float]]: List of (document_id, similarity_score) tuples,
                sorted by similarity score in descending order.
                
        Raises:
            ValidationError: If query_profile is invalid or missing required fields
            ProcessingError: If there's an error during ranking
            ConfigurationError: If no documents have been processed yet
        """
        try:
            # Validate inputs
            if query_profile is None:
                logger.error("query_profile cannot be None")
                raise ValidationError("query_profile cannot be None")
                
            # Check required fields
            required_fields = ['interests', 'keywords', 'features']
            missing_fields = [field for field in required_fields if field not in query_profile]
            
            if missing_fields:
                logger.error(f"Query profile is missing required fields: {missing_fields}")
                raise ValidationError(f"Query profile is missing required fields: {missing_fields}")
                
            if top_k <= 0:
                logger.error(f"Invalid top_k value: {top_k}. Must be positive.")
                raise ValidationError(f"top_k must be positive, got {top_k}")
                
            if min_similarity < 0 or min_similarity > 1:
                logger.warning(f"Invalid min_similarity: {min_similarity}. Clamping to [0,1]")
                min_similarity = max(0, min(1, min_similarity))
                
            # Check if we have processed documents
            if not self.document_vectors:
                logger.error("No documents have been processed yet")
                raise ConfigurationError("No documents have been processed. Call process_documents first.")
                
            logger.info(f"Ranking documents for query profile with {len(query_profile['keywords'])} keywords")
            
            # Process query with dynamic weighting based on number of keywords
            logger.debug("Generating text embedding for query profile")
            query_text_emb = self.embedder.combine_text_features(
                summary=query_profile['interests'],
                keywords=query_profile['keywords']
            )
            
            logger.debug("Encoding categorical features for query profile")
            query_categorical = self.feature_processor.encode_features(
                query_profile['features']
            )
            
            # Apply feature weights if provided
            if feature_weights and hasattr(self.similarity_computer, "set_categorical_feature_weights"):
                logger.debug(f"Applying temporary feature weights for this query: {feature_weights}")
                # Store original weights to restore later
                original_weights = None
                if hasattr(self.similarity_computer, "get_categorical_feature_weights"):
                    original_weights = self.similarity_computer.get_categorical_feature_weights()
                
                # Apply temporary weights
                self.similarity_computer.set_categorical_feature_weights(feature_weights)
            
            # Find similar documents
            logger.debug(f"Finding top {top_k} similar documents with min_similarity={min_similarity}")
            indices, similarities = self.similarity_computer.find_similar(
                query_text_emb,
                query_categorical,
                k=top_k
            )
            
            # Restore original weights if we applied temporary ones
            if feature_weights and original_weights and hasattr(self.similarity_computer, "set_categorical_feature_weights"):
                logger.debug("Restoring original feature weights")
                self.similarity_computer.set_categorical_feature_weights(original_weights)
            
            # Filter by similarity threshold
            valid_idx = similarities >= min_similarity
            indices = indices[valid_idx]
            similarities = similarities[valid_idx]
            
            # Get document IDs
            doc_ids = list(self.document_vectors.keys())
            
            # Create detailed results with document IDs, similarity scores, and feature-level details
            results = []
            for idx, sim in zip(indices, similarities):
                doc_id = doc_ids[idx]
                
                # Get feature-level similarity details if available
                feature_details = {}
                if hasattr(self.similarity_computer, "get_last_similarity_details"):
                    feature_details = self.similarity_computer.get_last_similarity_details(idx)
                
                # Create detailed result
                result = {
                    "id": doc_id,
                    "similarity": float(sim),
                    "feature_details": feature_details
                }
                
                results.append(result)
            
            # Apply temporal boosting if specified
            if temporal_boost is not None and temporal_boost > 0:
                logger.info(f"Applying temporal boost with weight {temporal_boost}")
                results = self._apply_temporal_boost(results, temporal_boost, reference_date)
            
            logger.info(f"Found {len(results)} documents above similarity threshold {min_similarity}")
            for i, result in enumerate(results[:5]):
                logger.debug(f"Rank {i+1}: Document {result['id']} with similarity {result['similarity']:.4f}")
                
            return results
            
        except (ValidationError, ConfigurationError):
            # Re-raise these exceptions
            raise
        except Exception as e:
            error_msg = f"Unexpected error ranking documents: {str(e)}"
            logger.error(error_msg)
            raise ProcessingError(error_msg)
        
    def _apply_temporal_boost(self, results: List[Dict[str, Any]], boost_weight: float, reference_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Apply temporal boosting to re-rank results, giving preference to more recent documents.
        
        This method adjusts the similarity scores of documents based on their publication date,
        boosting more recent documents in the ranking. The final score is a weighted combination
        of the original similarity score and the temporal recency score.
        
        Args:
            results (List[Dict[str, Any]]): Original ranked results
            boost_weight (float): Weight for temporal boosting (0.0-1.0)
            reference_date (str, optional): Reference date for temporal calculations in format 'YYYY-MM-DD'.
                If None, current date is used.
                
        Returns:
            List[Dict[str, Any]]: Re-ranked results with temporal boosting applied
        """
        # Validate boost weight
        boost_weight = max(0.0, min(1.0, boost_weight))
        
        # Set reference date
        if reference_date:
            try:
                ref_date = datetime.strptime(reference_date, '%Y-%m-%d').date()
            except ValueError:
                logger.warning(f"Invalid reference_date format: {reference_date}. Using current date.")
                ref_date = date.today()
        else:
            ref_date = date.today()
            
        logger.debug(f"Using reference date {ref_date} for temporal boosting")
        
        # Define date format patterns to try
        date_formats = [
            '%Y-%m-%d',      # Standard ISO format: 2023-01-15
            '%Y/%m/%d',      # Alternative format: 2023/01/15
            '%d-%m-%Y',      # European format: 15-01-2023
            '%d/%m/%Y',      # Alternative European: 15/01/2023
            '%Y',            # Year only: 2023
            '%Y-%m',         # Year and month: 2023-01
            '%b %d, %Y',     # Month name format: Jan 15, 2023
            '%d %b %Y'       # European with month name: 15 Jan 2023
        ]
        
        # Maximum timeframe to consider (in days)
        # 10 years = 3650 days as default
        max_timeframe_days = 3650
        
        # Extract dates from documents and calculate temporal scores
        boosted_results = []
        for result in results:
            doc_id = result['id']
            doc_date = None
            date_str = None
            
            # Try to get document date from document vectors
            # First check if we have document vectors stored
            if hasattr(self, 'document_vectors') and isinstance(self.document_vectors, dict):
                if doc_id in self.document_vectors:
                    doc_vector = self.document_vectors[doc_id]
                    if isinstance(doc_vector, dict) and 'features' in doc_vector:
                        features = doc_vector['features']
                        if 'date' in features:
                            date_str = features['date']
            
            # Try to parse the date using multiple formats
            if date_str:
                # First handle the case of year only (simple integer check)
                if isinstance(date_str, str) and date_str.isdigit() and len(date_str) == 4:
                    try:
                        doc_date = date(int(date_str), 1, 1)
                    except ValueError:
                        pass
                else:
                    # Try all date formats
                    for fmt in date_formats:
                        try:
                            if isinstance(date_str, str):
                                doc_date = datetime.strptime(date_str, fmt).date()
                                break
                        except ValueError:
                            continue
            
            # Calculate temporal score
            temporal_score = 0.0
            days_diff = None
            
            if doc_date:
                # Calculate days difference
                if ref_date >= doc_date:  # Only boost past documents
                    days_diff = min(max_timeframe_days, (ref_date - doc_date).days)
                    # Apply a non-linear decay function to favor recent documents more strongly
                    # This uses a square root function to give more boost to recent documents
                    temporal_score = 1.0 - (days_diff / max_timeframe_days)**0.5
                else:
                    # Future documents get a neutral score
                    temporal_score = 0.5
                
                # Apply boosting formula: final_score = (1-w)*similarity + w*temporal_score
                original_score = result['similarity']
                boosted_score = ((1 - boost_weight) * original_score) + (boost_weight * temporal_score)
                
                # Create boosted result with explanation
                boosted_result = result.copy()
                boosted_result['original_similarity'] = original_score
                boosted_result['temporal_score'] = temporal_score
                boosted_result['similarity'] = boosted_score
                boosted_result['boosting_applied'] = True
                if days_diff is not None:
                    boosted_result['days_from_reference'] = days_diff
                
                boosted_results.append(boosted_result)
            else:
                # If no date available, keep original score but mark as not boosted
                result_copy = result.copy()
                result_copy['boosting_applied'] = False
                boosted_results.append(result_copy)
        
        # Re-sort by boosted similarity score
        boosted_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Log some statistics about the boosting
        boosted_count = sum(1 for r in boosted_results if r.get('boosting_applied', False))
        logger.info(f"Applied temporal boosting with weight {boost_weight} to {boosted_count}/{len(boosted_results)} documents")
        
        return boosted_results
    
    def save_state(self, directory: Union[str, Path]) -> None:
        """
        Save ranker state to directory.
        
        This method saves the current state of the DocumentRanker, including the similarity
        index and document vectors, to the specified directory. This allows for later
        restoration of the ranker state without reprocessing documents.
        
        Args:
            directory (Union[str, Path]): Directory to save state in. Will be created
                if it doesn't exist.
                
        Raises:
            ValidationError: If directory is None or empty
            ProcessingError: If there's an error saving the state
        """
        try:
            # Validate input
            if directory is None or (isinstance(directory, str) and not directory.strip()):
                logger.error("directory cannot be None or empty")
                raise ValidationError("directory cannot be None or empty")
                
            # Convert to Path object
            directory = Path(directory)
            
            # Create directory if it doesn't exist
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory {directory} for saving state")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {str(e)}")
                raise ProcessingError(f"Failed to create directory {directory}: {str(e)}")
            
            # Save similarity index
            logger.info(f"Saving similarity index to {directory / 'similarity_index.faiss'}")
            self.similarity_computer.save_index(directory / 'similarity_index.faiss')
            
            # Save document vectors
            logger.info(f"Saving {len(self.document_vectors)} document vectors to {directory / 'document_vectors.npy'}")
            np.save(directory / 'document_vectors.npy', self.document_vectors)
            
            # Save metadata
            metadata = {
                'saved_at': datetime.now().isoformat(),
                'document_count': len(self.document_vectors),
                'version': '1.0'
            }
            
            with open(directory / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Successfully saved ranker state to {directory}")
            
        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            error_msg = f"Unexpected error saving ranker state: {str(e)}"
            logger.error(error_msg)
            raise ProcessingError(error_msg)
        
    @classmethod
    def load_state(cls,
                  directory: Union[str, Path],
                  embedder: BERTEmbedder,
                  feature_processor: FeatureProcessor) -> 'DocumentRanker':
        """
        Load ranker state from directory.
        
        This class method loads a previously saved DocumentRanker state from the
        specified directory. It reconstructs the similarity index and document vectors,
        creating a new DocumentRanker instance with the restored state.
        
        Args:
            directory (Union[str, Path]): Directory containing saved state
            embedder (BERTEmbedder): Instance for generating text embeddings
            feature_processor (FeatureProcessor): Instance for processing categorical features
            
        Returns:
            DocumentRanker: Loaded DocumentRanker instance with restored state
            
        Raises:
            ValidationError: If directory or required components are None
            ProcessingError: If there's an error loading the state
            ConfigurationError: If the saved state is missing required files
        """
        try:
            # Validate inputs
            if directory is None or (isinstance(directory, str) and not directory.strip()):
                logger.error("directory cannot be None or empty")
                raise ValidationError("directory cannot be None or empty")
                
            if embedder is None:
                logger.error("embedder cannot be None")
                raise ValidationError("embedder cannot be None")
                
            if feature_processor is None:
                logger.error("feature_processor cannot be None")
                raise ValidationError("feature_processor cannot be None")
            
            # Convert to Path object
            directory = Path(directory)
            
            # Check if directory exists
            if not directory.exists() or not directory.is_dir():
                logger.error(f"Directory {directory} does not exist or is not a directory")
                raise ConfigurationError(f"Directory {directory} does not exist or is not a directory")
            
            # Check for required files
            required_files = ['similarity_index.faiss', 'document_vectors.npy']
            missing_files = [file for file in required_files if not (directory / file).exists()]
            
            if missing_files:
                logger.error(f"Missing required files in {directory}: {missing_files}")
                raise ConfigurationError(f"Missing required files in {directory}: {missing_files}")
            
            logger.info(f"Loading ranker state from {directory}")
            
            # Create similarity computer and load index
            logger.debug("Creating SimilarityComputer and loading index")
            similarity_computer = SimilarityComputer()
            try:
                similarity_computer.load_index(directory / 'similarity_index.faiss')
            except Exception as e:
                logger.error(f"Failed to load similarity index: {str(e)}")
                raise ProcessingError(f"Failed to load similarity index: {str(e)}")
            
            # Create ranker
            logger.debug("Creating DocumentRanker instance")
            ranker = cls(embedder, feature_processor, similarity_computer)
            
            # Load document vectors
            logger.debug("Loading document vectors")
            try:
                ranker.document_vectors = np.load(directory / 'document_vectors.npy',
                                                allow_pickle=True).item()
                logger.info(f"Loaded {len(ranker.document_vectors)} document vectors")
            except Exception as e:
                logger.error(f"Failed to load document vectors: {str(e)}")
                raise ProcessingError(f"Failed to load document vectors: {str(e)}")
            
            # Load metadata if available
            metadata_path = directory / 'metadata.json'
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    logger.info(f"Loaded ranker state from {metadata.get('saved_at', 'unknown date')}, "
                               f"containing {metadata.get('document_count', 'unknown')} documents")
                except Exception as e:
                    logger.warning(f"Failed to load metadata: {str(e)}")
            
            logger.info(f"Successfully loaded ranker state from {directory}")
            return ranker
            
        except (ValidationError, ConfigurationError):
            # Re-raise these exceptions
            raise
        except Exception as e:
            error_msg = f"Unexpected error loading ranker state: {str(e)}"
            logger.error(error_msg)
            raise ProcessingError(error_msg)
