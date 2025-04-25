"""
Custom exceptions for the EU Legal Recommender system.

This module defines custom exceptions used throughout the recommender system
to provide consistent error handling and informative error messages.
"""

from typing import Optional


class RecommenderError(Exception):
    """Base exception class for all recommender system errors."""
    
    def __init__(self, message: str, code: Optional[str] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            code: Optional error code for categorization
        """
        self.message = message
        self.code = code
        super().__init__(self.message)


class ConfigurationError(RecommenderError):
    """Exception raised for errors in the configuration."""
    
    def __init__(self, message: str, code: Optional[str] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            code: Optional error code for categorization
        """
        super().__init__(f"Configuration error: {message}", code or "CONFIG_ERROR")


class ConnectionError(RecommenderError):
    """Exception raised for errors in connecting to external services."""
    
    def __init__(self, service: str, message: str, code: Optional[str] = None):
        """
        Initialize the exception.
        
        Args:
            service: Name of the service that failed to connect
            message: Error message
            code: Optional error code for categorization
        """
        super().__init__(f"Connection error to {service}: {message}", code or "CONNECTION_ERROR")


class PineconeError(ConnectionError):
    """Exception raised for errors in Pinecone operations."""
    
    def __init__(self, message: str, code: Optional[str] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            code: Optional error code for categorization
        """
        super().__init__("Pinecone", message, code or "PINECONE_ERROR")


class EmbeddingError(RecommenderError):
    """Exception raised for errors in embedding generation."""
    
    def __init__(self, message: str, code: Optional[str] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            code: Optional error code for categorization
        """
        super().__init__(f"Embedding error: {message}", code or "EMBEDDING_ERROR")


class ValidationError(RecommenderError):
    """Exception raised for validation errors in input data."""
    
    def __init__(self, message: str, field: Optional[str] = None, code: Optional[str] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            field: Optional field name that failed validation
            code: Optional error code for categorization
        """
        field_info = f" (field: {field})" if field else ""
        super().__init__(f"Validation error{field_info}: {message}", code or "VALIDATION_ERROR")


class DataError(RecommenderError):
    """Exception raised for errors in data processing."""
    
    def __init__(self, message: str, code: Optional[str] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            code: Optional error code for categorization
        """
        super().__init__(f"Data error: {message}", code or "DATA_ERROR")


class ProcessingError(RecommenderError):
    """Exception raised for errors during processing operations."""
    
    def __init__(self, message: str, code: Optional[str] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            code: Optional error code for categorization
        """
        super().__init__(f"Processing error: {message}", code or "PROCESSING_ERROR")


class ProfileError(RecommenderError):
    """Exception raised for errors in user profile operations."""
    
    def __init__(self, message: str, profile_id: Optional[str] = None, code: Optional[str] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            profile_id: Optional ID of the profile that caused the error
            code: Optional error code for categorization
        """
        profile_info = f" (profile: {profile_id})" if profile_id else ""
        super().__init__(f"Profile error{profile_info}: {message}", code or "PROFILE_ERROR")


class RecommendationError(RecommenderError):
    """Exception raised for errors in generating recommendations."""
    
    def __init__(self, message: str, query: Optional[str] = None, code: Optional[str] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            query: Optional query that caused the error
            code: Optional error code for categorization
        """
        query_info = f" for query '{query}'" if query else ""
        super().__init__(f"Recommendation error{query_info}: {message}", code or "RECOMMENDATION_ERROR")
