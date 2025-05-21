#!/usr/bin/env python3
"""
EU Legal Recommender System - Command Line Interface

A unified CLI interface for the EU Legal Recommender System that provides
access to all major functionality through subcommands including document indexing,
embedding recreation, recommendation generation, and user profile management.

This module serves as the main entry point for command-line interaction with the
recommender system, allowing users to perform various operations without writing code.

Usage:
    python cli.py <command> [options]

Commands:
    index     - Index documents into Pinecone
    recreate  - Recreate embeddings for all documents
    recommend - Get document recommendations
    profile   - Manage user profiles
"""

import os
import sys
import json
import time
import argparse
import functools
from pathlib import Path
from typing import Callable, Any, Dict, List, Optional, Union
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

# Import configuration
from src.config import LOGS_DIR, PINECONE_API_KEY, PINECONE_ENVIRONMENT, EMBEDDER, PINECONE

# Import utilities
from src.utils.logging import get_logger
from src.utils.exceptions import (
    ValidationError, PineconeError, EmbeddingError, 
    ProfileError, RecommendationError, ConfigurationError as ConfigError
)

# Set up logging
LOGS_DIR.mkdir(exist_ok=True)
logger = get_logger(__name__)

def setup_index_command(subparsers) -> None:
    """
    Set up the 'index' command parser.
    
    Args:
        subparsers: The subparser collection to add the new parser to
    
    Returns:
        None
    """
    parser = subparsers.add_parser(
        'index',
        help='Index documents into Pinecone'
    )
    parser.add_argument(
        '--db-type',
        choices=['consolidated', 'legacy'],
        default='consolidated',
        help='Database structure type'
    )
    parser.add_argument(
        '--tiers',
        type=str,
        default='1,2,3,4',
        help='Comma-separated list of tiers to process'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Batch size for processing documents'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit the number of documents to process per tier'
    )
    parser.add_argument(
        '--recreate-index',
        action='store_true',
        help='Recreate the Pinecone index (WARNING: this will delete all existing data)'
    )
    parser.set_defaults(func=run_index_command)

def setup_recreate_command(subparsers) -> None:
    """
    Set up the 'recreate' command parser.
    
    Args:
        subparsers: The subparser collection to add the new parser to
    
    Returns:
        None
    """
    parser = subparsers.add_parser(
        'recreate',
        help='Recreate embeddings for all documents'
    )
    parser.add_argument(
        '--db-type',
        choices=['consolidated', 'legacy'],
        default='consolidated',
        help='Database structure type'
    )
    parser.add_argument(
        '--tiers',
        type=str,
        default='1,2,3,4',
        help='Comma-separated list of tiers to process'
    )
    parser.set_defaults(func=run_recreate_command)

def setup_recommend_command(subparsers) -> None:
    """
    Set up the 'recommend' command parser.
    
    Args:
        subparsers: The subparser collection to add the new parser to
    
    Returns:
        None
    """
    parser = subparsers.add_parser(
        'recommend',
        help='Get document recommendations'
    )
    parser.add_argument(
        '--query',
        type=str,
        help='Text query for recommendations'
    )
    parser.add_argument(
        '--document-id',
        type=str,
        help='Document ID to find similar documents for'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of recommendations to return'
    )
    parser.add_argument(
        '--filter-type',
        type=str,
        choices=['regulation', 'directive', 'decision'],
        help='Filter results by document type'
    )
    parser.add_argument(
        '--profile',
        type=str,
        help='Path to user profile JSON file for personalized recommendations'
    )
    parser.add_argument(
        '--temporal-boost',
        type=float,
        help='Weight for temporal boosting (0.0-1.0) to favor more recent documents'
    )
    parser.add_argument(
        '--reference-date',
        type=str,
        help='Reference date for temporal calculations in format YYYY-MM-DD (defaults to current date)'
    )
    parser.set_defaults(func=run_recommend_command)

def setup_profile_command(subparsers) -> None:
    """
    Set up the 'profile' command parser.
    
    Args:
        subparsers: The subparser collection to add the new parser to
    
    Returns:
        None
    """
    parser = subparsers.add_parser(
        'profile',
        help='Manage user profiles'
    )
    parser.add_argument(
        '--user-id',
        type=str,
        required=True,
        help='User ID for the profile'
    )
    parser.add_argument(
        '--load',
        type=str,
        help='Load profile from JSON file'
    )
    parser.add_argument(
        '--save',
        type=str,
        help='Save profile to JSON file'
    )
    parser.add_argument(
        '--add-document',
        type=str,
        help='Add historical document to profile'
    )
    parser.add_argument(
        '--set-expert-profile',
        type=str,
        help='Set expert profile description'
    )
    parser.set_defaults(func=run_profile_command)

def run_index_command(args) -> None:
    """
    Run the 'index' command to index documents into Pinecone.
    
    This function handles the indexing of documents into Pinecone, including:
    - Converting document data to embeddings
    - Storing embeddings in Pinecone
    - Processing documents by tier
    - Handling batch processing for efficiency
    
    Args:
        args: Command line arguments containing indexing parameters
        
    Returns:
        None
        
    Raises:
        ValidationError: If input parameters are invalid
        PineconeError: If there's an error with Pinecone operations
        EmbeddingError: If there's an error generating embeddings
    """
    logger.info(f"Running index command with tiers: {args.tiers}, db_type: {args.db_type}, batch_size: {args.batch_size}")
    
    # Import necessary modules
    try:
        logger.debug("Importing required modules")
        from src.models.pinecone_recommender import PineconeRecommender
        from src.models.features import FeatureProcessor
        from src.utils.db_connector import get_connector
        from src.models.embeddings import BERTEmbedder
    except ImportError as e:
        logger.error(f"Failed to import required modules: {str(e)}")
        raise ValidationError(f"Failed to import required modules: {str(e)}") from e
    
    # Convert tiers string to list of integers
    try:
        tiers = [int(t.strip()) for t in args.tiers.split(',')]
        logger.info(f"Processing tiers: {tiers}")
    except ValueError as e:
        logger.error(f"Invalid tier format in: {args.tiers}")
        raise ValidationError(f"Tiers must be comma-separated integers") from e
    
    # Get database connector
    try:
        logger.info(f"Initializing database connector for type: {args.db_type}")
        db_connector = get_connector(args.db_type)
    except Exception as e:
        logger.error(f"Failed to initialize database connector: {str(e)}")
        raise ValidationError(f"Failed to initialize database connector: {str(e)}") from e
    
    # Initialize feature processor
    try:
        logger.info("Initializing feature processor")
        feature_processor = FeatureProcessor()
        feature_processor.fit(db_connector)
        logger.info("Feature processor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize feature processor: {str(e)}")
        raise ValidationError(f"Failed to initialize feature processor: {str(e)}") from e
    
    # Initialize embedder
    try:
        logger.info(f"Initializing BERT embedder with {EMBEDDER['model_name']} model")
        embedder = BERTEmbedder(
            model_name=EMBEDDER['model_name'],
            device=EMBEDDER['device'],
            batch_size=EMBEDDER['batch_size']
        )
        logger.info("BERT embedder initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize BERT embedder: {str(e)}")
        raise EmbeddingError(f"Failed to initialize BERT embedder: {str(e)}") from e
    
    # Initialize recommender
    try:
        logger.info(f"Initializing Pinecone recommender with index: {PINECONE['index_name']}")
        recommender = PineconeRecommender(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENVIRONMENT,
            index_name=PINECONE['index_name'],
            embedder=embedder,
            feature_processor=feature_processor,
            recreate_index=args.recreate_index
        )
        logger.info("Pinecone recommender initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone recommender: {str(e)}")
        raise PineconeError(f"Failed to initialize Pinecone recommender: {str(e)}") from e
    
    # Index documents
    try:
        logger.info(f"Indexing documents from tiers {tiers} with batch size {args.batch_size}")
        
        # Process each tier
        for tier in tiers:
            logger.info(f"Processing tier {tier}")
            
            # Get documents for this tier
            try:
                documents = db_connector.get_documents_by_tier(tier, limit=args.limit)
                logger.info(f"Found {len(documents)} documents in tier {tier}")
                
                if not documents:
                    logger.warning(f"No documents found in tier {tier}")
                    continue
            except Exception as e:
                logger.error(f"Error retrieving documents for tier {tier}: {str(e)}")
                raise ValidationError(f"Failed to retrieve documents for tier {tier}: {str(e)}") from e
            
            # Process documents in batches
            total_batches = (len(documents) - 1) // args.batch_size + 1
            successful_docs = 0
            failed_docs = 0
            
            for i in range(0, len(documents), args.batch_size):
                batch = documents[i:i + args.batch_size]
                current_batch = i // args.batch_size + 1
                logger.info(f"Processing batch {current_batch}/{total_batches} with {len(batch)} documents")
                
                try:
                    # Index the batch
                    recommender.index_documents(batch)
                    successful_docs += len(batch)
                    logger.info(f"Successfully indexed batch {current_batch}/{total_batches}")
                except Exception as e:
                    failed_docs += len(batch)
                    logger.error(f"Error indexing batch {current_batch}/{total_batches}: {str(e)}")
                    # Continue with next batch instead of failing completely
                    continue
            
            logger.info(f"Tier {tier} processing complete. Successfully indexed {successful_docs} documents, failed {failed_docs} documents")
        
        logger.info("Indexing complete for all tiers")
    except Exception as e:
        logger.error(f"Unexpected error during indexing: {str(e)}")
        raise

def run_recreate_command(args) -> None:
    """
    Run the 'recreate' command to regenerate all embeddings.
    
    This function handles the regeneration of embeddings for all documents, including:
    - Deleting existing vectors from Pinecone
    - Generating new embeddings for all documents
    - Processing documents by tier
    - Handling metadata extraction and storage
    
    Args:
        args: Command line arguments containing recreation parameters
        
    Returns:
        None
        
    Raises:
        ValidationError: If input parameters are invalid
        PineconeError: If there's an error with Pinecone operations
        EmbeddingError: If there's an error generating embeddings
    """
    logger.info(f"Running recreate command with tiers: {args.tiers}, db_type: {args.db_type}")
    
    # Import necessary modules
    try:
        logger.debug("Importing required modules")
        from src.models.embeddings import BERTEmbedder
        from src.utils.pinecone_embeddings import PineconeEmbeddingManager
        from src.utils.db_connector import get_connector
    except ImportError as e:
        logger.error(f"Failed to import required modules: {str(e)}")
        raise ValidationError(f"Failed to import required modules: {str(e)}") from e
    
    # Convert tiers string to list of integers
    try:
        tiers = [int(t.strip()) for t in args.tiers.split(',')]
        logger.info(f"Processing tiers: {tiers}")
    except ValueError as e:
        logger.error(f"Invalid tier format in: {args.tiers}")
        raise ValidationError(f"Tiers must be comma-separated integers") from e
    
    # Initialize embedder with the configured model
    try:
        logger.info(f"Initializing BERT embedder with {EMBEDDER['model_name']} model")
        embedder = BERTEmbedder(
            model_name=EMBEDDER['model_name'],
            device=EMBEDDER['device'],
            batch_size=EMBEDDER['batch_size']
        )
        logger.info("BERT embedder initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize BERT embedder: {str(e)}")
        raise EmbeddingError(f"Failed to initialize BERT embedder: {str(e)}") from e
    
    # Initialize Pinecone embedding manager
    try:
        logger.info("Initializing Pinecone embedding manager")
        pinecone_manager = PineconeEmbeddingManager(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENVIRONMENT,
            index_name=PINECONE['index_name'],
            dimension=EMBEDDER['dimension'],
            embedder_model=EMBEDDER['model_name']
        )
        logger.info("Pinecone embedding manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone embedding manager: {str(e)}")
        raise PineconeError(f"Failed to initialize Pinecone embedding manager: {str(e)}") from e
    
    # Get database connector
    try:
        logger.info(f"Initializing database connector for type: {args.db_type}")
        db_connector = get_connector(args.db_type)
        logger.info("Database connector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database connector: {str(e)}")
        raise ValidationError(f"Failed to initialize database connector: {str(e)}") from e
    
    # Delete all existing vectors
    try:
        logger.info("Deleting all existing vectors from Pinecone")
        pinecone_manager.delete_all_vectors()
        logger.info("Successfully deleted all existing vectors")
    except Exception as e:
        logger.error(f"Failed to delete existing vectors: {str(e)}")
        raise PineconeError(f"Failed to delete existing vectors: {str(e)}") from e
    
    # Process each tier
    successful_docs = 0
    failed_docs = 0
    skipped_docs = 0
    
    try:
        for tier in tiers:
            logger.info(f"Processing tier {tier}")
            
            # Get documents for this tier
            try:
                documents = db_connector.get_documents_by_tier(tier)
                logger.info(f"Found {len(documents)} documents in tier {tier}")
                
                if not documents:
                    logger.warning(f"No documents found in tier {tier}")
                    continue
            except Exception as e:
                logger.error(f"Error retrieving documents for tier {tier}: {str(e)}")
                raise ValidationError(f"Failed to retrieve documents for tier {tier}: {str(e)}") from e
            
            # Generate and upsert embeddings for each document
            tier_successful = 0
            tier_failed = 0
            tier_skipped = 0
            
            for doc in documents:
                doc_id = doc.get('document_id')
                if not doc_id:
                    logger.warning("Skipping document with missing document_id")
                    skipped_docs += 1
                    tier_skipped += 1
                    continue
                    
                summary = doc.get('summary', '')
                keywords = doc.get('keywords', [])
                
                # Skip documents without summary or keywords
                if not summary and not keywords:
                    logger.warning(f"Skipping document {doc_id} - no summary or keywords")
                    skipped_docs += 1
                    tier_skipped += 1
                    continue
                
                # Generate embeddings
                try:
                    pinecone_manager.generate_and_upsert_embeddings(
                        document_id=doc_id,
                        summary=summary,
                        keywords=keywords,
                        metadata={
                            'document_type': doc.get('document_type', ''),
                            'title': doc.get('title', ''),
                            'date': doc.get('date', ''),
                            'author': doc.get('author', ''),
                            'subject_matters': doc.get('subject_matters', []),
                            'form': doc.get('form', ''),
                            'tier': tier
                        }
                    )
                    successful_docs += 1
                    tier_successful += 1
                except Exception as e:
                    logger.error(f"Failed to generate and upsert embeddings for document {doc_id}: {str(e)}")
                    failed_docs += 1
                    tier_failed += 1
                    # Continue with next document instead of failing completely
                    continue
            
            logger.info(f"Tier {tier} processing complete. Successfully processed {tier_successful} documents, failed {tier_failed}, skipped {tier_skipped}")
        
        logger.info(f"Recreating embeddings complete. Successfully processed {successful_docs} documents, failed {failed_docs}, skipped {skipped_docs}")
    except Exception as e:
        logger.error(f"Unexpected error during embedding recreation: {str(e)}")
        raise

def run_recommend_command(args) -> None:
    """
    Run the 'recommend' command to get document recommendations.
    
    Args:
        args: Command line arguments containing recommendation parameters
        
    Returns:
        None
        
    Raises:
        ValidationError: If input parameters are invalid
        PineconeError: If there's an error with Pinecone operations
        EmbeddingError: If there's an error generating embeddings
        RecommendationError: If there's an error generating recommendations
    """
    # Validate that either query or document_id is provided
    if not args.query and not args.document_id:
        logger.error("Either --query or --document-id must be provided")
        raise ValidationError("Either --query or --document-id must be provided")
    
    if args.query:
        # Text-based recommendation
        try:
            # Validate query
            if not isinstance(args.query, str) or not args.query.strip():
                logger.error("Query must be a non-empty string")
                raise ValidationError("Query must be a non-empty string")
                
            logger.info(f"Initializing recommender for query: '{args.query[:50]}...'")
            
            # Initialize feature processor with default weights
            from src.models.features import FeatureProcessor
            feature_processor = FeatureProcessor()
            
            # Choose recommender class based on whether a profile is provided
            if args.profile:
                from src.models.personalized_recommender import PersonalizedRecommender
                logger.info(f"Using PersonalizedRecommender with profile: {args.profile}")
                
                # Initialize personalized recommender
                recommender = PersonalizedRecommender(
                    api_key=PINECONE_API_KEY,
                    index_name=PINECONE['index_name'],
                    embedder_model=EMBEDDER['model_name'],
                    feature_processor=feature_processor
                )
            else:
                from src.models.pinecone_recommender import PineconeRecommender
                logger.info("Using standard PineconeRecommender")
                
                # Initialize standard recommender
                recommender = PineconeRecommender(
                    api_key=PINECONE_API_KEY,
                    index_name=PINECONE['index_name'],
                    embedder_model=EMBEDDER['model_name'],
                    feature_processor=feature_processor
                )
        except ImportError as e:
            logger.error(f"Failed to import required modules: {str(e)}")
            raise RecommendationError(f"Failed to import required modules: {str(e)}") from e
        except Exception as e:
            logger.error(f"Failed to initialize recommender: {str(e)}")
            raise RecommendationError(f"Failed to initialize recommender: {str(e)}") from e
        
        # Get recommendations
        try:
            # Load user profile if specified
            client_preferences = None
            if args.profile:
                try:
                    logger.info(f"Loading user profile from {args.profile}")
                    with open(args.profile, 'r') as f:
                        profile_data = json.load(f)
                    
                    # Extract profile information
                    profile = profile_data.get('profile', {})
                    user_id = profile_data.get('user_id')
                    
                    if not user_id:
                        logger.warning("Profile missing user_id, using 'default_user'")
                        user_id = 'default_user'
                    
                    # Set up client preferences from categorical preferences
                    if 'categorical_preferences' in profile:
                        client_preferences = profile['categorical_preferences']
                        logger.info(f"Loaded client preferences with {len(client_preferences)} categories")
                    
                    # Add historical documents if available
                    if 'historical_documents' in profile and profile['historical_documents']:
                        historical_docs = profile['historical_documents']
                        logger.info(f"Adding {len(historical_docs)} historical documents to user profile")
                        for doc_id in historical_docs:
                            try:
                                recommender.add_historical_document(user_id, doc_id)
                                logger.info(f"Added historical document {doc_id} to user profile")
                            except Exception as e:
                                logger.warning(f"Failed to add historical document {doc_id}: {str(e)}")
                    
                    # Create expert profile if available
                    if 'expert_profile' in profile and 'description' in profile['expert_profile']:
                        expert_description = profile['expert_profile']['description']
                        if expert_description:
                            try:
                                logger.info(f"Creating expert profile for user {user_id}")
                                recommender.create_expert_profile(user_id, expert_description)
                                logger.info(f"Created expert profile for user {user_id}")
                            except Exception as e:
                                logger.warning(f"Failed to create expert profile: {str(e)}")
                        else:
                            logger.warning("Expert profile description is empty, skipping")
                            
                except Exception as e:
                    logger.error(f"Error loading profile: {str(e)}")
                    raise ProfileError(f"Error loading profile: {str(e)}")
            
            logger.info(f"Getting recommendations for query with top_k={args.top_k}, filter_type={args.filter_type}")
            # Create filter if document type is specified
            filter_dict = None
            if args.filter_type:
                filter_dict = {"document_type": args.filter_type}
                
            # Set up temporal boosting parameters if provided
            temporal_boost = args.temporal_boost if hasattr(args, 'temporal_boost') else None
            reference_date = args.reference_date if hasattr(args, 'reference_date') else None
            
            if temporal_boost is not None:
                logger.info(f"Using temporal boosting with weight {temporal_boost}")
                if reference_date:
                    logger.info(f"Using reference date: {reference_date}")
                else:
                    logger.info("Using current date as reference date")
            
            # Use personalized recommendations if profile is provided, otherwise use standard recommendations
            if args.profile:
                # Get user ID from profile
                user_id = profile_data.get('user_id', 'default_user')
                
                logger.info(f"Getting personalized recommendations for user {user_id}")
                # PersonalizedRecommender.get_personalized_recommendations doesn't accept temporal_boost
                # We need to filter out parameters it doesn't accept
                try:
                    recommendations = recommender.get_personalized_recommendations(
                        user_id=user_id,
                        query_text=args.query,
                        top_k=args.top_k,
                        filter=filter_dict
                        # temporal_boost is not supported in this method
                    )
                except TypeError as e:
                    if "temporal_boost" in str(e) or "reference_date" in str(e):
                        logger.warning("Personalized recommender doesn't support temporal_boost, trying without it")
                        recommendations = recommender.get_personalized_recommendations(
                            user_id=user_id,
                            query_text=args.query,
                            top_k=args.top_k,
                            filter=filter_dict
                        )
                    else:
                        raise
            else:
                logger.info("Getting standard recommendations")
                recommendations = recommender.get_recommendations(
                    query_text=args.query,
                    top_k=args.top_k,
                    filter=filter_dict,
                    temporal_boost=temporal_boost,
                    reference_date=reference_date
                )
            logger.info(f"Found {len(recommendations)} recommendations")
        except Exception as e:
            logger.error(f"Failed to get recommendations: {str(e)}")
            raise RecommendationError(f"Failed to get recommendations: {str(e)}") from e
        
        # Display recommendations
        print(f"\nFound {len(recommendations)} recommendations for query: '{args.query}'")
        if args.temporal_boost:
            print(f"Temporal boosting applied with weight: {args.temporal_boost}")
            
        for i, rec in enumerate(recommendations, 1):
            score_info = f"Score: {rec['score']:.4f}"
            
            # Add temporal boosting details if available
            if 'original_similarity' in rec and 'temporal_score' in rec:
                score_info += f" (Original: {rec['original_similarity']:.4f}, Temporal: {rec['temporal_score']:.4f})"
                
            print(f"\n{i}. Document ID: {rec['id']} ({score_info})")
            
            if 'metadata' in rec and rec['metadata']:
                # Display CELEX number
                if 'celex_number' in rec['metadata']:
                    print(f"   - CELEX: {rec['metadata']['celex_number']}")
                
                # Display title
                if 'title' in rec['metadata']:
                    title = rec['metadata']['title']
                    if len(title) > 100:
                        title = title[:100] + "..."
                    print(f"   - Title: {title}")
                
                # Display document type
                if 'document_type' in rec['metadata']:
                    print(f"   - Type: {rec['metadata']['document_type']}")
                
                # Display subject matters
                if 'subject_matters' in rec['metadata']:
                    subjects = rec['metadata']['subject_matters']
                    if isinstance(subjects, list) and len(subjects) > 3:
                        subjects = subjects[:3] + ["..."]
                    print(f"   - Subjects: {subjects}")
    
    elif args.document_id:
        # Document similarity
        try:
            from src.models.features import FeatureProcessor
            
            # Validate document_id
            if not isinstance(args.document_id, str) or not args.document_id.strip():
                logger.error("Document ID must be a non-empty string")
                raise ValidationError("Document ID must be a non-empty string")
                
            logger.info(f"Initializing recommender for document similarity: '{args.document_id}'")
            
            # Initialize feature processor with default weights
            feature_processor = FeatureProcessor()
            
            # Choose recommender class based on whether a profile is provided
            if args.profile:
                from src.models.personalized_recommender import PersonalizedRecommender
                logger.info(f"Using PersonalizedRecommender with profile: {args.profile}")
                
                # Initialize personalized recommender
                recommender = PersonalizedRecommender(
                    api_key=PINECONE_API_KEY,
                    index_name=PINECONE['index_name'],
                    embedder_model=EMBEDDER['model_name'],
                    feature_processor=feature_processor
                )
            else:
                from src.models.pinecone_recommender import PineconeRecommender
                logger.info("Using standard PineconeRecommender")
                
                # Initialize standard recommender
                recommender = PineconeRecommender(
                    api_key=PINECONE_API_KEY,
                    index_name=PINECONE['index_name'],
                    embedder_model=EMBEDDER['model_name'],
                    feature_processor=feature_processor
                )
        except ImportError as e:
            logger.error(f"Failed to import required modules: {str(e)}")
            raise RecommendationError(f"Failed to import required modules: {str(e)}") from e
        except Exception as e:
            logger.error(f"Failed to initialize recommender: {str(e)}")
            raise RecommendationError(f"Failed to initialize recommender: {str(e)}") from e
        
        # Get similar documents
        try:
            # Load user profile if specified
            client_preferences = None
            user_id = None
            if args.profile:
                try:
                    logger.info(f"Loading user profile from {args.profile}")
                    with open(args.profile, 'r') as f:
                        profile_data = json.load(f)
                    
                    # Extract profile information
                    profile = profile_data.get('profile', {})
                    user_id = profile_data.get('user_id')
                    
                    if not user_id:
                        logger.warning("Profile missing user_id, using 'default_user'")
                        user_id = 'default_user'
                    
                    # Set up client preferences from categorical preferences
                    if 'categorical_preferences' in profile:
                        client_preferences = profile['categorical_preferences']
                        logger.info(f"Loaded client preferences with {len(client_preferences)} categories")
                    
                    # Add historical documents if available
                    if 'historical_documents' in profile and profile['historical_documents']:
                        historical_docs = profile['historical_documents']
                        logger.info(f"Adding {len(historical_docs)} historical documents to user profile")
                        for doc_id in historical_docs:
                            try:
                                recommender.add_historical_document(user_id, doc_id)
                                logger.info(f"Added historical document {doc_id} to user profile")
                            except Exception as e:
                                logger.warning(f"Failed to add historical document {doc_id}: {str(e)}")
                    
                    # Create expert profile if available
                    if 'expert_profile' in profile and 'description' in profile['expert_profile']:
                        expert_description = profile['expert_profile']['description']
                        if expert_description:
                            try:
                                logger.info(f"Creating expert profile for user {user_id}")
                                recommender.create_expert_profile(user_id, expert_description)
                                logger.info(f"Created expert profile for user {user_id}")
                            except Exception as e:
                                logger.warning(f"Failed to create expert profile: {str(e)}")
                        else:
                            logger.warning("Expert profile description is empty, skipping")
                            
                except Exception as e:
                    logger.error(f"Error loading profile: {str(e)}")
                    raise ProfileError(f"Error loading profile: {str(e)}")
            
            logger.info(f"Getting similar documents for {args.document_id} with top_k={args.top_k}, filter_type={args.filter_type}, temporal_boost={args.temporal_boost}")
            # Create filter if document type is specified
            filter_dict = None
            if args.filter_type:
                filter_dict = {"document_type": args.filter_type}
                
            similar_docs = recommender.get_recommendations_by_id(
                document_id=args.document_id,
                top_k=args.top_k,
                filter=filter_dict,
                client_preferences=client_preferences,
                temporal_boost=args.temporal_boost,
                reference_date=args.reference_date
            )
            logger.info(f"Found {len(similar_docs)} similar documents")
        except Exception as e:
            logger.error(f"Failed to get similar documents: {str(e)}")
            raise RecommendationError(f"Failed to get similar documents: {str(e)}") from e
        
        # Display similar documents
        print(f"\nFound {len(similar_docs)} documents similar to '{args.document_id}'")
        for i, doc in enumerate(similar_docs, 1):
            # Display document with score and temporal score if available
            score_info = f"Score: {doc['score']:.4f}"
            if 'temporal_score' in doc:
                score_info += f", Temporal: {doc['temporal_score']:.4f}"
            if 'original_similarity' in doc:
                score_info += f", Original: {doc['original_similarity']:.4f}"
            print(f"\n{i}. Document ID: {doc['id']} ({score_info})")
            
            if 'metadata' in doc and doc['metadata']:
                # Display CELEX number
                if 'celex_number' in doc['metadata']:
                    print(f"   - CELEX: {doc['metadata']['celex_number']}")
                
                # Display title
                if 'title' in doc['metadata']:
                    title = doc['metadata']['title']
                    if len(title) > 100:
                        title = title[:100] + "..."
                    print(f"   - Title: {title}")
                
                # Display document type
                if 'document_type' in doc['metadata']:
                    print(f"   - Type: {doc['metadata']['document_type']}")
                
                # Display subject matters
                if 'subject_matters' in doc['metadata']:
                    subjects = doc['metadata']['subject_matters']
                    if isinstance(subjects, list) and len(subjects) > 3:
                        subjects = subjects[:3] + ["..."]
                    print(f"   - Subjects: {subjects}")
                
                # Display date information (important for temporal boosting)
                date_fields = ['date', 'publication_date', 'document_date']
                for field in date_fields:
                    if field in doc['metadata'] and doc['metadata'][field]:
                        print(f"   - {field.replace('_', ' ').title()}: {doc['metadata'][field]}")
                        break
    
    else:
        # This should never happen due to the validation at the beginning of the function
        logger.error("Either --query or --document-id must be provided")
        raise ValidationError("Either --query or --document-id must be provided")

def run_profile_command(args) -> None:
    """
    Run the 'profile' command to manage user profiles.
    
    This function handles various profile management operations, including:
    - Loading profiles from files
    - Saving profiles to files
    - Adding historical documents to profiles
    - Setting expert profiles based on descriptions
    - Managing categorical preferences
    
    Args:
        args: Command line arguments containing profile management parameters
        
    Returns:
        None
        
    Raises:
        ValidationError: If input parameters are invalid
        ProfileError: If there's an error with profile operations
        PineconeError: If there's an error with Pinecone operations
        EmbeddingError: If there's an error generating embeddings for profiles
    """
    # Validate user_id
    if not args.user_id or not isinstance(args.user_id, str):
        logger.error("User ID must be a non-empty string")
        raise ValidationError("User ID must be a non-empty string")
    
    # Validate that at least one operation is specified
    if not any([args.load, args.save, args.add_document, args.set_expert_profile]):
        logger.error("At least one profile operation must be specified")
        raise ValidationError("At least one profile operation must be specified. Use --load, --save, --add-document, or --set-expert-profile.")
    
    try:
        logger.debug("Importing required modules")
        from src.models.personalized_recommender import PersonalizedRecommender
        from src.models.features import FeatureProcessor
    except ImportError as e:
        logger.error(f"Failed to import required modules: {str(e)}")
        raise ValidationError(f"Failed to import required modules: {str(e)}") from e
    
    logger.info(f"Initializing personalized recommender for user {args.user_id}")
    
    try:
        # Initialize feature processor with default weights
        logger.debug("Initializing feature processor")
        feature_processor = FeatureProcessor()
        
        # Initialize recommender
        logger.debug(f"Initializing personalized recommender with index: {PINECONE['index_name']}")
        recommender = PersonalizedRecommender(
            api_key=PINECONE_API_KEY,
            index_name=PINECONE['index_name'],
            embedder_model=EMBEDDER['model_name'],
            feature_processor=feature_processor
        )
        logger.info("Personalized recommender initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize personalized recommender: {str(e)}")
        raise ProfileError(f"Failed to initialize personalized recommender: {str(e)}") from e
    
    user_id = args.user_id
    
    if args.load:
        # Load profile from file
        try:
            logger.info(f"Loading profile from file: {args.load}")
            
            # Validate file exists
            if not os.path.exists(args.load):
                logger.error(f"Profile file not found: {args.load}")
                raise ValidationError(f"Profile file not found: {args.load}")
                
            # Load and parse JSON
            try:
                logger.debug(f"Reading profile data from {args.load}")
                with open(args.load, 'r') as f:
                    profile_data = json.load(f)
                logger.debug("Successfully parsed profile JSON data")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in profile file: {str(e)}")
                raise ValidationError(f"Invalid JSON in profile file: {str(e)}") from e
            except IOError as e:
                logger.error(f"Error reading profile file: {str(e)}")
                raise ValidationError(f"Error reading profile file: {str(e)}") from e
                
            # Extract profile components
            profile = profile_data.get('profile', {})
            if not profile:
                logger.warning("Profile data is empty or missing 'profile' key")
                logger.debug(f"Profile data structure: {profile_data}")
            
            # Create expert profile if available
            if 'expert_profile' in profile and 'description' in profile['expert_profile']:
                expert_description = profile['expert_profile']['description']
                if expert_description:
                    logger.info(f"Creating expert profile from description: {expert_description[:50]}...")
                    try:
                        recommender.create_expert_profile(user_id, expert_description)
                        logger.info("Expert profile created successfully")
                    except Exception as e:
                        logger.error(f"Failed to create expert profile: {str(e)}")
                        raise ProfileError(f"Failed to create expert profile: {str(e)}") from e
                else:
                    logger.warning("Expert profile description is empty, skipping")
            else:
                logger.debug("No expert profile found in the loaded data")
            
            # Add historical documents if available
            if 'historical_documents' in profile:
                historical_docs = profile['historical_documents']
                if historical_docs:
                    logger.info(f"Adding {len(historical_docs)} historical documents to profile")
                    successful_docs = 0
                    failed_docs = 0
                    
                    for doc_id in historical_docs:
                        try:
                            recommender.add_historical_document(user_id, doc_id)
                            successful_docs += 1
                        except Exception as e:
                            logger.warning(f"Failed to add document {doc_id}: {str(e)}")
                            failed_docs += 1
                    
                    logger.info(f"Added {successful_docs} documents successfully, {failed_docs} failed")
                else:
                    logger.info("No historical documents to add")
            else:
                logger.debug("No historical documents found in the loaded data")
            
            # Set categorical preferences if available
            if 'categorical_preferences' in profile:
                categorical_prefs = profile['categorical_preferences']
                if categorical_prefs:
                    logger.info(f"Setting categorical preferences: {list(categorical_prefs.keys())}")
                    try:
                        recommender.set_categorical_preferences(user_id, categorical_prefs)
                        logger.info("Categorical preferences set successfully")
                    except Exception as e:
                        logger.error(f"Failed to set categorical preferences: {str(e)}")
                        raise ProfileError(f"Failed to set categorical preferences: {str(e)}") from e
                else:
                    logger.info("No categorical preferences to set")
            else:
                logger.debug("No categorical preferences found in the loaded data")
                
            logger.info(f"Successfully loaded profile for user: {user_id}")
            
        except ValidationError as e:
            # Re-raise validation errors
            logger.error(f"Validation error loading profile: {str(e)}")
            raise
        except ProfileError as e:
            # Re-raise profile errors
            logger.error(f"Profile error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading profile: {str(e)}")
            raise ProfileError(f"Failed to load profile: {str(e)}") from e
    
    if args.save:
        # Save profile to file
        try:
            logger.info(f"Saving profile to file: {args.save}")
            
            # Validate save path
            if not args.save or not isinstance(args.save, str):
                logger.error("Save path must be a non-empty string")
                raise ValidationError("Save path must be a non-empty string")
            
            # Get user profile
            try:
                logger.debug(f"Retrieving profile for user: {user_id}")
                user_profile = recommender.get_user_profile(user_id)
                
                if not user_profile:
                    logger.error(f"No profile found for user: {user_id}")
                    raise ProfileError(f"No profile found for user: {user_id}")
                
                logger.debug("User profile retrieved successfully")
            except Exception as e:
                logger.error(f"Error retrieving user profile: {str(e)}")
                raise ProfileError(f"Failed to retrieve user profile: {str(e)}") from e
            
            # Create profile data structure
            try:
                logger.debug("Creating profile data structure for serialization")
                
                # Handle potential None values in expert profile
                expert_profile_embedding = None
                if user_profile.expert_profile is not None:
                    try:
                        expert_profile_embedding = user_profile.expert_profile.tolist()
                    except Exception as e:
                        logger.warning(f"Could not convert expert profile embedding to list: {str(e)}")
                
                profile_data = {
                    'user_id': user_id,
                    'profile': {
                        'expert_profile': {
                            'description': user_profile.expert_profile_text,
                            'embedding': expert_profile_embedding
                        },
                        'historical_documents': user_profile.historical_documents,
                        'categorical_preferences': user_profile.categorical_preferences
                    }
                }
                logger.debug("Profile data structure created successfully")
            except Exception as e:
                logger.error(f"Error creating profile data structure: {str(e)}")
                raise ProfileError(f"Failed to create profile data structure: {str(e)}") from e
            
            # Save to file
            try:
                # Ensure directory exists
                logger.debug(f"Creating directory structure for: {args.save}")
                save_path = Path(args.save)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                logger.debug("Writing profile data to file")
                with open(args.save, 'w') as f:
                    json.dump(profile_data, f, indent=2)
                    
                logger.info(f"Successfully saved profile for user: {user_id} to {args.save}")
            except IOError as e:
                logger.error(f"I/O error writing profile to file: {str(e)}")
                raise ValidationError(f"I/O error writing profile to file: {str(e)}") from e
            except Exception as e:
                logger.error(f"Error writing profile to file: {str(e)}")
                raise ValidationError(f"Failed to write profile to file: {str(e)}") from e
            
        except ValidationError:
            # Re-raise validation errors
            raise
        except ProfileError:
            # Re-raise profile errors
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving profile: {str(e)}")
            raise ProfileError(f"Failed to save profile: {str(e)}") from e
    
    if args.add_document:
        # Add historical document to profile
        try:
            logger.info(f"Adding document {args.add_document} to profile for user: {user_id}")
            
            # Validate document ID
            if not args.add_document or not isinstance(args.add_document, str):
                logger.error("Document ID must be a non-empty string")
                raise ValidationError("Document ID must be a non-empty string")
            
            # Check if document exists in Pinecone
            try:
                logger.debug(f"Verifying document exists: {args.add_document}")
                # Note: This verification would depend on your implementation
                # You might want to add a method to check if a document exists
            except Exception as e:
                logger.warning(f"Could not verify document existence: {str(e)}")
                # Continue anyway, as the document might still be valid
                
            # Add document to profile
            try:
                recommender.add_historical_document(user_id, args.add_document)
                logger.info(f"Successfully added document {args.add_document} to profile for user: {user_id}")
            except Exception as e:
                logger.error(f"Failed to add document to profile: {str(e)}")
                raise ProfileError(f"Failed to add document to profile: {str(e)}") from e
            
        except ValidationError:
            # Re-raise validation errors
            raise
        except ProfileError:
            # Re-raise profile errors
            raise
        except Exception as e:
            logger.error(f"Unexpected error adding document to profile: {str(e)}")
            raise ProfileError(f"Failed to add document to profile: {str(e)}") from e
    
    if args.set_expert_profile:
        # Set expert profile description
        try:
            logger.info(f"Setting expert profile for user: {user_id}")
            
            # Validate profile text
            if not args.set_expert_profile or not isinstance(args.set_expert_profile, str):
                logger.error("Expert profile description must be a non-empty string")
                raise ValidationError("Expert profile description must be a non-empty string")
            
            if len(args.set_expert_profile.strip()) < 10:
                logger.warning("Expert profile description is very short, this may result in poor recommendations")
                
            # Create expert profile
            try:
                logger.debug("Generating expert profile embedding")
                recommender.create_expert_profile(user_id, args.set_expert_profile)
                logger.info(f"Successfully set expert profile for user: {user_id}")
            except Exception as e:
                logger.error(f"Failed to create expert profile: {str(e)}")
                if "embedding" in str(e).lower():
                    raise EmbeddingError(f"Failed to generate embedding for expert profile: {str(e)}") from e
                else:
                    raise ProfileError(f"Failed to set expert profile: {str(e)}") from e
            
        except ValidationError:
            # Re-raise validation errors
            raise
        except ProfileError:
            # Re-raise profile errors
            raise
        except EmbeddingError:
            # Re-raise embedding errors
            raise
        except Exception as e:
            logger.error(f"Unexpected error setting expert profile: {str(e)}")
            raise ProfileError(f"Failed to set expert profile: {str(e)}") from e

# Caching decorator for expensive operations
def cache_result(func: Callable) -> Callable:
    """
    Cache the result of a function call to improve performance.
    
    This decorator stores the results of function calls based on their arguments
    to avoid redundant computation for repeated calls with the same parameters.
    
    Args:
        func: The function to cache results for
        
    Returns:
        A wrapped function that caches results
        
    Example:
        @cache_result
        def expensive_operation(param1, param2):
            # Expensive computation here
            return result
    """
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Create a cache key from the function arguments
            key = str(args) + str(sorted(kwargs.items()))
            
            # Check if result is in cache
            if key in cache:
                logger.debug(f"Using cached result for {func.__name__}")
                return cache[key]
            
            # Call the function and cache the result
            logger.debug(f"Computing new result for {func.__name__}")
            result = func(*args, **kwargs)
            cache[key] = result
            return result
        except Exception as e:
            logger.error(f"Error in cached function {func.__name__}: {str(e)}")
            raise
    
    return wrapper

def main() -> None:
    """
    Main entry point for the CLI.
    
    Parses command line arguments and dispatches to the appropriate handler.
    This function handles the top-level command line interface, including:
    - Loading environment variables
    - Validating required configuration
    - Setting up command parsers
    - Dispatching to command handlers
    - Error handling for all commands
    
    Returns:
        None
        
    Raises:
        SystemExit: If required environment variables are missing or arguments are invalid
    """
    # Load environment variables
    logger.info("Loading environment variables")
    load_dotenv()
    
    # Check for required environment variables
    try:
        logger.info("Validating environment variables")
        missing_vars = []
        
        if not PINECONE_API_KEY:
            missing_vars.append("PINECONE_API_KEY")
            
        if not PINECONE_ENVIRONMENT:
            missing_vars.append("PINECONE_ENVIRONMENT")
        
        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            logger.error(error_msg)
            raise ConfigError(error_msg)
            
        logger.info("Environment variables validated successfully")
    except ConfigError as e:
        logger.error(f"Configuration error: {str(e)}")
        print(f"Error: {str(e)}")
        print("Please set the required environment variables in a .env file or in your environment.")
        sys.exit(1)
    
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description='EU Legal Recommender System CLI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        title='commands',
        dest='command',
        help='Command to run'
    )
    
    # Set up command parsers
    setup_index_command(subparsers)
    setup_recreate_command(subparsers)
    setup_recommend_command(subparsers)
    setup_profile_command(subparsers)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the appropriate command
    if hasattr(args, 'func'):
        try:
            logger.info(f"Running command: {args.command}")
            start_time = time.time()
            args.func(args)
            end_time = time.time()
            logger.info(f"Command {args.command} completed successfully in {end_time - start_time:.2f} seconds")
        except ValidationError as e:
            logger.error(f"Validation error executing {args.command} command: {str(e)}")
            print(f"Error: {str(e)}")
            sys.exit(1)
        except PineconeError as e:
            logger.error(f"Pinecone error executing {args.command} command: {str(e)}")
            print(f"Pinecone Error: {str(e)}")
            sys.exit(1)
        except EmbeddingError as e:
            logger.error(f"Embedding error executing {args.command} command: {str(e)}")
            print(f"Embedding Error: {str(e)}")
            sys.exit(1)
        except ProfileError as e:
            logger.error(f"Profile error executing {args.command} command: {str(e)}")
            print(f"Profile Error: {str(e)}")
            sys.exit(1)
        except RecommendationError as e:
            logger.error(f"Recommendation error executing {args.command} command: {str(e)}")
            print(f"Recommendation Error: {str(e)}")
            sys.exit(1)
        except ConfigError as e:
            logger.error(f"Configuration error executing {args.command} command: {str(e)}")
            print(f"Configuration Error: {str(e)}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error executing {args.command} command: {str(e)}", exc_info=True)
            print(f"Error: An unexpected error occurred: {str(e)}")
            print("Check the logs for more details.")
            sys.exit(1)
    else:
        logger.warning("No command specified")
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
