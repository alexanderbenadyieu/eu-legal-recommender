#!/usr/bin/env python
"""
Test script for debugging and testing individual components of the EU Legal Recommender system.

This script allows testing of:
- Embeddings generation
- Feature processing
- Similarity computation
- Document ranking
- Temporal boosting
- Weight adjustment
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path to ensure imports work
sys.path.append(str(Path(__file__).parent.parent))

from src.models.embeddings import BERTEmbedder
from src.models.features import FeatureProcessor
from src.models.similarity import SimilarityComputer
from src.models.ranker import DocumentRanker
from src.utils.logging import setup_logger

# Set up logger
logger = setup_logger('component_tester', log_file='logs/component_tests.log')


def test_embedder(model_name: str = "nlpaueb/legal-bert-base-uncased", 
                 sample_texts: Optional[List[str]] = None) -> None:
    """
    Test the embedding component.
    
    Args:
        model_name: Name of the BERT model to use
        sample_texts: List of sample texts to embed
    """
    logger.info(f"Testing BERTEmbedder with model {model_name}")
    
    # Create default sample texts if none provided
    if not sample_texts:
        sample_texts = [
            "This is a test of the EU legal recommender system.",
            "Regulation concerning the rights of citizens in the European Union.",
            "Directive on environmental protection and climate change mitigation.",
            "Decision on competition law and market regulation."
        ]
    
    # Initialize embedder
    start_time = time.time()
    embedder = BERTEmbedder(model_name=model_name)
    init_time = time.time() - start_time
    logger.info(f"Embedder initialization time: {init_time:.4f}s")
    
    # Test individual embeddings
    logger.info("Testing individual embeddings:")
    for i, text in enumerate(sample_texts):
        start_time = time.time()
        embedding = embedder.generate_embedding(text)
        embed_time = time.time() - start_time
        
        logger.info(f"  Text {i+1} ({len(text)} chars): {embed_time:.4f}s, "
                   f"embedding shape: {embedding.shape}")
    
    # Test batch embeddings
    start_time = time.time()
    batch_embeddings = embedder.generate_embeddings(sample_texts)
    batch_time = time.time() - start_time
    
    logger.info(f"Batch embedding ({len(sample_texts)} texts): {batch_time:.4f}s, "
               f"embeddings shape: {batch_embeddings.shape}")
    
    # Test keyword weighting
    sample_summary = "This is a summary of an EU legal document."
    sample_keywords = ["legal", "document", "EU", "regulation"]
    
    start_time = time.time()
    combined_embedding = embedder.combine_text_features(
        summary=sample_summary,
        keywords=sample_keywords
    )
    combine_time = time.time() - start_time
    
    logger.info(f"Combined embedding (summary + {len(sample_keywords)} keywords): "
               f"{combine_time:.4f}s, shape: {combined_embedding.shape}")
    
    logger.info("Embedder test completed successfully")


def test_feature_processor(sample_documents: Optional[List[Dict[str, Any]]] = None) -> None:
    """
    Test the feature processing component.
    
    Args:
        sample_documents: List of sample documents with features
    """
    logger.info("Testing FeatureProcessor")
    
    # Create default sample documents if none provided
    if not sample_documents:
        sample_documents = [
            {
                "id": "doc1",
                "features": {
                    "document_type": "regulation",
                    "subject_matter": "environment",
                    "author": "commission",
                    "date": "2022"
                }
            },
            {
                "id": "doc2",
                "features": {
                    "document_type": "directive",
                    "subject_matter": "competition",
                    "author": "parliament",
                    "date": "2021"
                }
            },
            {
                "id": "doc3",
                "features": {
                    "document_type": "decision",
                    "subject_matter": "agriculture",
                    "author": "council",
                    "date": "2023"
                }
            },
            {
                "id": "doc4",
                "features": {
                    "document_type": "regulation",
                    "subject_matter": "finance",
                    "author": "commission",
                    "date": "2020"
                }
            }
        ]
    
    # Define feature configuration
    feature_config = {
        "document_type": {"weight": 0.4},
        "subject_matter": {"weight": 0.4},
        "author": {"weight": 0.2}
    }
    
    # Initialize feature processor
    start_time = time.time()
    feature_processor = FeatureProcessor(feature_config=feature_config)
    init_time = time.time() - start_time
    logger.info(f"FeatureProcessor initialization time: {init_time:.4f}s")
    
    # Fit feature processor
    start_time = time.time()
    feature_processor.fit([doc["features"] for doc in sample_documents])
    fit_time = time.time() - start_time
    logger.info(f"Feature processor fitting time: {fit_time:.4f}s")
    
    # Get feature info
    for feature_name in feature_config:
        if hasattr(feature_processor.encoders, feature_name):
            encoder = getattr(feature_processor.encoders, feature_name)
            categories = getattr(encoder, 'categories_', None)
            if categories:
                logger.info(f"  Feature '{feature_name}' has {len(categories[0])} categories: {categories[0]}")
    
    # Test encoding individual documents
    logger.info("Testing feature encoding:")
    for i, doc in enumerate(sample_documents):
        start_time = time.time()
        encoded = feature_processor.encode_features(doc["features"])
        encode_time = time.time() - start_time
        
        logger.info(f"  Document {i+1}: {encode_time:.4f}s, encoded shape: {encoded.shape}")
    
    # Test encoding batch (manually implement batch encoding)
    start_time = time.time()
    batch_encoded = np.vstack([
        feature_processor.encode_features(doc["features"]) for doc in sample_documents
    ])
    batch_time = time.time() - start_time
    
    logger.info(f"Batch encoding ({len(sample_documents)} documents): {batch_time:.4f}s, "
               f"encoded shape: {batch_encoded.shape}")
    
    # Test with adjusted weights
    adjusted_weights = {
        "document_type": 0.6,
        "subject_matter": 0.3,
        "author": 0.1
    }
    
    start_time = time.time()
    # Directly update the feature_weights attribute
    feature_processor.feature_weights = adjusted_weights
    update_time = time.time() - start_time
    
    logger.info(f"Weight update time: {update_time:.4f}s")
    logger.info(f"Updated weights: {feature_processor.get_feature_weights()}")
    
    logger.info("Feature processor test completed successfully")


def test_similarity_computer(sample_documents: Optional[List[Dict[str, Any]]] = None) -> None:
    """
    Test the similarity computation component.
    
    Args:
        sample_documents: List of sample documents with text embeddings and features
    """
    logger.info("Testing SimilarityComputer")
    
    # Create embedder and feature processor for testing
    embedder = BERTEmbedder()
    
    feature_config = {
        "document_type": {"weight": 0.4},
        "subject_matter": {"weight": 0.4},
        "author": {"weight": 0.2}
    }
    feature_processor = FeatureProcessor(feature_config=feature_config)
    
    # Create default sample documents if none provided
    if not sample_documents:
        # Generate sample documents with text and features
        sample_texts = [
            "Regulation on environmental protection measures in the European Union.",
            "Directive concerning competition law and market regulation.",
            "Decision on agricultural policy and rural development.",
            "Regulation regarding financial services and banking supervision."
        ]
        
        sample_features = [
            {
                "document_type": "regulation",
                "subject_matter": "environment",
                "author": "commission"
            },
            {
                "document_type": "directive",
                "subject_matter": "competition",
                "author": "parliament"
            },
            {
                "document_type": "decision",
                "subject_matter": "agriculture",
                "author": "council"
            },
            {
                "document_type": "regulation",
                "subject_matter": "finance",
                "author": "commission"
            }
        ]
        
        # Generate embeddings
        text_embeddings = embedder.generate_embeddings(sample_texts)
        
        # Fit and encode features
        feature_processor.fit(sample_features)
        # Manually implement batch encoding
        feature_embeddings = np.vstack([
            feature_processor.encode_features(features) for features in sample_features
        ])
        
        # Create sample documents
        sample_documents = []
        for i in range(len(sample_texts)):
            sample_documents.append({
                "id": f"doc{i+1}",
                "text_embedding": text_embeddings[i],
                "feature_embedding": feature_embeddings[i],
                "features": sample_features[i]
            })
    
    # Initialize similarity computer
    start_time = time.time()
    similarity_computer = SimilarityComputer(
        text_weight=0.7,
        categorical_weight=0.3,
        use_faiss=True
    )
    init_time = time.time() - start_time
    logger.info(f"SimilarityComputer initialization time: {init_time:.4f}s")
    
    # Create query embedding and features
    query_text = "Environmental protection measures in the EU"
    query_text_embedding = embedder.generate_embedding(query_text)
    
    query_features = {
        "document_type": "regulation",
        "subject_matter": "environment",
        "author": "commission"
    }
    query_feature_embedding = feature_processor.encode_features(query_features)
    
    # Test similarity computation
    logger.info("Testing similarity computation:")
    
    # Individual similarity
    for i, doc in enumerate(sample_documents):
        start_time = time.time()
        similarity = similarity_computer.compute_similarity(
            query_text_embedding=query_text_embedding,
            query_categorical=query_feature_embedding,
            doc_text_embedding=doc["text_embedding"],
            doc_categorical=doc["feature_embedding"]
        )
        sim_time = time.time() - start_time
        
        logger.info(f"  Document {i+1}: similarity={similarity:.4f}, time={sim_time:.4f}s")
    
    # Batch similarity using FAISS index
    doc_text_embeddings = np.vstack([doc["text_embedding"] for doc in sample_documents])
    doc_feature_embeddings = np.vstack([doc["feature_embedding"] for doc in sample_documents])
    
    # Build index
    start_time = time.time()
    similarity_computer.build_index(
        text_embeddings=doc_text_embeddings,
        categorical_features=doc_feature_embeddings
    )
    build_time = time.time() - start_time
    logger.info(f"Index build time: {build_time:.4f}s")
    
    # Search similar documents
    start_time = time.time()
    indices, similarities = similarity_computer.find_similar(
        query_text_embedding=query_text_embedding,
        query_categorical=query_feature_embedding,
        k=len(sample_documents)
    )
    search_time = time.time() - start_time
    
    logger.info(f"Batch similarity search ({len(sample_documents)} documents): {search_time:.4f}s")
    logger.info(f"Similarities: {[f'{s:.4f}' for s in similarities]}")
    
    # Test with adjusted weights
    logger.info("Testing with adjusted weights:")
    
    # Text-heavy weights - directly update attributes
    start_time = time.time()
    # Store original weights
    original_text_weight = similarity_computer.text_weight
    original_categorical_weight = similarity_computer.categorical_weight
    
    # Update weights directly
    similarity_computer.text_weight = 0.9
    similarity_computer.categorical_weight = 0.1
    
    # Rebuild index with new weights
    similarity_computer.build_index(
        text_embeddings=doc_text_embeddings,
        categorical_features=doc_feature_embeddings
    )
    
    # Find similar documents with new weights
    indices, text_heavy_similarities = similarity_computer.find_similar(
        query_text_embedding=query_text_embedding,
        query_categorical=query_feature_embedding,
        k=len(sample_documents)
    )
    text_heavy_time = time.time() - start_time
    
    logger.info(f"Text-heavy weights (0.9/0.1): {text_heavy_time:.4f}s")
    logger.info(f"Similarities: {[f'{s:.4f}' for s in text_heavy_similarities]}")
    
    # Feature-heavy weights
    start_time = time.time()
    # Update weights directly
    similarity_computer.text_weight = 0.3
    similarity_computer.categorical_weight = 0.7
    
    # Rebuild index with new weights
    similarity_computer.build_index(
        text_embeddings=doc_text_embeddings,
        categorical_features=doc_feature_embeddings
    )
    
    # Find similar documents with new weights
    indices, feature_heavy_similarities = similarity_computer.find_similar(
        query_text_embedding=query_text_embedding,
        query_categorical=query_feature_embedding,
        k=len(sample_documents)
    )
    feature_heavy_time = time.time() - start_time
    
    logger.info(f"Feature-heavy weights (0.3/0.7): {feature_heavy_time:.4f}s")
    logger.info(f"Similarities: {[f'{s:.4f}' for s in feature_heavy_similarities]}")
    
    logger.info("Similarity computer test completed successfully")


def test_document_ranker(sample_documents: Optional[List[Dict[str, Any]]] = None) -> None:
    """
    Test the document ranking component.
    
    Args:
        sample_documents: List of sample documents
    """
    logger.info("Testing DocumentRanker")
    
    # Create default sample documents if none provided
    if not sample_documents:
        # Create sample documents with varying dates
        current_year = date.today().year
        sample_documents = []
        
        for i in range(10):
            year = current_year - i
            sample_documents.append({
                "id": f"doc_{year}",
                "title": f"Sample Document {i+1}",
                "summary": f"This is a sample document from {year}.",
                "keywords": ["sample", "document", f"keyword{i+1}"],
                "features": {
                    "document_type": "regulation" if i % 3 == 0 else ("directive" if i % 3 == 1 else "decision"),
                    "subject_matter": ["environment", "competition", "agriculture", "finance", "digital"][i % 5],
                    "author": ["commission", "parliament", "council", "court"][i % 4],
                    "date": str(year)
                }
            })
    
    # Initialize components
    embedder = BERTEmbedder()
    
    feature_config = {
        "document_type": {"weight": 0.4},
        "subject_matter": {"weight": 0.4},
        "author": {"weight": 0.2}
    }
    feature_processor = FeatureProcessor(feature_config=feature_config)
    
    similarity_computer = SimilarityComputer(
        text_weight=0.7,
        categorical_weight=0.3
    )
    
    # Initialize document ranker
    start_time = time.time()
    ranker = DocumentRanker(
        embedder=embedder,
        feature_processor=feature_processor,
        similarity_computer=similarity_computer
    )
    init_time = time.time() - start_time
    logger.info(f"DocumentRanker initialization time: {init_time:.4f}s")
    
    # Fit feature processor first
    logger.info("Fitting feature processor before processing documents")
    feature_processor.fit([doc["features"] for doc in sample_documents])
    
    # Process sample documents
    start_time = time.time()
    ranker.process_documents(sample_documents)
    process_time = time.time() - start_time
    logger.info(f"Document processing time ({len(sample_documents)} documents): {process_time:.4f}s")
    
    # Create query profile
    query_profile = {
        "summary": "Environmental protection measures in the European Union",
        "keywords": ["environment", "protection", "regulation", "EU"],
        "interests": "Environmental protection and climate change measures in the European Union",
        "features": {
            "document_type": "regulation",
            "subject_matter": "environment",
            "author": "commission"
        }
    }
    
    # Test standard ranking
    logger.info("Testing standard document ranking:")
    start_time = time.time()
    results = ranker.rank_documents(
        query_profile=query_profile,
        top_k=10
    )
    rank_time = time.time() - start_time
    
    logger.info(f"Standard ranking time: {rank_time:.4f}s")
    logger.info(f"Found {len(results)} results")
    
    for i, result in enumerate(results):
        logger.info(f"  {i+1}. Document {result['id']} (Score: {result['similarity']:.4f})")
    
    # Test with feature weights
    logger.info("Testing ranking with adjusted feature weights:")
    feature_weights = {
        "document_type": 0.6,
        "subject_matter": 0.3,
        "author": 0.1
    }
    
    # Update feature weights and rebuild the index
    logger.info("Updating feature weights and rebuilding the index")
    # Directly update the feature_weights attribute
    ranker.feature_processor.feature_weights = feature_weights
    # Call update_weights to handle cache invalidation
    ranker.update_weights()
    
    # Reprocess documents to rebuild the index
    logger.info("Reprocessing documents to rebuild the index")
    ranker.process_documents(sample_documents)
    
    start_time = time.time()
    weighted_results = ranker.rank_documents(
        query_profile=query_profile,
        top_k=10
    )
    weighted_time = time.time() - start_time
    
    logger.info(f"Weighted ranking time: {weighted_time:.4f}s")
    logger.info(f"Found {len(weighted_results)} results with feature weights")
    
    for i, result in enumerate(weighted_results):
        logger.info(f"  {i+1}. Document {result['id']} (Score: {result['similarity']:.4f})")
    
    # Test temporal boost
    logger.info("Testing temporal boost functionality:")
    
    # Custom temporal boost implementation
    def apply_temporal_boost(results, boost_weight, ref_date):
        boosted_results = []
        for result in results:
            doc_id = result['id']
            original_similarity = result['similarity']
            
            # Find document date from our sample documents
            doc_date = None
            for doc in sample_documents:
                if doc['id'] == doc_id and 'features' in doc and 'date' in doc['features']:
                    date_str = doc['features']['date']
                    if len(date_str) == 4:  # Year only
                        doc_date = date(int(date_str), 1, 1)
                    else:  # Try full date format
                        try:
                            doc_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                        except ValueError:
                            pass
                    break
            
            # Calculate temporal score
            temporal_score = 0.0
            if doc_date:
                # Calculate days difference
                days_diff = abs((ref_date - doc_date).days)
                max_days = 365 * 10  # 10 years as max timeframe
                
                # Normalize: newer documents get higher scores
                if days_diff <= max_days:
                    temporal_score = 1.0 - (days_diff / max_days)
                    
            # Combine scores
            final_score = ((1 - boost_weight) * original_similarity) + (boost_weight * temporal_score)
            
            # Create boosted result
            boosted_result = result.copy()
            boosted_result['similarity'] = final_score
            boosted_result['original_similarity'] = original_similarity
            boosted_result['temporal_score'] = temporal_score
            
            boosted_results.append(boosted_result)
        
        # Sort by final score
        boosted_results.sort(key=lambda x: x['similarity'], reverse=True)
        return boosted_results
    
    # Apply temporal boost
    temporal_boost = 0.5
    reference_date = date.today()
    
    start_time = time.time()
    temporal_results = apply_temporal_boost(results, temporal_boost, reference_date)
    temporal_time = time.time() - start_time
    
    logger.info(f"Temporal boost time: {temporal_time:.4f}s")
    logger.info(f"Results with temporal boost ({temporal_boost}):")
    
    for i, result in enumerate(temporal_results):
        logger.info(f"  {i+1}. Document {result['id']} "
                   f"(Score: {result['similarity']:.4f}, "
                   f"Original: {result['original_similarity']:.4f}, "
                   f"Temporal: {result['temporal_score']:.4f})")
    
    logger.info("Document ranker test completed successfully")


def main():
    """Run component tests."""
    parser = argparse.ArgumentParser(description='Test EU Legal Recommender components')
    parser.add_argument('--component', choices=['embedder', 'features', 'similarity', 'ranker', 'all'],
                       default='all', help='Component to test')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Set log level
    logger.setLevel(getattr(logging, args.log_level))
    
    # Run tests
    if args.component in ['embedder', 'all']:
        test_embedder()
        print("\n" + "-"*80 + "\n")
    
    if args.component in ['features', 'all']:
        test_feature_processor()
        print("\n" + "-"*80 + "\n")
    
    if args.component in ['similarity', 'all']:
        test_similarity_computer()
        print("\n" + "-"*80 + "\n")
    
    if args.component in ['ranker', 'all']:
        test_document_ranker()
        print("\n" + "-"*80 + "\n")
    
    print("All tests completed successfully!")


if __name__ == '__main__':
    main()
