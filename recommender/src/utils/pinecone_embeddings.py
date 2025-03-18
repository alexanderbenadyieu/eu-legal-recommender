"""
Utility script to generate embeddings for document summaries and store them in Pinecone.
"""
import os
import sys
import json
import logging
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# Import our database connector
from utils.db_connector import get_connector, DocumentDBConnector

# Add parent directory to path to import from models
sys.path.append(str(Path(__file__).parent.parent))
from models.embeddings import BERTEmbedder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('embeddings.log')
    ]
)
logger = logging.getLogger(__name__)

class PineconeEmbeddingManager:
    """Manage document embeddings in Pinecone."""
    
    def __init__(
        self,
        api_key: str,
        environment: str,
        index_name: str,
        dimension: int = 512,  # Default for legal-bert-small-uncased
        embedder_model: str = 'nlpaueb/legal-bert-small-uncased',
        metric: str = 'cosine'
    ):
        """
        Initialize the Pinecone embedding manager.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Name of the Pinecone index
            dimension: Dimension of the embeddings
            embedder_model: Name of the sentence-transformer model to use
            metric: Distance metric for similarity search
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        
        # Initialize embedder
        self.embedder = BERTEmbedder(model_name=embedder_model)
        
        # Initialize Pinecone
        self._init_pinecone()
        
    def _init_pinecone(self) -> None:
        """Initialize Pinecone client and create or connect to index."""
        logger.info(f"Initializing Pinecone")
        self.pc = Pinecone(api_key=self.api_key)
        
        # Check if index exists
        if self.index_name not in self.pc.list_indexes().names():
            logger.info(f"Index {self.index_name} does not exist. Creating new index with dimension {self.dimension}")
            # Create a new index with the specified dimension
            self.create_index()
            logger.info(f"Created new Pinecone index: {self.index_name}")
        else:
            logger.info(f"Found existing Pinecone index: {self.index_name}")
        
        # Connect to index
        self.index = self.pc.Index(self.index_name)
        logger.info(f"Connected to Pinecone index: {self.index_name}")
        
    def delete_all_embeddings(self) -> None:
        """Delete all embeddings from the Pinecone index."""
        logger.info(f"Deleting all embeddings from index: {self.index_name}")
        # Use delete_all method to remove all vectors
        self.index.delete(delete_all=True)
        logger.info("All embeddings deleted successfully")
    
    def create_index(self) -> None:
        """Create a new Pinecone index with the specified dimension."""
        logger.info(f"Creating new Pinecone index: {self.index_name} with dimension {self.dimension}")
        try:
            # Create index with the specified dimension
            # For AWS free tier in us-east-1 (as per latest Pinecone documentation)
            from pinecone import ServerlessSpec
            
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            logger.info(f"Created new Pinecone index: {self.index_name} with dimension {self.dimension}")
        except Exception as e:
            logger.error(f"Error creating Pinecone index: {e}")
            raise
    
    def delete_index(self) -> None:
        """Delete the Pinecone index."""
        logger.info(f"Deleting Pinecone index: {self.index_name}")
        try:
            if self.index_name in self.pc.list_indexes().names():
                self.pc.delete_index(self.index_name)
                logger.info(f"Deleted Pinecone index: {self.index_name}")
            else:
                logger.warning(f"Index {self.index_name} does not exist, nothing to delete")
        except Exception as e:
            logger.error(f"Error deleting Pinecone index: {e}")
            raise
    
    def delete_and_recreate_index(self) -> None:
        """Delete the entire Pinecone index and recreate it with the current dimension."""
        logger.info(f"Deleting and recreating Pinecone index with dimension {self.dimension}")
        try:
            # Delete the index if it exists
            self.delete_index()
            
            # Create a new index with the specified dimension
            self.create_index()
            
            # Connect to the new index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to new Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Error recreating Pinecone index: {e}")
            raise
    
    def fetch_tier_documents(
        self, 
        db_connector: DocumentDBConnector, 
        tier: int
    ) -> List[Dict[str, Any]]:
        """
        Fetch documents of a specific tier using the provided database connector.
        
        Args:
            db_connector: Database connector instance
            tier: Tier to fetch (1, 2, 3, or 4)
            
        Returns:
            List of document dictionaries
        """
        logger.info(f"Fetching tier {tier} documents from database")
        
        # Use the connector to fetch documents
        documents = db_connector.fetch_tier_documents(tier)
        
        logger.info(f"Retrieved {len(documents)} tier {tier} documents")
        return documents
    
    def generate_document_embeddings(
        self, 
        documents: List[Dict[str, Any]]
    ) -> List[Tuple[str, np.ndarray, Dict[str, Any]]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of tuples (id, embedding, metadata)
        """
        logger.info(f"Generating embeddings for {len(documents)} documents")
        
        embeddings_data = []
        for doc in tqdm(documents, desc="Generating embeddings"):
            # Skip if summary is missing
            if not doc['summary']:
                logger.warning(f"Skipping document {doc['id']} - missing summary")
                continue
                
            # Generate combined embedding from summary and keywords
            keywords = doc.get('keywords', [])
            if not keywords:
                logger.warning(f"Document {doc['id']} has no keywords, using summary only")
                # Generate embedding from summary only
                embedding = self.embedder.generate_embeddings(
                    [doc['summary']], 
                    show_progress=False
                )[0]
                # Normalize
                embedding = embedding / np.linalg.norm(embedding)
            else:
                # Generate combined embedding
                embedding = self.embedder.combine_text_features(
                    summary=doc['summary'],
                    keywords=keywords
                )
            
            # Extract title from summary if not available
            title = doc.get('title', '')
            if not title and doc['summary']:
                # Try to extract title from the first line of the summary
                summary_lines = doc['summary'].strip().split('\n')
                if len(summary_lines) > 0:
                    # Use the first non-empty line as the title
                    for line in summary_lines:
                        if line.strip():
                            title = line.strip()
                            break
                
                # If still no title, try to use the first sentence
                if not title:
                    first_sentence = doc['summary'].split('.')[0]
                    if len(first_sentence) > 10:  # Ensure it's a meaningful sentence
                        title = first_sentence + '.'
                
                # If still no title, use a formatted version of the CELEX number
                if not title:
                    title = f"Document {doc['celex_number']}"
            
            # Prepare base metadata - include title, celex_number, summary, keywords, and tier
            base_metadata = {
                'celex_number': doc['celex_number'],
                'title': title,
                'summary': doc['summary'],
                'keywords': ','.join(keywords) if keywords else '',
                'tier': doc.get('tier', 1),  # Default to tier 1 if not specified
                'num_keywords': len(keywords)
            }
            
            # Infer categorical features from content
            categorical_features = {}
            
            # Infer document_type from CELEX number
            celex = doc['celex_number']
            if 'R' in celex:
                categorical_features['document_type'] = ['regulation']
            elif 'L' in celex:
                categorical_features['document_type'] = ['directive']
            elif 'D' in celex:
                categorical_features['document_type'] = ['decision']
            elif 'Q' in celex:
                categorical_features['document_type'] = ['recommendation']
            else:
                categorical_features['document_type'] = ['other']
            
            # Infer subject_matter from keywords or summary
            keywords_text = ' '.join(keywords).lower() if keywords else ''
            summary_text = doc['summary'].lower() if doc['summary'] else ''
            
            subject_matters = []
            
            # Check for data protection related content
            if 'data protection' in keywords_text or 'data protection' in summary_text or 'gdpr' in keywords_text or 'gdpr' in summary_text:
                subject_matters.append('data protection')
            
            # Check for privacy related content
            if 'privacy' in keywords_text or 'privacy' in summary_text:
                subject_matters.append('privacy')
            
            # Check for financial related content
            if any(term in keywords_text or term in summary_text for term in ['financial', 'banking', 'credit', 'loan', 'investment']):
                subject_matters.append('financial')
            
            # Check for environmental related content
            if any(term in keywords_text or term in summary_text for term in ['environment', 'climate', 'emission', 'pollution']):
                subject_matters.append('environment')
            
            # Check for health related content
            if any(term in keywords_text or term in summary_text for term in ['health', 'medical', 'medicine', 'patient']):
                subject_matters.append('health')
            
            # If we found subject matters, add them
            if subject_matters:
                categorical_features['subject_matter'] = subject_matters
            else:
                # Default subject matter
                categorical_features['subject_matter'] = ['general']
            
            # Add author (always EU for these documents)
            categorical_features['author'] = ['European Union']
            
            # Add form (infer from content or default to 'legal text')
            if 'regulation' in summary_text.lower():
                categorical_features['form'] = ['regulation']
            elif 'directive' in summary_text.lower():
                categorical_features['form'] = ['directive']
            elif 'decision' in summary_text.lower():
                categorical_features['form'] = ['decision']
            else:
                categorical_features['form'] = ['legal text']
            
            # Store categorical features in metadata
            for feature_name, feature_values in categorical_features.items():
                base_metadata[feature_name] = feature_values[0] if len(feature_values) == 1 else feature_values
            
            # Convert to JSON string for storage in metadata
            base_metadata['categorical_features'] = json.dumps(categorical_features)
            
            # Use base_metadata as our metadata
            metadata = base_metadata
            
            # Use celex_number as the ID for Pinecone
            embeddings_data.append((doc['celex_number'], embedding, metadata))
        
        logger.info(f"Generated embeddings for {len(embeddings_data)} documents")
        return embeddings_data
    
    def upload_to_pinecone(
        self, 
        embeddings_data: List[Tuple[str, np.ndarray, Dict[str, Any]]],
        batch_size: int = 100
    ) -> None:
        """
        Upload embeddings to Pinecone.
        
        Args:
            embeddings_data: List of tuples (id, embedding, metadata)
            batch_size: Number of embeddings to upload in each batch
        """
        logger.info(f"Uploading {len(embeddings_data)} embeddings to Pinecone")
        
        # Process in batches
        for i in tqdm(range(0, len(embeddings_data), batch_size), desc="Uploading to Pinecone"):
            batch = embeddings_data[i:i+batch_size]
            
            # Format for Pinecone
            vectors = [
                {
                    'id': str(doc_id),
                    'values': embedding.tolist(),
                    'metadata': metadata
                }
                for doc_id, embedding, metadata in batch
            ]
            
            # Upload batch
            self.index.upsert(vectors=vectors)
        
        logger.info(f"Successfully uploaded {len(embeddings_data)} embeddings to Pinecone")
    
    def process_tier_documents(
        self, 
        db_connector: DocumentDBConnector, 
        tier: int,
        batch_size: int = 100
    ) -> None:
        """
        Process documents of a specific tier and upload to Pinecone.
        
        Args:
            db_connector: Database connector instance
            tier: Tier to process (1, 2, 3, or 4)
            batch_size: Batch size for Pinecone uploads
        """
        try:
            # Fetch documents
            documents = self.fetch_tier_documents(db_connector, tier)
            
            # Generate embeddings
            embeddings_data = self.generate_document_embeddings(documents)
            
            # Upload to Pinecone
            self.upload_to_pinecone(embeddings_data, batch_size)
            
            logger.info(f"Completed processing tier {tier} documents")
        finally:
            # Always close the database connection
            db_connector.close()

def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate and store document embeddings in Pinecone")
    parser.add_argument("--db-type", type=str, choices=["sqlite", "api"], default="sqlite", 
                        help="Type of database connector to use")
    parser.add_argument("--db-path", type=str, help="Path to the SQLite database (required for sqlite db-type)")
    parser.add_argument("--api-url", type=str, help="URL for the API (required for api db-type)")
    parser.add_argument("--db-api-key", type=str, help="API key for the database API (optional for api db-type)")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4], default=1, help="Document tier to process")
    parser.add_argument("--pinecone-api-key", type=str, required=True, help="Pinecone API key")
    parser.add_argument("--index-name", type=str, default="eu-legal-documents", help="Pinecone index name")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for Pinecone uploads")
    
    args = parser.parse_args()
    
    # Validate arguments based on db-type
    if args.db_type == "sqlite" and not args.db_path:
        parser.error("--db-path is required when using sqlite db-type")
    elif args.db_type == "api" and not args.api_url:
        parser.error("--api-url is required when using api db-type")
    
    # Initialize database connector
    if args.db_type == "sqlite":
        db_connector = get_connector("sqlite", db_path=args.db_path)
    else:  # api
        db_connector = get_connector("api", api_url=args.api_url, api_key=args.db_api_key)
    
    # Initialize manager
    manager = PineconeEmbeddingManager(
        api_key=args.pinecone_api_key,
        environment="gcp-starter",  # Using serverless GCP starter environment
        index_name=args.index_name
    )
    
    # Process documents
    manager.process_tier_documents(
        db_connector=db_connector,
        tier=args.tier,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
