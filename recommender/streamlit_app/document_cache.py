"""
Document caching module for the EU Legal Recommender Streamlit app.

This module provides functions for caching document data and recommendations
to improve performance by reducing redundant API calls.
"""
import json
from pathlib import Path
import logging
import pickle
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import sqlite3

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cache file paths
CACHE_DIR = Path(__file__).parent / "cache"
DOCUMENT_CACHE_FILE = CACHE_DIR / "document_cache.pkl"
RECOMMENDATION_CACHE_FILE = CACHE_DIR / "recommendation_cache.pkl"

# Cache expiration time (in hours)
CACHE_EXPIRY = 24  # 24 hours

def ensure_cache_dir():
    """Ensure the cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

def load_cache(cache_file: Path) -> Dict:
    """Load a cache file or return an empty dict if not exists/expired."""
    if not cache_file.exists():
        return {}
    
    try:
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
            
        # Check if cache metadata exists and is still valid
        if 'metadata' in cache and 'last_updated' in cache['metadata']:
            last_updated = cache['metadata']['last_updated']
            expiry_time = last_updated + timedelta(hours=CACHE_EXPIRY)
            
            if datetime.now() > expiry_time:
                logger.info(f"Cache expired ({cache_file.name}), creating fresh cache")
                return {'metadata': {'last_updated': datetime.now()}, 'data': {}}
            
        return cache
    except Exception as e:
        logger.warning(f"Failed to load cache file {cache_file}: {str(e)}")
        return {'metadata': {'last_updated': datetime.now()}, 'data': {}}

def save_cache(cache: Dict, cache_file: Path) -> None:
    """Save cache data to file."""
    ensure_cache_dir()
    
    try:
        # Update metadata
        if 'metadata' not in cache:
            cache['metadata'] = {}
        cache['metadata']['last_updated'] = datetime.now()
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
            
        logger.info(f"Cache saved to {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to save cache to {cache_file}: {str(e)}")

def get_document_from_cache(document_id: str) -> Optional[Dict]:
    """Get a document from cache by its ID."""
    cache = load_cache(DOCUMENT_CACHE_FILE)
    return cache.get('data', {}).get(document_id)

def add_document_to_cache(document_id: str, document_data: Dict) -> None:
    """Add a document to the cache."""
    cache = load_cache(DOCUMENT_CACHE_FILE)
    
    if 'data' not in cache:
        cache['data'] = {}
        
    cache['data'][document_id] = document_data
    save_cache(cache, DOCUMENT_CACHE_FILE)

def get_recommendations_from_cache(cache_key: str) -> Optional[List[Dict]]:
    """Get recommendations from cache by key.
    
    The cache_key should be a string that uniquely identifies the recommendation request,
    for example, a combination of query/document_id/profile and other parameters.
    """
    cache = load_cache(RECOMMENDATION_CACHE_FILE)
    return cache.get('data', {}).get(cache_key)

def add_recommendations_to_cache(cache_key: str, recommendations: List[Dict]) -> None:
    """Add recommendations to the cache."""
    cache = load_cache(RECOMMENDATION_CACHE_FILE)
    
    if 'data' not in cache:
        cache['data'] = {}
        
    cache['data'][cache_key] = recommendations
    save_cache(cache, RECOMMENDATION_CACHE_FILE)

def generate_cache_key(params: Dict) -> str:
    """Generate a cache key from a dictionary of parameters."""
    # Sort the keys to ensure consistent ordering
    sorted_keys = sorted(params.keys())
    key_parts = []
    
    for key in sorted_keys:
        value = params[key]
        if isinstance(value, (list, dict)):
            # Convert complex types to sorted string representation
            value = json.dumps(value, sort_keys=True)
        # Make sure we differentiate between different profiles even if mode is the same
        key_parts.append(f"{key}:{value}")
    
    # Add timestamp to force refresh on cold start (optional, remove if cached results needed)
    # import time
    # key_parts.append(f"timestamp:{int(time.time())}")
    
    return "|".join(key_parts)

def clear_cache() -> None:
    """Clear all caches."""
    logger.info("Clearing all caches")
    _clear_document_cache()
    _clear_recommendation_cache()

def _clear_document_cache() -> None:
    try:
        if DOCUMENT_CACHE_FILE.exists():
            DOCUMENT_CACHE_FILE.unlink()
        logger.info("Document cache cleared successfully")
    except Exception as e:
        logger.warning(f"Failed to clear document cache: {str(e)}")

def _clear_recommendation_cache() -> None:
    try:
        if RECOMMENDATION_CACHE_FILE.exists():
            RECOMMENDATION_CACHE_FILE.unlink()
        logger.info("Recommendation cache cleared successfully")
    except Exception as e:
        logger.warning(f"Failed to clear recommendation cache: {str(e)}")

def get_document_from_database(document_id: str) -> Dict:
    """Get document metadata from SQLite database.
    
    Args:
        document_id: The document ID (CELEX number) to retrieve
        
    Returns:
        Dictionary containing document metadata or empty dict if not found
    """
    try:
        # Path to the SQLite database
        db_path = Path("/Users/alexanderbenady/DataThesis/eu-legal-recommender/scraper/data/eurlex.db")
        
        if not db_path.exists():
            logger.warning(f"Database file not found at {db_path}")
            return {}
            
        # Connect to the database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        cursor = conn.cursor()
        
        # First, let's check the table structure to ensure we use the right columns
        cursor.execute("PRAGMA table_info(documents)")
        columns = [col[1] for col in cursor.fetchall()]
        logger.info(f"Database document columns: {columns}")
        
        # Query for document based on CELEX number
        # Use only celex_number as it's the standard identifier
        cursor.execute(
            "SELECT * FROM documents WHERE celex_number = ? LIMIT 1", 
            (document_id,)
        )
        
        document = cursor.fetchone()
        
        if not document:
            logger.warning(f"Document {document_id} not found in database")
            return {}
            
        # Convert row to dictionary 
        doc_dict = dict(document)
        
        # Get additional metadata from other tables
        
        # Check if subject matters tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND (name='document_subject_matters' OR name='subject_matters')")
        subject_tables = [row[0] for row in cursor.fetchall()]
        subject_matters = []
        
        if len(subject_tables) == 2:  # Both tables exist
            try:
                # Based on the actual schema detected in the logs
                cursor.execute(
                    """SELECT sm.subject_name 
                       FROM document_subject_matters dsm 
                       JOIN subject_matters sm ON dsm.subject_id = sm.subject_id 
                       WHERE dsm.document_id = ?""", 
                    (doc_dict.get('document_id'),)
                )
                subject_matters = [row[0] for row in cursor.fetchall()]
            except sqlite3.OperationalError as e:
                # Handle case where schema is different
                logger.warning(f"Could not query subject matters: {e}")
                # Try an alternative approach if the first query fails
                try:
                    cursor.execute("SELECT * FROM document_subject_matters LIMIT 1")
                    cols = [col[0] for col in cursor.description]
                    logger.info(f"document_subject_matters columns: {cols}")
                    
                    cursor.execute("SELECT * FROM subject_matters LIMIT 1")
                    cols = [col[0] for col in cursor.description]
                    logger.info(f"subject_matters columns: {cols}")
                except Exception as e2:
                    logger.error(f"Error examining subject_matters tables: {e2}")
        else:
            logger.warning(f"Subject matter tables not found. Available tables: {subject_tables}")
            
        # Check if summaries table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='summaries'")
        summary_table_exists = cursor.fetchone() is not None
        summary = ""
        
        if summary_table_exists:
            try:
                # Get summary if available
                cursor.execute(
                    "SELECT summary FROM summaries WHERE document_id = ? LIMIT 1",
                    (doc_dict.get('id'),)
                )
                summary_row = cursor.fetchone()
                summary = summary_row[0] if summary_row else ""
            except sqlite3.OperationalError as e:
                logger.warning(f"Could not query summaries: {e}")
                try:
                    cursor.execute("SELECT * FROM summaries LIMIT 1")
                    cols = [col[0] for col in cursor.description]
                    logger.info(f"summaries columns: {cols}")
                except Exception as e2:
                    logger.error(f"Error examining summaries table: {e2}")
        else:
            logger.warning("Summaries table not found")
            
        # If we couldn't get a summary from the database, try to extract one from text
        if not summary and 'text' in doc_dict and doc_dict['text']:
            # Take first 300 characters of text as summary
            summary = doc_dict['text'][:300] + '...'
        
        # Prepare metadata in format compatible with the app - using actual column names from logs
        metadata = {
            'id': doc_dict.get('document_id'),
            'celex_number': doc_dict.get('celex_number'),
            'title': doc_dict.get('title'),
            'document_type': 'EU Document',  # Default type since it might not be in the schema
            'date': doc_dict.get('date_of_document') or doc_dict.get('date_of_effect'),
            'year': doc_dict.get('date_of_document', '').split('-')[0] if doc_dict.get('date_of_document') else '',
            'url': doc_dict.get('html_url') or doc_dict.get('pdf_url') or f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{document_id}",
            'subject_matters': subject_matters,
            'text': doc_dict.get('content', ''),
            'summary': doc_dict.get('summary', summary)
        }
        
        # Close the connection
        conn.close()
        
        return {'metadata': metadata}
        
    except Exception as e:
        logger.error(f"Error retrieving document from database: {str(e)}")
        return {}

def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about the current cache."""
    doc_cache = load_cache(DOCUMENT_CACHE_FILE)
    rec_cache = load_cache(RECOMMENDATION_CACHE_FILE)
    
    doc_count = len(doc_cache.get('data', {}))
    rec_count = len(rec_cache.get('data', {}))
    
    doc_last_updated = doc_cache.get('metadata', {}).get('last_updated', 'Never')
    rec_last_updated = rec_cache.get('metadata', {}).get('last_updated', 'Never')
    
    return {
        'document_count': doc_count,
        'recommendation_count': rec_count,
        'document_last_updated': doc_last_updated,
        'recommendation_last_updated': rec_last_updated
    }
