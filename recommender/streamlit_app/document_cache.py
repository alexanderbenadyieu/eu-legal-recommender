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
        key_parts.append(f"{key}:{value}")
    
    return "|".join(key_parts)

def clear_cache() -> None:
    """Clear all cache files."""
    try:
        if DOCUMENT_CACHE_FILE.exists():
            DOCUMENT_CACHE_FILE.unlink()
        if RECOMMENDATION_CACHE_FILE.exists():
            RECOMMENDATION_CACHE_FILE.unlink()
        logger.info("Cache cleared successfully")
    except Exception as e:
        logger.warning(f"Failed to clear cache: {str(e)}")

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
