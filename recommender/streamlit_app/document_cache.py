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
import os
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

def get_document_from_db(document_id: str) -> Dict:
    """Get document metadata from the SQLite database.
    
    Args:
        document_id (str): The document ID (celex number)
        
    Returns:
        Dict: A dictionary containing document metadata
    """
    # Use the standard path for the database
    db_path = os.path.expanduser("~/DataThesis/eu-legal-recommender/scraper/data/eurlex.db")
    
    if not os.path.exists(db_path):
        logger.warning(f"Database file not found at {db_path}")
        return {}
    
    # Ensure we're working with a valid CELEX number - clean it
    document_id = document_id.strip().upper()
    
    # Skip test documents - debug this carefully
    if document_id.startswith('TEST'):
        logger.info(f"Skipping test document {document_id}")
        # Return empty dict to handle this in the app
        return {}
        
    try:
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
        
        # Get and format date properly
        doc_date = None
        for date_field in ['date', 'publication_date', 'doc_date', 'year']:
            if date_field in doc_dict and doc_dict[date_field]:
                doc_date = doc_dict[date_field]
                logger.info(f"Found date in field {date_field}: {doc_date}")
                break
        
        formatted_date = 'N/A'
        year = 'N/A'
        
        if doc_date:
            try:
                # Check if it's a string date in ISO format (YYYY-MM-DD)
                if isinstance(doc_date, str):
                    # Try to format ISO date (YYYY-MM-DD) to more readable DD/MM/YYYY
                    if '-' in doc_date:
                        parts = doc_date.split('-')
                        if len(parts) >= 1:  # At least get the year
                            year = parts[0]  # First element is year in ISO format
                            if len(parts) == 3:
                                year, month, day = parts
                                formatted_date = f"{day.zfill(2)}/{month.zfill(2)}/{year}"
                            else:
                                formatted_date = doc_date
                    else:
                        formatted_date = doc_date
                        # Try to extract a 4-digit year from the date string
                        import re
                        year_match = re.search(r'\b(19|20)\d{2}\b', doc_date)
                        if year_match:
                            year = year_match.group(0)
                # If it's a datetime object, format it directly
                elif hasattr(doc_date, 'strftime'):
                    formatted_date = doc_date.strftime('%d/%m/%Y')
                    year = doc_date.strftime('%Y')
                # If all else fails, convert to string
                else:
                    formatted_date = str(doc_date)
            except Exception as e:
                logger.warning(f"Error formatting date {doc_date}: {e}")
                formatted_date = str(doc_date)
        
        # Extract all subjects for this document
        subject_matters = []
        try:
            # Query directly using celex_number for more reliability
            cursor.execute(
                """SELECT sm.subject_name FROM subject_matters sm
                   JOIN document_subject_matters dsm ON sm.subject_id = dsm.subject_id
                   JOIN documents d ON dsm.document_id = d.document_id
                   WHERE d.celex_number = ?""",
                (document_id,)
            )
            subject_matters = [row[0] for row in cursor.fetchall()]
        except sqlite3.OperationalError as e:
            logger.warning(f"Subject query failed: {e}")
        
        # If no subject matters were found from the join, try a direct query
        if not subject_matters:
            try:
                # Direct query for subject matters
                cursor.execute(
                    """SELECT sm.subject_name FROM subject_matters sm
                       JOIN document_subject_matters dsm ON sm.subject_id = dsm.subject_id
                       WHERE dsm.document_id = ?""",
                    (doc_dict.get('document_id'),)
                )
                direct_subjects = [row[0] for row in cursor.fetchall()]
                if direct_subjects:
                    subject_matters = direct_subjects
                    logger.info(f"Found {len(subject_matters)} subject matters directly for {document_id}")
            except sqlite3.OperationalError as e:
                logger.warning(f"Direct subject query failed: {e}")
        
        # If still no subject matters, extract from content as fallback
        if not subject_matters and isinstance(doc_dict.get('content', ''), str):
            # Extract any keywords in content that might be subject matters
            content = doc_dict.get('content', '') 
            # More comprehensive list of EU subject matter keywords
            potential_subjects = [
                'Agriculture', 'Fisheries', 'Environment', 'Climate Change', 'Energy', 'Transport',
                'Competition', 'Internal Market', 'Taxation', 'Economic Policy', 'Foreign Policy',
                'Justice', 'Migration', 'Health', 'Consumer Protection', 'Education', 
                'Culture', 'Employment', 'Social Policy', 'Research', 'Technology',
                'Digital', 'Food Safety', 'Financial Services', 'Trade', 'Industrial Policy',
                'Regional Development', 'Human Rights', 'Security', 'Defence', 'Customs',
                'Banking', 'Insurance', 'Telecommunications', 'Data Protection', 'Intellectual Property'
            ]
            
            # Extract potential subject matters from content
            extracted_subjects = [subject for subject in potential_subjects 
                               if subject.lower() in content.lower()]
            
            # Use these if we found some
            if extracted_subjects:
                subject_matters = extracted_subjects[:5]  # Limit to top 5
                logger.info(f"Extracted {len(subject_matters)} subject matters from content for {document_id}")
            # If still nothing, add default subject matters for legal documents
            elif document_id.startswith('3'):
                subject_matters = ['Legal Affairs']  # Default for legal documents
            elif document_id.startswith('5'):
                subject_matters = ['External Relations']  # Default for international agreements
            else:
                subject_matters = ['EU Legislation']  # Generic default
        
        # ALWAYS assign reasonable defaults for critical fields
        
        # 1. Make sure to provide a title even if NULL in database
        title = doc_dict.get('title')
        if not title or title.strip() == '':
            # Generate a title based on document ID if missing
            title = f"EU Legal Document {document_id}"
            
            # Try to be more specific based on CELEX number format
            if document_id.startswith('3'):
                title = f"EU Regulation {document_id}"
            elif document_id.startswith('1'):
                title = f"EU Treaty {document_id}"
            elif document_id.startswith('2'):
                title = f"EU Directive {document_id}"
            elif document_id.startswith('4'):
                title = f"EU Agreement {document_id}"
        
        # 2. ALWAYS ensure subject matters is never empty
        if not subject_matters or len(subject_matters) == 0:
            # If still empty after all previous attempts, add generic EU legal areas
            if document_id.startswith('3'):
                subject_matters = ["Single Market", "EU Regulation", "Legal Affairs"]
            elif document_id.startswith('2'):
                subject_matters = ["EU Directive", "Harmonization", "Legal Affairs"]
            else:
                subject_matters = ["EU Law", "Single Market", "Legal Affairs"]
        
        # 3. ALWAYS ensure date is not N/A
        if formatted_date == 'N/A' or not formatted_date:
            # Try to extract from CELEX number format (3YYYYNXXXX)
            if len(document_id) >= 5 and document_id[1:5].isdigit() and 1950 <= int(document_id[1:5]) <= 2030:
                year = document_id[1:5]
                formatted_date = f"Year: {year}"
                logger.info(f"Extracted year from CELEX: {year}")
            else:
                # Try to get year from the document_id field directly
                doc_id_field = doc_dict.get('document_id', '')
                if isinstance(doc_id_field, str) and len(doc_id_field) >= 8:
                    # Try to extract a year string
                    import re
                    year_match = re.search(r'(19|20)\d{2}', doc_id_field)
                    if year_match:
                        year = year_match.group(0)
                        formatted_date = f"Year: {year}"
                        logger.info(f"Extracted year from document_id: {year}")
                    else:
                        # Use a default based on when it was likely published
                        current_year = datetime.now().year
                        formatted_date = f"{current_year} (Estimated)"
                else:
                    # Use current year as a last resort
                    current_year = datetime.now().year
                    formatted_date = f"{current_year} (Estimated)"
        
        # Prepare metadata in format compatible with the app - using actual column names from logs
        # INCLUDE MANY REDUNDANT FIELDS to ensure we always have data even in different database formats
        eurlex_url = f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{document_id}"
        
        metadata = {
            # Primary fields
            'title': title,
            'celex_number': document_id,
            'document_type': doc_dict.get('document_type', 'N/A'),
            'date': formatted_date,  # Always use our formatted date
            'year': year,
            'subject_matters': subject_matters,
            'url': doc_dict.get('url', eurlex_url),
            'summary': doc_dict.get('summary', 'No summary available.'),
            
            # Redundant fields for robustness
            'doc_date': formatted_date,  # Duplicate to handle different field names
            'publication_date': formatted_date,  # Duplicate to handle different field names
            'document_date': formatted_date,  # Duplicate to handle different field names
            'eur_lex_url': eurlex_url,  # Extra URL field 
            'subjects': subject_matters,  # Duplicate to handle different field names
            'categories': subject_matters,  # Duplicate to handle different field names
            'document_id': document_id  # Keep the original ID
        }
        
        # Close the connection
        conn.close()
        
        # Return the metadata directly instead of nesting it
        logger.info(f"Returning metadata for {document_id}: {metadata}")
        return metadata
        
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
