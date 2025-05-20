"""Database utilities for the summarization module.

This module provides database-related functionality for the summarization pipeline,
including data structures and helper functions for interacting with the database.
"""

import sqlite3
import logging
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Import from project root by adding the project root to the Python path
import os
import sys

# Add the project root to the Python path to access database_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))) 
from database_utils import get_db_connection, get_document_by_celex, get_document_sections

logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Represents a legal document in the EU Legal Recommender system.
    
    This class stores document metadata and summarization results.
    
    Attributes:
        id: Database identifier for the document
        celex_number: Unique CELEX identifier for EU legal documents
        html_url: URL to the HTML version of the document
        total_words: Total word count of the document
        summary: Generated summary text
        summary_word_count: Word count of the summary
    """
    id: int
    celex_number: str
    html_url: Optional[str]
    total_words: Optional[int]
    summary: Optional[str]
    summary_word_count: Optional[int]
    compression_ratio: Optional[float]

class DatabaseConnection:
    def __init__(self, db_path: str):
        self.db_path = db_path
        
    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self.conn
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()

def load_documents(db_path: str = None, limit: Optional[int] = None, filter_tier1: bool = False, tier: Optional[int] = None, db_type: str = 'consolidated') -> List[Document]:
    """
    Load documents from the database.
    
    Args:
        db_path: Path to the SQLite database (legacy parameter, use db_type instead)
        limit: Optional limit on number of documents to retrieve
        filter_tier1: If True, only load Tier 1 documents (≤ 600 words)
        tier: Optional tier to filter documents (1, 2, 3, or 4)
        db_type: Type of database to use ('consolidated' or 'legacy')
        
    Returns:
        List of Document objects
    """
    logger.info(f"Loading documents from {db_type} database")
    
    # Connect to the appropriate database
    if db_type == 'consolidated':
        conn = get_db_connection(db_type=db_type)
        conn.row_factory = sqlite3.Row
        
        # Determine tier boundaries if tier is specified
        if tier:
            tier_boundaries = {
                1: (0, 600),
                2: (601, 2500),
                3: (2501, 20000),
                4: (20001, 1000000)  # Large upper bound for tier 4
            }
            min_words, max_words = tier_boundaries.get(tier, (0, 1000000))
            
            query = """
                SELECT 
                    document_id as id,
                    celex_number,
                    html_url,
                    word_count as total_words,
                    summary,
                    summary_word_count,
                    compression_ratio
                FROM documents
                WHERE word_count BETWEEN ? AND ?
            """
            params = [min_words, max_words]
        elif filter_tier1:
            query = """
                SELECT 
                    document_id as id,
                    celex_number,
                    html_url,
                    word_count as total_words,
                    summary,
                    summary_word_count,
                    compression_ratio
                FROM documents
                WHERE word_count <= 600 AND word_count > 0
            """
            params = []
        else:
            query = """
                SELECT 
                    document_id as id,
                    celex_number,
                    html_url,
                    word_count as total_words,
                    summary,
                    summary_word_count,
                    compression_ratio
                FROM documents
            """
            params = []
    else:
        # Legacy database connection
        if not db_path:
            raise ValueError("db_path must be provided for legacy database")
            
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        query = """
            SELECT 
                id,
                celex_number,
                html_url,
                total_words,
                summary,
                summary_word_count,
                compression_ratio
            FROM processed_documents
        """
        params = []
        
        if filter_tier1:
            query += " WHERE total_words <= 600 AND total_words > 0"
        elif tier:
            tier_boundaries = {
                1: (0, 600),
                2: (601, 2500),
                3: (2501, 20000),
                4: (20001, 1000000)  # Large upper bound for tier 4
            }
            min_words, max_words = tier_boundaries.get(tier, (0, 1000000))
            query += " WHERE total_words BETWEEN ? AND ?"
            params = [min_words, max_words]
    
    if limit:
        query += " LIMIT ?"
        params.append(limit)
        
    cursor = conn.execute(query, params)
    rows = cursor.fetchall()
    logger.info(f"Found {len(rows)} documents")
    
    documents = []
    for row in rows:
        doc = Document(
            id=row['id'],
            celex_number=row['celex_number'],
            html_url=row['html_url'],
            total_words=row['total_words'],
            summary=row['summary'],
            summary_word_count=row['summary_word_count'],
            compression_ratio=row['compression_ratio']
        )
        documents.append(doc)
        
    return documents
