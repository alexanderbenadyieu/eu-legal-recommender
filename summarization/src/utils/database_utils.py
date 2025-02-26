import sqlite3
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class Document:
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

def load_documents(db_path: str, limit: Optional[int] = None, filter_tier1: bool = False) -> List[Document]:
    """
    Load documents from the SQLite database.
    
    Args:
        db_path: Path to the SQLite database
        limit: Optional limit on number of documents to retrieve
        filter_tier1: If True, only load Tier 1 documents (≤ 600 words)
        
    Returns:
        List of Document objects
    """
    logger.info(f"Loading documents from {db_path}")
    with DatabaseConnection(db_path) as conn:
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
        logger.debug(f"Base query: {query}")
        
        if filter_tier1:
            query += " WHERE total_words <= 600 AND total_words > 0"
            logger.info("Filtering for Tier 1 documents (≤ 600 words)")
        
        if limit:
            query += f" LIMIT {limit}"
            
        cursor = conn.execute(query)
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
