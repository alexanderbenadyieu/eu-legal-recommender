"""
Database connector module for fetching document data from various sources.
This abstraction allows for easy switching between different database sources.
"""
import os
import sqlite3
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentDBConnector(ABC):
    """Abstract base class for document database connectors."""
    
    @abstractmethod
    def fetch_tier_documents(self, tier: int) -> List[Dict[str, Any]]:
        """
        Fetch documents of a specific tier from the database.
        
        Args:
            tier: Tier to fetch (1, 2, 3, or 4)
            
        Returns:
            List of document dictionaries
        """
        pass
    
    @abstractmethod
    def fetch_document_keywords(self, document_id: str) -> List[str]:
        """
        Fetch keywords for a specific document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            List of keywords
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
        pass

class SQLiteDocumentConnector(DocumentDBConnector):
    """Connector for SQLite document database."""
    
    def __init__(self, db_path: str, db_type: str = 'consolidated'):
        """
        Initialize the SQLite connector.
        
        Args:
            db_path: Path to the SQLite database
            db_type: Type of database structure ('consolidated' or 'legacy')
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")
            
        self.db_path = db_path
        self.db_type = db_type
        self.conn = sqlite3.connect(db_path)
        logger.info(f"Connected to SQLite database: {db_path} (type: {db_type})")
    
    def fetch_tier_documents(self, tier: int) -> List[Dict[str, Any]]:
        """
        Fetch documents of a specific tier from the SQLite database.
        
        Args:
            tier: Tier to fetch (1, 2, 3, or 4)
            
        Returns:
            List of document dictionaries with all categorical data
        """
        logger.info(f"Fetching tier {tier} documents from SQLite database (type: {self.db_type})")
        
        cursor = self.conn.cursor()
        
        # Query for documents with summaries based on database type
        if self.db_type == 'consolidated':
            cursor.execute(
                """
                SELECT d.document_id, d.celex_number, d.title, d.summary, d.total_words, 
                       d.summary_word_count, d.date_of_document, d.date_of_effect, 
                       d.date_of_end_validity, f.form_name, rb.body_name
                FROM documents d
                LEFT JOIN forms f ON d.form_id = f.form_id
                LEFT JOIN responsible_bodies rb ON d.responsible_body_id = rb.responsible_body_id
                WHERE d.tier = ? AND d.summary IS NOT NULL
                """,
                (tier,)
            )
        else:  # legacy
            cursor.execute(
                """
                SELECT id, celex_number, NULL as title, summary, total_words, summary_word_count,
                       NULL as date_of_document, NULL as date_of_effect, NULL as date_of_end_validity,
                       NULL as form_name, NULL as body_name
                FROM processed_documents
                WHERE tier = ? AND summary IS NOT NULL
                """,
                (tier,)
            )
        
        documents = []
        for row in cursor.fetchall():
            if self.db_type == 'consolidated':
                doc_id, celex, title, summary, total_words, summary_words, date_doc, date_effect, date_end, form, body = row
                
                # Fetch all categorical data for this document
                keywords = self.fetch_document_keywords(doc_id)
                subject_matters = self.fetch_document_subject_matters(doc_id)
                eurovoc_descriptors = self.fetch_document_eurovoc_descriptors(doc_id)
                authors = self.fetch_document_authors(doc_id)
                directory_codes = self.fetch_document_directory_codes(doc_id)
                
                documents.append({
                    'id': doc_id,
                    'celex_number': celex,
                    'title': title,
                    'summary': summary,
                    'keywords': keywords,
                    'total_words': total_words,
                    'summary_word_count': summary_words,
                    'date_of_document': date_doc,
                    'date_of_effect': date_effect,
                    'date_of_end_validity': date_end,
                    'form': form,
                    'responsible_body': body,
                    'subject_matters': subject_matters,
                    'eurovoc_descriptors': eurovoc_descriptors,
                    'authors': authors,
                    'directory_codes': directory_codes,
                    'tier': tier
                })
            else:  # legacy
                doc_id, celex, title, summary, total_words, summary_words = row[:6]
                
                # Fetch keywords for this document
                keywords = self.fetch_document_keywords(doc_id)
                
                documents.append({
                    'id': doc_id,
                    'celex_number': celex,
                    'title': '',
                    'summary': summary,
                    'keywords': keywords,
                    'total_words': total_words,
                    'summary_word_count': summary_words,
                    'subject_matters': [],
                    'eurovoc_descriptors': [],
                    'authors': [],
                    'directory_codes': [],
                    'tier': tier
                })
        
        logger.info(f"Retrieved {len(documents)} tier {tier} documents with full categorical data")
        return documents
    
    def fetch_document_keywords(self, document_id: str) -> List[str]:
        """
        Fetch keywords for a specific document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            List of keywords
        """
        cursor = self.conn.cursor()
        
        # The document_keywords table structure is the same in both database types,
        # but the foreign key references different tables
        cursor.execute(
            """
            SELECT keyword
            FROM document_keywords
            WHERE document_id = ?
            ORDER BY score DESC
            LIMIT 20
            """,
            (document_id,)
        )
        
        return [kw[0] for kw in cursor.fetchall()]
        
    def fetch_document_subject_matters(self, document_id: str) -> List[str]:
        """
        Fetch subject matters for a specific document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            List of subject matters
        """
        if self.db_type != 'consolidated':
            return []
            
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT sm.subject_name
            FROM document_subject_matters dsm
            JOIN subject_matters sm ON dsm.subject_id = sm.subject_id
            WHERE dsm.document_id = ?
            """,
            (document_id,)
        )
        
        return [row[0] for row in cursor.fetchall()]
        
    def fetch_document_eurovoc_descriptors(self, document_id: str) -> List[str]:
        """
        Fetch EuroVoc descriptors for a specific document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            List of EuroVoc descriptors
        """
        if self.db_type != 'consolidated':
            return []
            
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT ed.descriptor_name
            FROM document_eurovoc_descriptors ded
            JOIN eurovoc_descriptors ed ON ded.descriptor_id = ed.descriptor_id
            WHERE ded.document_id = ?
            """,
            (document_id,)
        )
        
        return [row[0] for row in cursor.fetchall()]
        
    def fetch_document_authors(self, document_id: str) -> List[str]:
        """
        Fetch authors for a specific document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            List of authors
        """
        if self.db_type != 'consolidated':
            return []
            
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT a.name
            FROM document_authors da
            JOIN authors a ON da.author_id = a.author_id
            WHERE da.document_id = ?
            """,
            (document_id,)
        )
        
        return [row[0] for row in cursor.fetchall()]
        
    def fetch_document_directory_codes(self, document_id: str) -> List[Dict[str, str]]:
        """
        Fetch directory codes for a specific document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            List of directory codes with their labels
        """
        if self.db_type != 'consolidated':
            return []
            
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT dc.directory_code
            FROM document_directory_codes ddc
            JOIN directory_codes dc ON ddc.directory_id = dc.directory_id
            WHERE ddc.document_id = ?
            """,
            (document_id,)
        )
        
        return [{'code': row[0], 'label': row[0]} for row in cursor.fetchall()]
    
    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Closed SQLite database connection")

class APIDocumentConnector(DocumentDBConnector):
    """Connector for API-based document database."""
    
    def __init__(self, api_url: str, api_key: Optional[str] = None):
        """
        Initialize the API connector.
        
        Args:
            api_url: URL of the API
            api_key: Optional API key for authentication
        """
        self.api_url = api_url
        self.api_key = api_key
        logger.info(f"Initialized API connector for: {api_url}")
        
        # This is a placeholder for actual API connection setup
        # In a real implementation, you might initialize an API client here
    
    def fetch_tier_documents(self, tier: int) -> List[Dict[str, Any]]:
        """
        Fetch documents of a specific tier from the API.
        
        Args:
            tier: Tier to fetch (1, 2, 3, or 4)
            
        Returns:
            List of document dictionaries
        """
        logger.info(f"Fetching tier {tier} documents from API")
        
        # This is a placeholder for actual API implementation
        # In a real implementation, you would make HTTP requests to the API
        # Example:
        # response = requests.get(
        #     f"{self.api_url}/documents",
        #     params={"tier": tier, "has_summary": True},
        #     headers={"Authorization": f"Bearer {self.api_key}"}
        # )
        # documents = response.json()
        
        # For now, return an empty list
        logger.warning("API connector is not fully implemented yet")
        return []
    
    def fetch_document_keywords(self, document_id: str) -> List[str]:
        """
        Fetch keywords for a specific document from the API.
        
        Args:
            document_id: ID of the document
            
        Returns:
            List of keywords
        """
        # This is a placeholder for actual API implementation
        # Example:
        # response = requests.get(
        #     f"{self.api_url}/documents/{document_id}/keywords",
        #     headers={"Authorization": f"Bearer {self.api_key}"}
        # )
        # keywords = response.json()
        
        # For now, return an empty list
        return []
    
    def close(self) -> None:
        """Close the API connection."""
        # This is a placeholder for actual API connection cleanup
        logger.info("Closed API connection")

def get_connector(connector_type: str, **kwargs) -> DocumentDBConnector:
    """
    Factory function to get the appropriate connector based on type.
    
    Args:
        connector_type: Type of connector ('sqlite' or 'api')
        **kwargs: Additional arguments for the connector
            - db_path: Path to the SQLite database (for sqlite connector)
            - db_type: Type of database structure ('consolidated' or 'legacy', default: 'legacy')
            - api_url: URL for the API (for api connector)
            - api_key: Optional API key (for api connector)
        
    Returns:
        DocumentDBConnector instance
    """
    if connector_type.lower() == 'sqlite':
        if 'db_path' not in kwargs:
            raise ValueError("db_path is required for SQLite connector")
        return SQLiteDocumentConnector(
            db_path=kwargs['db_path'],
            db_type=kwargs.get('db_type', 'legacy')
        )
    elif connector_type.lower() == 'api':
        if 'api_url' not in kwargs:
            raise ValueError("api_url is required for API connector")
        return APIDocumentConnector(
            api_url=kwargs['api_url'],
            api_key=kwargs.get('api_key')
        )
    else:
        raise ValueError(f"Unsupported connector type: {connector_type}")
