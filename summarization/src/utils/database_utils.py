import sqlite3
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Document:
    document_id: int
    celex_number: str
    title: str
    identifier: Optional[str]
    eli_uri: Optional[str]
    html_url: Optional[str]
    pdf_url: Optional[str]
    date_of_document: Optional[datetime]
    date_of_effect: Optional[datetime]
    date_of_end_validity: Optional[datetime]
    content: str
    content_html: str

class DatabaseConnection:
    def __init__(self, db_path: str):
        self.db_path = db_path
        
    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self.conn
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()

def load_documents(db_path: str, limit: Optional[int] = None) -> List[Document]:
    """
    Load documents from the SQLite database.
    
    Args:
        db_path: Path to the SQLite database
        limit: Optional limit on number of documents to retrieve
        
    Returns:
        List of Document objects
    """
    with DatabaseConnection(db_path) as conn:
        query = """
            SELECT 
                document_id,
                celex_number,
                title,
                identifier,
                eli_uri,
                html_url,
                pdf_url,
                date_of_document,
                date_of_effect,
                date_of_end_validity,
                content,
                content_html
            FROM documents
            WHERE content_html IS NOT NULL
        """
        
        if limit:
            query += f" LIMIT {limit}"
            
        cursor = conn.execute(query)
        rows = cursor.fetchall()
        
        documents = []
        for row in rows:
            doc = Document(
                document_id=row['document_id'],
                celex_number=row['celex_number'],
                title=row['title'],
                identifier=row['identifier'],
                eli_uri=row['eli_uri'],
                html_url=row['html_url'],
                pdf_url=row['pdf_url'],
                date_of_document=datetime.strptime(row['date_of_document'], '%Y-%m-%d') if row['date_of_document'] else None,
                date_of_effect=datetime.strptime(row['date_of_effect'], '%Y-%m-%d') if row['date_of_effect'] else None,
                date_of_end_validity=datetime.strptime(row['date_of_end_validity'], '%Y-%m-%d') if row['date_of_end_validity'] else None,
                content=row['content'],
                content_html=row['content_html']
            )
            documents.append(doc)
            
        return documents
