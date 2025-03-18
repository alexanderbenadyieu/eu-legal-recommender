"""Process legal documents and store them in a SQLite database."""
import sys
from pathlib import Path
import sqlite3
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
from src.preprocessing.html_parser import LegalDocumentParser, DocumentSection

# Add the project root to the Python path to import database_utils
project_root = str(Path(__file__).parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

from database_utils import get_db_connection, save_document_section

@dataclass
class Document:
    """Represents a legal document with all its metadata and content."""
    id: int
    celex: str
    html_url: str
    content: str
    content_html: str
    sections: Optional[List[DocumentSection]] = None

def init_processed_db(db_path: Path = None):
    """Initialize the processed documents database.
    
    Note: This function is kept for backward compatibility but is no longer needed
    as we're using the consolidated database.
    """
    print("Using consolidated database - no initialization needed")

def store_processed_document(conn: sqlite3.Connection, doc: Document, sections: List[DocumentSection]):
    """Store a processed document and its sections in the database."""
    cursor = conn.cursor()
    
    # Get the document_id from the documents table using celex_number
    cursor.execute("SELECT document_id FROM documents WHERE celex_number = ?", (doc.celex,))
    result = cursor.fetchone()
    
    if not result:
        print(f"Warning: Document with CELEX {doc.celex} not found in documents table")
        return
    
    doc_id = result[0]
    
    # Insert all sections using the utility function
    for i, section in enumerate(sections):
        # Calculate word count
        word_count = len(section.content.split()) if section.content else 0
        
        # Use the save_document_section utility function
        save_document_section(
            document_id=doc_id,
            title=section.title,
            content=section.content,
            section_type=section.section_type,
            section_order=i,
            word_count=word_count
        )
    
    print(f"Stored document {doc.celex} with ID {doc_id} and {len(sections)} sections")

def get_total_documents(conn: sqlite3.Connection) -> int:
    """Get total number of documents to process."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) 
        FROM documents 
        WHERE content_html IS NOT NULL
    """)
    return cursor.fetchone()[0]

def get_processed_documents(conn: sqlite3.Connection) -> set:
    """Get set of already processed document IDs."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT celex_number FROM documents d
        WHERE EXISTS (
            SELECT 1 FROM document_sections ds
            WHERE ds.document_id = d.document_id
        )
    """)
    return {row[0] for row in cursor}

def process_documents(source_db: Path = None, target_db: Path = None, batch_size: int = 10):
    """Process documents and store them in the target database.
    
    Note: source_db and target_db parameters are kept for backward compatibility,
    but we're now using the consolidated database for both source and target.
    """
    # Initialize parser
    parser = LegalDocumentParser(Path.cwd())
    
    # Connect to the consolidated database
    conn = get_db_connection(db_type='consolidated')
    
    # Get total documents and already processed ones
    total_docs = get_total_documents(conn)
    processed_docs = get_processed_documents(conn)
    
    print(f"Found {total_docs} documents to process")
    print(f"Already processed: {len(processed_docs)} documents")
    
    cursor = conn.cursor()
    processed_count = 0
    error_count = 0
    
    try:
        # Get documents in batches
        cursor.execute("""
            SELECT document_id, celex_number, html_url, content, content_html 
            FROM documents
            WHERE content_html IS NOT NULL
        """)
        
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
                
            for row in rows:
                try:
                    doc = Document(
                        id=row[0],
                        celex=row[1],
                        html_url=row[2],
                        content=row[3],
                        content_html=row[4]
                    )
                    
                    # Skip if already processed
                    if doc.celex in processed_docs:
                        continue
                    
                    print(f"Processing document {doc.celex}... ({processed_count + 1}/{total_docs})")
                    
                    # Parse HTML content into sections
                    sections = parser.parse_html_content(doc.content_html)
                    
                    # Store in database
                    store_processed_document(conn, doc, sections)
                    
                    processed_count += 1
                    print(f"Found {len(sections)} sections")
                    print(f"Saved to database\n")
                    
                except Exception as e:
                    error_count += 1
                    print(f"Error processing document {row[1]}: {str(e)}\n")
                    continue
    
    finally:
        print(f"\nProcessing complete:")
        print(f"- Successfully processed: {processed_count} documents")
        print(f"- Errors encountered: {error_count} documents")
        print(f"- Total documents in database: {len(processed_docs) + processed_count}")
        
        conn.close()

def main():
    # No need to set up paths as we're using the consolidated database
    print("Using consolidated database for document processing")
    
    # Process documents
    print("\nStarting document processing...")
    process_documents(batch_size=10)

if __name__ == "__main__":
    main()
