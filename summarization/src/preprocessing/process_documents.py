"""Process legal documents and store them in a SQLite database."""
import sys
from pathlib import Path
import sqlite3
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
from src.preprocessing.html_parser import LegalDocumentParser, DocumentSection

@dataclass
class Document:
    """Represents a legal document with all its metadata and content."""
    id: int
    celex: str
    html_url: str
    content: str
    content_html: str
    sections: Optional[List[DocumentSection]] = None

def init_processed_db(db_path: Path):
    """Initialize the processed documents database."""
    conn = sqlite3.connect(db_path)
    with open(db_path.parent / 'data_models.sql', 'r') as f:
        conn.executescript(f.read())
    conn.close()

def store_processed_document(conn: sqlite3.Connection, doc: Document, sections: List[DocumentSection]):
    """Store a processed document and its sections in the database."""
    cursor = conn.cursor()
    
    # Insert the processed document
    cursor.execute("""
        INSERT INTO processed_documents (celex_number, html_url, processed_date)
        VALUES (?, ?, ?)
    """, (doc.celex, doc.html_url, datetime.now()))
    
    doc_id = cursor.lastrowid
    
    # Insert all sections
    for i, section in enumerate(sections):
        cursor.execute("""
            INSERT INTO document_sections 
            (document_id, title, content, section_type, section_order)
            VALUES (?, ?, ?, ?, ?)
        """, (doc_id, section.title, section.content, section.section_type, i))
    
    conn.commit()

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
    cursor.execute("SELECT celex_number FROM processed_documents")
    return {row[0] for row in cursor}

def process_documents(source_db: Path, target_db: Path, batch_size: int = 10):
    """Process documents and store them in the target database."""
    # Initialize parser
    parser = LegalDocumentParser(Path.cwd())
    
    # Connect to both databases
    source_conn = sqlite3.connect(source_db)
    target_conn = sqlite3.connect(target_db)
    
    # Get total documents and already processed ones
    total_docs = get_total_documents(source_conn)
    processed_docs = get_processed_documents(target_conn)
    
    print(f"Found {total_docs} documents to process")
    print(f"Already processed: {len(processed_docs)} documents")
    
    source_cursor = source_conn.cursor()
    processed_count = 0
    error_count = 0
    
    try:
        # Get documents in batches
        source_cursor.execute("""
            SELECT document_id, celex_number, html_url, content, content_html 
            FROM documents
            WHERE content_html IS NOT NULL
        """)
        
        while True:
            rows = source_cursor.fetchmany(batch_size)
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
                    store_processed_document(target_conn, doc, sections)
                    
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
        
        source_conn.close()
        target_conn.close()

def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent
    source_db = base_dir / "scraper" / "data" / "eurlex.db"
    target_db = base_dir / "summarization" / "data" / "processed_documents.db"
    
    # Create data directory if it doesn't exist
    target_db.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize the processed documents database if it doesn't exist
    if not target_db.exists():
        print("Initializing processed documents database...")
        init_processed_db(target_db)
    else:
        print("Using existing processed documents database")
    
    # Process documents
    print("\nStarting document processing...")
    process_documents(source_db, target_db, batch_size=10)

if __name__ == "__main__":
    main()
