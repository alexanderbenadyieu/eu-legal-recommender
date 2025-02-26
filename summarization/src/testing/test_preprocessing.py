"""Test the preprocessing pipeline with documents from the database."""
import sys
import json
from pathlib import Path
import sqlite3
from dataclasses import dataclass, asdict
from typing import List, Optional
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

def load_documents(db_path: Path) -> List[Document]:
    """Load all documents from the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all documents with their metadata and content
    cursor.execute("""
        SELECT id, celex_number, html_url, content, content_html 
        FROM legal_documents
        WHERE content_html IS NOT NULL
    """)
    
    documents = []
    for row in cursor:
        doc = Document(
            id=row[0],
            celex=row[1],
            html_url=row[2],
            content=row[3],
            content_html=row[4]
        )
        documents.append(doc)
    
    conn.close()
    return documents

def process_documents(documents: List[Document], output_dir: Path):
    """Process documents and save their sections."""
    parser = LegalDocumentParser(output_dir)
    
    for doc in documents:
        print(f"Processing document {doc.celex}...")
        
        # Parse HTML content into sections
        sections = parser.parse_html_content(doc.content_html)
        doc.sections = sections
        
        # Save processed document
        doc_dir = output_dir / doc.celex
        doc_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata and sections
        metadata = {
            'id': doc.id,
            'celex': doc.celex,
            'html_url': doc.html_url,
            'sections': [
                {
                    'title': section.title,
                    'content': section.content,
                    'type': section.section_type
                }
                for section in sections
            ]
        }
        
        with open(doc_dir / 'processed.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Found {len(sections)} sections")
        print(f"Saved to {doc_dir / 'processed.json'}\n")

def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent
    db_path = base_dir / "scraper" / "data" / "eurlex.db"
    output_dir = base_dir / "summarization" / "processed_documents"
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get the first document that has HTML content
    cursor.execute("""
        SELECT document_id, celex_number, html_url, content, content_html 
        FROM documents
        WHERE content_html IS NOT NULL
        LIMIT 1
    """)
    
    row = cursor.fetchone()
    if not row:
        print(f"Document {celex} not found in database")
        sys.exit(1)
        
    # Create document object
    doc = Document(
        id=row[0],
        celex=row[1],
        html_url=row[2],
        content=row[3],
        content_html=row[4]  # html_content in DB
    )
    
    # Process the document
    process_documents([doc], output_dir)
    
    conn.close()

if __name__ == "__main__":
    main()
