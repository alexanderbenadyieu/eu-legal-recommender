"""Process a single document to test the HTML parser."""
import sqlite3
from pathlib import Path
from datetime import datetime
from src.preprocessing.html_parser import LegalDocumentParser, DocumentSection

def process_document(source_db: Path, target_db: Path, celex: str):
    """Process a single document."""
    # Initialize parser
    parser = LegalDocumentParser(Path.cwd())
    
    # Connect to both databases
    source_conn = sqlite3.connect(source_db)
    target_conn = sqlite3.connect(target_db)
    
    source_cursor = source_conn.cursor()
    
    try:
        # Get document from source
        source_cursor.execute("""
            SELECT document_id, celex_number, html_url, content, content_html 
            FROM documents
            WHERE celex_number = ?
        """, (celex,))
        
        row = source_cursor.fetchone()
        if not row:
            print(f"Document {celex} not found in source database")
            return
        
        print(f"\nProcessing document {celex}...")
        
        # Parse HTML content into sections
        sections = parser.parse_html_content(row[4])
        
        print(f"Found {len(sections)} sections")
        print("\nSection types found:")
        section_types = {}
        for section in sections:
            section_types[section.section_type] = section_types.get(section.section_type, 0) + 1
        for stype, count in section_types.items():
            print(f"  {stype}: {count} sections")
        
        # Store in target database
        cursor = target_conn.cursor()
        
        # Insert the processed document
        cursor.execute("""
            INSERT INTO processed_documents (celex_number, html_url, processed_date)
            VALUES (?, ?, ?)
        """, (row[1], row[2], datetime.now()))
        
        doc_id = cursor.lastrowid
        
        # Insert all sections
        for i, section in enumerate(sections):
            cursor.execute("""
                INSERT INTO document_sections 
                (document_id, title, content, section_type, section_order)
                VALUES (?, ?, ?, ?, ?)
            """, (doc_id, section.title, section.content, section.section_type, i))
        
        target_conn.commit()
        print("\nSaved to database")
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")
    finally:
        source_conn.close()
        target_conn.close()

def main():
    base_dir = Path(__file__).parent.parent
    source_db = base_dir / "scraper" / "data" / "eurlex.db"
    target_db = base_dir / "summarization" / "data" / "processed_documents.db"
    
    process_document(source_db, target_db, "32023R2772")

if __name__ == "__main__":
    main()
