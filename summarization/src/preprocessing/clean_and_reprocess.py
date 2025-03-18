"""
Clean and reprocess all documents in the consolidated database
"""
import sqlite3
from pathlib import Path
import sys
from tqdm import tqdm

# Add parent directory to path to import parser
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parents[3]))  # Add project root to path

from summarization.src.preprocessing.html_parser import LegalDocumentParser
from database_utils import get_db_connection, save_document_section

def get_word_count(text: str) -> int:
    """Get word count of text."""
    return len(text.split())

def clean_database():
    """Clean document sections in the consolidated database"""
    print("Cleaning document sections in the consolidated database...")
    
    conn = get_db_connection(db_type='consolidated')
    cursor = conn.cursor()
    
    try:
        # Delete all document sections
        cursor.execute("DELETE FROM document_sections")
        
        # Reset the auto-increment counter for document_sections
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='document_sections'")
        
        conn.commit()
        print("Document sections cleaned successfully!")
        
    finally:
        conn.close()

def process_documents():
    """Process all documents from the consolidated database"""
    
    # Initialize parser
    parser = LegalDocumentParser(Path('.'))
    
    # Connect to the consolidated database
    conn = get_db_connection(db_type='consolidated')
    cursor = conn.cursor()
    
    try:
        # Get all documents from the consolidated database
        cursor.execute("""
            SELECT document_id, celex_number, content_html, html_url 
            FROM documents
            WHERE content_html IS NOT NULL
        """)
        documents = cursor.fetchall()
        
        print(f"\nProcessing {len(documents)} documents...")
        
        # Process each document
        for document_id, celex_number, html_content, html_url in tqdm(documents):
            try:
                # Parse HTML content
                sections = parser.parse_html_content(html_content)
                
                if not sections:
                    print(f"\nWarning: No sections found for {celex_number}")
                    continue
                
                # Document already exists in the consolidated database
                # We'll use the document_id directly
                
                # Insert sections using the utility function
                for order, section in enumerate(sections):
                    word_count = get_word_count(section.content)
                    
                    save_document_section(
                        document_id=document_id,
                        title=section.title,
                        content=section.content,
                        section_type=section.section_type,
                        section_order=order,
                        word_count=word_count
                    )
                
                # Update total word count in the documents table
                cursor.execute("""
                    UPDATE documents 
                    SET word_count = (
                        SELECT SUM(word_count)
                        FROM document_sections
                        WHERE document_id = ?
                    )
                    WHERE document_id = ?
                """, (document_id, document_id))
                
                conn.commit()
                
            except Exception as e:
                print(f"\nError processing {celex_number}: {str(e)}")
                continue
        
        # Print statistics after processing
        cursor.execute("""
            SELECT 
                COUNT(*) as total_docs,
                AVG(word_count) as avg_words,
                MIN(word_count) as min_words,
                MAX(word_count) as max_words
            FROM documents
            WHERE word_count IS NOT NULL
        """)
        stats = cursor.fetchone()
        
        print("\nProcessing complete!")
        print(f"Total Documents: {stats[0]}")
        print(f"Average Words: {stats[1]:.1f}")
        print(f"Min Words: {stats[2]}")
        print(f"Max Words: {stats[3]}")
        
        # Print distribution across tiers
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN word_count <= 600 THEN 'Tier 1 (0-600)'
                    WHEN word_count <= 2500 THEN 'Tier 2 (601-2500)'
                    WHEN word_count <= 20000 THEN 'Tier 3 (2501-20000)'
                    ELSE 'Tier 4 (20000+)'
                END as tier,
                COUNT(*) as count
            FROM documents
            WHERE word_count IS NOT NULL
            GROUP BY tier
            ORDER BY MIN(word_count)
        """)
        
        print("\nDistribution across tiers:")
        for tier, count in cursor.fetchall():
            print(f"{tier}: {count} documents")
            
    finally:
        conn.close()

if __name__ == "__main__":
    clean_database()
    process_documents()
