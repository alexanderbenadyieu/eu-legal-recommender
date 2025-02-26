"""
Clean the processed_documents database and reprocess all documents
"""
import sqlite3
from pathlib import Path
import sys
from tqdm import tqdm

# Add parent directory to path to import parser
sys.path.append(str(Path(__file__).parent.parent))
from summarization.src.preprocessing.html_parser import LegalDocumentParser

def get_word_count(text: str) -> int:
    """Get word count of text."""
    return len(text.split())

def clean_database():
    """Clean the processed_documents database"""
    print("Cleaning processed_documents database...")
    
    conn = sqlite3.connect('summarization/data/processed_documents.db')
    cursor = conn.cursor()
    
    try:
        # First get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        # Drop all non-system tables
        for table in tables:
            if table[0] != 'sqlite_sequence':
                cursor.execute(f"DROP TABLE IF EXISTS {table[0]};")
        
        # Recreate tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                celex_number TEXT UNIQUE NOT NULL,
                html_url TEXT,
                total_words INTEGER
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_sections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                title TEXT,
                content TEXT NOT NULL,
                section_type TEXT NOT NULL,
                section_order INTEGER NOT NULL,
                word_count INTEGER NOT NULL,
                FOREIGN KEY (document_id) REFERENCES processed_documents(id)
            )
        """)
        
        conn.commit()
        print("Database cleaned successfully!")
        
    finally:
        conn.close()

def process_documents():
    """Process all documents from eurlex.db"""
    
    # Initialize parser
    parser = LegalDocumentParser(Path('.'))
    
    # Connect to both databases
    eurlex_conn = sqlite3.connect('scraper/data/eurlex.db')
    processed_conn = sqlite3.connect('summarization/data/processed_documents.db')
    
    eurlex_cursor = eurlex_conn.cursor()
    processed_cursor = processed_conn.cursor()
    
    try:
        # Get all documents from eurlex.db
        eurlex_cursor.execute("""
            SELECT celex_number, content_html, html_url 
            FROM documents
        """)
        documents = eurlex_cursor.fetchall()
        
        print(f"\nProcessing {len(documents)} documents...")
        
        # Process each document
        for celex_number, html_content, html_url in tqdm(documents):
            try:
                # Parse HTML content
                sections = parser.parse_html_content(html_content)
                
                if not sections:
                    print(f"\nWarning: No sections found for {celex_number}")
                    continue
                
                # Insert into processed_documents
                processed_cursor.execute("""
                    INSERT INTO processed_documents (celex_number, html_url)
                    VALUES (?, ?)
                """, (celex_number, html_url))
                
                document_id = processed_cursor.lastrowid
                
                # Insert sections
                for order, section in enumerate(sections):
                    processed_cursor.execute("""
                        INSERT INTO document_sections 
                        (document_id, title, content, section_type, section_order, word_count)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        document_id,
                        section.title,
                        section.content,
                        section.section_type,
                        order,
                        get_word_count(section.content)
                    ))
                
                # Update total word count
                processed_cursor.execute("""
                    UPDATE processed_documents 
                    SET total_words = (
                        SELECT SUM(word_count)
                        FROM document_sections
                        WHERE document_id = ?
                    )
                    WHERE id = ?
                """, (document_id, document_id))
                
                processed_conn.commit()
                
            except Exception as e:
                print(f"\nError processing {celex_number}: {str(e)}")
                continue
        
        # Print statistics after processing
        processed_cursor.execute("""
            SELECT 
                COUNT(*) as total_docs,
                AVG(total_words) as avg_words,
                MIN(total_words) as min_words,
                MAX(total_words) as max_words
            FROM processed_documents
            WHERE total_words IS NOT NULL
        """)
        stats = processed_cursor.fetchone()
        
        print("\nProcessing complete!")
        print(f"Total Documents: {stats[0]}")
        print(f"Average Words: {stats[1]:.1f}")
        print(f"Min Words: {stats[2]}")
        print(f"Max Words: {stats[3]}")
        
        # Print distribution across tiers
        processed_cursor.execute("""
            SELECT 
                CASE 
                    WHEN total_words <= 600 THEN 'Tier 1 (0-600)'
                    WHEN total_words <= 2500 THEN 'Tier 2 (601-2500)'
                    WHEN total_words <= 20000 THEN 'Tier 3 (2501-20000)'
                    ELSE 'Tier 4 (20000+)'
                END as tier,
                COUNT(*) as count
            FROM processed_documents
            WHERE total_words IS NOT NULL
            GROUP BY tier
            ORDER BY MIN(total_words)
        """)
        
        print("\nDistribution across tiers:")
        for tier, count in processed_cursor.fetchall():
            print(f"{tier}: {count} documents")
            
    finally:
        eurlex_conn.close()
        processed_conn.close()

if __name__ == "__main__":
    clean_database()
    process_documents()
