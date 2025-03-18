import sqlite3
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

from database_utils import get_db_connection

def get_word_count(text: str) -> int:
    """Get word count of text."""
    return len(text.split())

def add_word_count_column(db_type='consolidated'):
    """Add word_count column to documents table if it doesn't exist and calculate section word counts"""
    # Connect to the appropriate database
    if db_type == 'consolidated':
        conn = get_db_connection(db_type='consolidated')
    else:
        conn = sqlite3.connect('summarization/data/processed_documents.db')
        
    cursor = conn.cursor()
    
    # Check if word_count column exists in the appropriate table
    if db_type == 'consolidated':
        cursor.execute("PRAGMA table_info(documents)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'word_count' not in columns:
            cursor.execute("ALTER TABLE documents ADD COLUMN word_count INTEGER")
        
        # Check if word_count column exists in document_sections
        cursor.execute("PRAGMA table_info(document_sections)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'word_count' not in columns:
            cursor.execute("ALTER TABLE document_sections ADD COLUMN word_count INTEGER")
    else:
        # Legacy database
        cursor.execute("PRAGMA table_info(processed_documents)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'total_words' not in columns:
            cursor.execute("ALTER TABLE processed_documents ADD COLUMN total_words INTEGER")
        
        # Check if word_count column exists in document_sections
        cursor.execute("PRAGMA table_info(document_sections)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'word_count' not in columns:
            cursor.execute("ALTER TABLE document_sections ADD COLUMN word_count INTEGER")
    
    # Update word counts for all sections
    if db_type == 'consolidated':
        cursor.execute("""
            SELECT section_id, content 
            FROM document_sections 
            WHERE word_count IS NULL
        """)
    else:
        cursor.execute("""
            SELECT id, content 
            FROM document_sections 
            WHERE word_count IS NULL
        """)
        
    sections = cursor.fetchall()
    
    print(f"Calculating word counts for {len(sections)} sections...")
    
    for section_id, content in sections:
        word_count = get_word_count(content)
        if db_type == 'consolidated':
            cursor.execute("""
                UPDATE document_sections 
                SET word_count = ? 
                WHERE section_id = ?
            """, (word_count, section_id))
        else:
            cursor.execute("""
                UPDATE document_sections 
                SET word_count = ? 
                WHERE id = ?
            """, (word_count, section_id))
        print(f"Section {section_id}: {word_count} words")
    
    # Update total word counts for documents
    if db_type == 'consolidated':
        cursor.execute("""
            UPDATE documents
            SET word_count = (
                SELECT SUM(word_count)
                FROM document_sections
                WHERE document_sections.document_id = documents.document_id
            )
            WHERE word_count IS NULL
        """)
    else:
        cursor.execute("""
            UPDATE processed_documents
            SET total_words = (
                SELECT SUM(word_count)
                FROM document_sections
                WHERE document_sections.document_id = processed_documents.id
            )
            WHERE total_words IS NULL
        """)
    
    conn.commit()
    
    # Print summary statistics
    if db_type == 'consolidated':
        cursor.execute("""
            SELECT 
                COUNT(*) as total_docs,
                AVG(word_count) as avg_words,
                MIN(word_count) as min_words,
                MAX(word_count) as max_words
            FROM documents
            WHERE word_count IS NOT NULL
        """)
    else:
        cursor.execute("""
            SELECT 
                COUNT(*) as total_docs,
                AVG(total_words) as avg_words,
                MIN(total_words) as min_words,
                MAX(total_words) as max_words
            FROM processed_documents
            WHERE total_words IS NOT NULL
        """)
    stats = cursor.fetchone()
    
    print("\nSummary Statistics:")
    print(f"Total Documents: {stats[0]}")
    print(f"Average Words: {stats[1]:.1f}")
    print(f"Min Words: {stats[2]}")
    print(f"Max Words: {stats[3]}")
    
    # Print distribution across tiers
    if db_type == 'consolidated':
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
    else:
        cursor.execute("""
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
    for tier, count in cursor.fetchall():
        print(f"{tier}: {count} documents")
    
    conn.close()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Add word counts to documents and sections")
    parser.add_argument("--db-type", type=str, choices=['consolidated', 'legacy'], default='consolidated',
                      help="Database type to use ('consolidated' or 'legacy')")
    
    args = parser.parse_args()
    add_word_count_column(db_type=args.db_type)
