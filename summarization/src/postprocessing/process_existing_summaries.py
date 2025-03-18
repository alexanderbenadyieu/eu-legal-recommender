import argparse
import logging
import sqlite3
import sys
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

# Add the project root to the Python path
project_root = str(Path(__file__).parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

from database_utils import get_db_connection
from .deepseek_processor import DeepSeekPostProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_documents_by_tier(cursor: sqlite3.Cursor, eurlex_cursor: sqlite3.Cursor, tier: int, db_type: str = 'consolidated') -> List[Tuple[int, str, str, int]]:
    """Get all documents for a specific tier."""
    # Tier word count ranges
    tier_ranges = {
        1: (0, 600),
        2: (601, 2500),
        3: (2501, 20000),
        4: (20001, float('inf'))
    }
    
    min_words, max_words = tier_ranges[tier]
    
    # Get documents with their CELEX numbers based on database type
    if db_type == 'consolidated':
        cursor.execute("""
            SELECT document_id, celex_number, summary, word_count 
            FROM documents
            WHERE word_count >= ? AND word_count <= ?
            AND summary IS NOT NULL
        """, (min_words, max_words))
    else:
        # Legacy database query
        cursor.execute("""
            SELECT pd.id, pd.celex_number, pd.summary, pd.total_words 
            FROM processed_documents pd
            WHERE pd.total_words >= ? AND pd.total_words <= ?
            AND pd.summary IS NOT NULL
        """, (min_words, max_words))
    
    documents = cursor.fetchall()
    
    # Get titles from eurlex database
    processed_docs = []
    for doc_id, celex_number, summary, word_count in documents:
        # Get title from eurlex database
        eurlex_cursor.execute("""
            SELECT title 
            FROM documents 
            WHERE celex_number = ?
        """, (celex_number,))
        
        result = eurlex_cursor.fetchone()
        title = result[0] if result and result[0] else f'Document {celex_number}'
        
        processed_docs.append((doc_id, title, summary, word_count))
    
    return processed_docs

def update_document_summary(cursor: sqlite3.Cursor, doc_id: int, new_summary: str, db_type: str = 'consolidated'):
    """Update the summary for a document."""
    # Update summary and recalculate word count and compression ratio
    from nltk.tokenize import word_tokenize
    new_word_count = len(word_tokenize(new_summary))
    
    if db_type == 'consolidated':
        # Get the current word_count to calculate compression ratio
        cursor.execute("SELECT word_count FROM documents WHERE document_id = ?", (doc_id,))
        result = cursor.fetchone()
        if not result:
            logger.warning(f"Document with ID {doc_id} not found in database")
            return
            
        total_words = result[0]
        compression_ratio = new_word_count / total_words if total_words > 0 else 0
        
        cursor.execute("""
            UPDATE documents 
            SET summary = ?,
                summary_word_count = ?,
                compression_ratio = ?
            WHERE document_id = ?
        """, (new_summary, new_word_count, compression_ratio, doc_id))
    else:
        # Legacy database update
        cursor.execute("""
            UPDATE processed_documents 
            SET summary = ?,
                summary_word_count = ?,
                compression_ratio = CAST(? AS FLOAT) / CAST(total_words AS FLOAT)
            WHERE id = ?
        """, (new_summary, new_word_count, new_word_count, doc_id))

def process_tier(
    processor: DeepSeekPostProcessor,
    cursor: sqlite3.Cursor,
    eurlex_cursor: sqlite3.Cursor,
    conn: sqlite3.Connection,
    tier: int,
    batch_size: int = 10,
    start_from: int = 0,
    db_type: str = 'consolidated'
):
    """Process all documents in a specific tier."""
    documents = get_documents_by_tier(cursor, eurlex_cursor, tier, db_type=db_type)
    total_docs = len(documents)
    logger.info(f"Found {total_docs} documents in tier {tier}")
    
    if start_from > 0:
        logger.info(f"Resuming from document {start_from}")
        documents = documents[start_from:]
    
    for i, (doc_id, title, summary, word_count) in enumerate(tqdm(documents, initial=start_from, total=total_docs)):
        try:
            # Skip if summary is empty
            if not summary or not summary.strip():
                logger.warning(f"Empty summary for document {doc_id}, skipping")
                continue
                
            # Process summary
            refined_summary = processor.refine_summary(title, summary, str(doc_id))
            
            if refined_summary:
                update_document_summary(cursor, doc_id, refined_summary, db_type=db_type)
                
                # Commit every batch_size documents
                if (i + 1) % batch_size == 0:
                    conn.commit()
                    logger.info(f"Processed and committed batch of {batch_size} documents (total: {i + 1 + start_from}/{total_docs})")
            else:
                logger.warning(f"Failed to refine summary for document {doc_id}")
                
        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {str(e)}")
            # Save progress before raising
            conn.commit()
            raise
    
    # Final commit for any remaining documents
    conn.commit()

def main():
    parser = argparse.ArgumentParser(description="Process existing summaries using DeepSeek")
    parser.add_argument("--db-path", type=str, help="Path to SQLite database (legacy option)")
    parser.add_argument("--eurlex-db-path", type=str, required=True, help="Path to Eurlex SQLite database")
    parser.add_argument("--api-key", type=str, required=True, help="DeepSeek API key")
    parser.add_argument("--tier", type=int, choices=[1,2,3,4], required=True, help="Document tier to process")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for commits")
    parser.add_argument("--resume-from", type=int, default=0, help="Resume from document number (0-based index)")
    parser.add_argument("--db-type", type=str, choices=['consolidated', 'legacy'], default='consolidated', 
                      help="Database type to use ('consolidated' or 'legacy')")
    
    args = parser.parse_args()
    
    # Connect to databases
    if args.db_type == 'consolidated':
        conn = get_db_connection(db_type='consolidated')
    else:
        if not args.db_path:
            parser.error("--db-path is required when using legacy database")
        conn = sqlite3.connect(args.db_path)
    
    cursor = conn.cursor()
    
    eurlex_conn = sqlite3.connect(args.eurlex_db_path)
    eurlex_cursor = eurlex_conn.cursor()
    
    processor = DeepSeekPostProcessor(args.api_key)
    
    try:
        process_tier(
            processor=processor,
            cursor=cursor,
            eurlex_cursor=eurlex_cursor,
            conn=conn,
            tier=args.tier,
            batch_size=args.batch_size,
            start_from=args.resume_from,
            db_type=args.db_type
        )
        logger.info(f"Successfully processed all documents in tier {args.tier}")
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
        conn.commit()  # Save progress
        logger.info("Progress saved")
    except Exception as e:
        logger.error(f"Error processing tier {args.tier}: {str(e)}")
    finally:
        conn.close()
        eurlex_conn.close()

if __name__ == "__main__":
    main()
