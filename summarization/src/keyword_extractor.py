"""
Keyword extraction for EU legal documents using KeyBERT.
"""
import sqlite3
from pathlib import Path
import logging
from typing import List, Tuple, Dict
from keybert import KeyBERT
import numpy as np
from tqdm import tqdm
import gc
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DynamicKeywordExtractor:
    def __init__(self, 
                 a: float = 2.5,
                 b: float = -4.77,
                 min_keywords: int = 2,
                 max_keywords: int = 15,  # Reduced max keywords
                 model_name: str = 'distilbert-base-nli-mean-tokens'):
        self.a = a
        self.b = b
        self.min_keywords = min_keywords
        self.max_keywords = max_keywords
        logger.info(f"Initializing KeyBERT with model: {model_name}")
        self.kw_model = KeyBERT(model=model_name)
    
    def calculate_num_keywords(self, doc_length: int) -> int:
        """Calculate number of keywords based on document length."""
        num = self.a * np.log(doc_length) + self.b
        return int(np.clip(num, self.min_keywords, self.max_keywords))
    
    def extract_keywords(self, 
                        text: str,
                        ngram_range: Tuple[int, int] = (1, 2),
                        stop_words: str = 'english',
                        use_maxsum: bool = True,
                        diversity: float = 0.7) -> List[Tuple[str, float]]:
        """Extract keywords with dynamic count based on text length."""
        # Count words (approximate)
        word_count = len(text.split())
        
        # Calculate number of keywords
        top_n = self.calculate_num_keywords(word_count)
        
        # Extract keywords
        keywords = self.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=ngram_range,
            stop_words=stop_words,
            top_n=top_n,
            use_maxsum=use_maxsum,
            diversity=diversity
        )
        
        return keywords

def process_documents(db_path: str, batch_size: int = 3, max_text_length: int = 100000, resume_from: int = 0) -> None:
    """Process documents from SQLite database and extract keywords."""
    # Create keyword extractor
    extractor = DynamicKeywordExtractor()
    
    # Connect to database with timeout and immediate transaction mode
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for better concurrency
    cursor = conn.cursor()
    
    try:
        # Create keywords table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_keywords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            keyword TEXT NOT NULL,
            score REAL NOT NULL,
            FOREIGN KEY (document_id) REFERENCES processed_documents(id)
        )
        """)
        
        # Get count of unprocessed documents
        cursor.execute("""
            SELECT COUNT(*)
            FROM processed_documents pd
            WHERE NOT EXISTS (
                SELECT 1 FROM document_keywords dk WHERE dk.document_id = pd.id
            )
        """)
        total_docs = cursor.fetchone()[0]
        logger.info(f"Found {total_docs} documents without keywords to process")
        
        # Process documents in batches
        if resume_from > 0:
            logger.info(f"Resuming from offset {resume_from}")
        
        processed_count = 0
        total_batches = (total_docs - resume_from + batch_size - 1) // batch_size
        
        logger.info(f"Will process {total_docs} documents in {total_batches} batches of {batch_size}")
        
        for batch_num in tqdm(range(total_batches), desc=f"Processing {total_docs} documents"):
            offset = resume_from + (batch_num * batch_size)
            # Get batch of documents with retries
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    cursor.execute("""
                SELECT pd.id, pd.celex_number, GROUP_CONCAT(ds.content, ' ') as full_text
                FROM processed_documents pd
                LEFT JOIN document_sections ds ON pd.id = ds.document_id
                WHERE NOT EXISTS (
                    SELECT 1 FROM document_keywords dk WHERE dk.document_id = pd.id
                )
                GROUP BY pd.id, pd.celex_number
                LIMIT ? OFFSET ?
            """, (batch_size, offset))
                    documents = cursor.fetchall()
                    break
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e):
                        retry_count += 1
                        if retry_count < max_retries:
                            logger.warning(f"Database locked, retrying... (attempt {retry_count}/{max_retries})")
                            import time
                            time.sleep(5)  # Wait 5 seconds before retrying
                        else:
                            logger.error(f"Failed to access database after {max_retries} attempts")
                            raise
                    else:
                        raise
            
            for doc_id, celex_number, full_text in documents:
                if not full_text:
                    logger.warning(f"No content found for document {celex_number}")
                    continue
                
                # Limit text length to prevent memory issues
                if len(full_text) > max_text_length:
                    logger.warning(f"Document {celex_number} text too long ({len(full_text)} chars), truncating to {max_text_length}")
                    full_text = full_text[:max_text_length]
                
                try:
                    # Extract keywords from full text
                    keywords = extractor.extract_keywords(full_text)
                    
                    # Store keywords with retries
                    retry_count = 0
                    while retry_count < max_retries:
                        try:
                            cursor.executemany(
                                "INSERT INTO document_keywords (document_id, keyword, score) VALUES (?, ?, ?)",
                                [(doc_id, kw, score) for kw, score in keywords]
                            )
                            break
                        except sqlite3.OperationalError as e:
                            if "database is locked" in str(e):
                                retry_count += 1
                                if retry_count < max_retries:
                                    logger.warning(f"Database locked during insert, retrying... (attempt {retry_count}/{max_retries})")
                                    time.sleep(5)
                                else:
                                    logger.error(f"Failed to insert keywords after {max_retries} attempts")
                                    raise
                            else:
                                raise
                    
                except Exception as e:
                    logger.error(f"Error processing document {celex_number}: {str(e)}")
                    continue
            
            # Commit batch and cleanup
            conn.commit()
            
            # Aggressive memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            gc.collect()  # Double collection to ensure cleanup
            
            # Clear variables
            if 'keywords' in locals():
                del keywords
            
            import time
            time.sleep(1)  # Short pause to let memory settle
            
            # Update and log progress
            docs_in_batch = len(documents)
            processed_count += docs_in_batch
            if docs_in_batch > 0 and (batch_num == 0 or (batch_num + 1) % 10 == 0 or batch_num == total_batches - 1):
                percent_done = (processed_count / total_docs) * 100
                logger.info(f"Processed {processed_count}/{total_docs} documents ({percent_done:.1f}%)")
            
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        conn.rollback()
    
    finally:
        conn.close()

if __name__ == "__main__":
    import argparse
    import signal
    import sys
    
    def signal_handler(sig, frame):
        print('\nGracefully shutting down...')
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-from", type=int, default=0, help="Resume from offset")
    parser.add_argument("--batch-size", type=int, default=3, help="Batch size for processing")
    parser.add_argument("--max-text-length", type=int, default=100000, help="Maximum text length to process")
    args = parser.parse_args()
    
    # Path to database
    DB_PATH = Path(__file__).parents[1] / "data" / "processed_documents.db"
    
    # Process documents
    logger.info(f"Starting keyword extraction from {DB_PATH}")
    try:
        process_documents(
            str(DB_PATH),
            batch_size=args.batch_size,
            max_text_length=args.max_text_length,
            resume_from=args.resume_from
        )
        logger.info("Keyword extraction complete")
    except KeyboardInterrupt:
        logger.info("\nGracefully shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        sys.exit(1)
