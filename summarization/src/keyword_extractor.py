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
                 max_keywords: int = 20,
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

def process_documents(db_path: str, batch_size: int = 10) -> None:
    """Process documents from SQLite database and extract keywords."""
    # Create keyword extractor
    extractor = DynamicKeywordExtractor()
    
    # Connect to database
    conn = sqlite3.connect(db_path)
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
        
        # Get total number of documents
        cursor.execute("SELECT COUNT(*) FROM processed_documents")
        total_docs = cursor.fetchone()[0]
        logger.info(f"Found {total_docs} documents to process")
        
        # Process documents in batches
        for offset in tqdm(range(0, total_docs, batch_size), desc="Processing documents"):
            # Get batch of documents
            cursor.execute("""
                SELECT pd.id, pd.celex_number, GROUP_CONCAT(ds.content, ' ') as full_text
                FROM processed_documents pd
                LEFT JOIN document_sections ds ON pd.id = ds.document_id
                GROUP BY pd.id, pd.celex_number
                LIMIT ? OFFSET ?
            """, (batch_size, offset))
            documents = cursor.fetchall()
            
            for doc_id, celex_number, full_text in documents:
                if not full_text:
                    logger.warning(f"No content found for document {celex_number}")
                    continue
                
                try:
                    # Extract keywords from full text
                    keywords = extractor.extract_keywords(full_text)
                    
                    # Store keywords
                    cursor.executemany(
                        "INSERT INTO document_keywords (document_id, keyword, score) VALUES (?, ?, ?)",
                        [(doc_id, kw, score) for kw, score in keywords]
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing document {celex_number}: {str(e)}")
                    continue
            
            # Commit batch
            conn.commit()
            
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        conn.rollback()
    
    finally:
        conn.close()

if __name__ == "__main__":
    # Path to database
    DB_PATH = Path(__file__).parents[1] / "data" / "processed_documents.db"
    
    # Process documents
    logger.info(f"Starting keyword extraction from {DB_PATH}")
    process_documents(str(DB_PATH))
    logger.info("Keyword extraction complete")
