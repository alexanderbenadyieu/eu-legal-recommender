import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .utils.database_utils import load_documents, Document
from .preprocessing.html_parser import LegalDocumentParser
from .utils.text_chunking import chunk_text
from .extractive.lexlm_wrapper import LexLMExtractor
from .abstractive.bart_finetuner import BartFineTuner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SummarizationPipeline:
    def __init__(self, db_path: str, config: Dict[str, Any]):
        """
        Initialize the summarization pipeline.
        
        Args:
            db_path: Path to the SQLite database
            config: Configuration dictionary containing chunking parameters
        """
        self.db_path = db_path
        self.full_config = config  # Store full config
        self.config = config['chunking']  # Chunking config for backward compatibility
        self.html_parser = LegalDocumentParser(Path(db_path).parent)
        self.extractor = LexLMExtractor(config=config)
        self.generator = BartFineTuner()
        
    def _get_document_tier(self, word_count: int) -> int:
        """Determine document tier based on word count."""
        if word_count <= 600:
            return 1
        elif word_count <= 2500:
            return 2
        elif word_count <= 20000:
            return 3
        else:
            return 4

    def _process_tier_1(self, text: str) -> str:
        """Process Tier 1 document (0-600 words) - Direct abstractive summarization."""
        return self.generator.summarize(text)

    def _process_tier_2(self, document: Document, cursor: sqlite3.Cursor) -> str:
        """Process Tier 2 document (600-2,500 words) - Two-step summarization.
        
        Process:
        1. Split text into chunks that fit within LexLM context (514 tokens)
        2. Extract K words where K = max(300, min(0.3 × D, 600))
        3. Generate final summary of 0.6K to 0.8K words
        """
        logger.info(f"Processing Tier 2 document {document.celex_number} by sections")
        
        # Get document content from database
        cursor.execute("SELECT content FROM document_sections WHERE document_id = ?", (document.id,))
        sections = cursor.fetchall()
        if not sections:
            logger.warning(f"No sections found for document {document.celex_number}")
            return ""
            
        # Combine all sections into one text
        text = "\n\n".join(section[0] for section in sections)
        
        # Step 1: Split into chunks that fit within context length (514 tokens)
        chunks = chunk_text(text)  # Default max_tokens=514
        logger.info(f"Split document into {len(chunks)} chunks")
        
        # Step 2: Extractive summarization
        word_count = document.total_words  # Use stored word count
        target_extraction = max(300, min(int(0.3 * word_count), 600))  # K = max(300, min(0.3D, 600))
        
        logger.info(f"Extracting approximately {target_extraction} words from {word_count} words")
        
        # Process each chunk
        extracted_chunks = []
        for i, chunk in enumerate(chunks, 1):
            chunk_extracted = self.extractor.extract_key_sentences(chunk, chunk_number=i)
            extracted_chunks.append(chunk_extracted)
        
        # Combine extracted chunks
        extracted = ' '.join(extracted_chunks)
        
        # Count words in extracted text for summary length calculation
        from nltk.tokenize import word_tokenize
        extracted_words = len(word_tokenize(extracted))
        logger.info(f"Extracted {extracted_words} words from {len(chunks)} chunks")
        
        # Step 3: Abstractive summarization
        min_summary_length = int(0.6 * extracted_words)  # 0.6K
        max_summary_length = int(0.8 * extracted_words)  # 0.8K
        logger.info(f"Generating abstractive summary (target length: {min_summary_length}-{max_summary_length} words)")
        
        summary = self.generator.summarize(
            extracted,
            min_length=min_summary_length,
            max_length=max_summary_length
        )
        
        # Update document with summary statistics
        summary_words = len(word_tokenize(summary))
        document.summary = summary
        document.summary_word_count = summary_words
        document.compression_ratio = summary_words / word_count
        
        return summary

    def _process_tier_3(self, text: str) -> str:
        """Process Tier 3 document (2,500-20,000 words) - Hierarchical summarization."""
        # Step 1: Split into chunks
        chunks = chunk_text(text, max_chunk_size=self.config['max_chunk_size'])
        
        # Step 2: Extract from each chunk
        extracted_chunks = []
        for chunk in chunks:
            extracted = self.extractor.extract_key_sentences(chunk, tier=3)
            extracted_chunks.append(extracted)
        
        # Step 3: Combine and extract again
        combined = '\n'.join(extracted_chunks)
        final_extraction = self.extractor.extract_key_sentences(combined, tier=3)
        
        # Step 4: Final abstractive summary
        return self.generator.summarize(final_extraction)

    def _process_tier_4(self, text: str) -> str:
        """Process Tier 4 document (20,000+ words) - Advanced hierarchical."""
        # Similar to Tier 3 but with different target lengths
        chunks = chunk_text(text, max_chunk_size=self.config['max_chunk_size'])
        
        # First level of extraction
        extracted_chunks = []
        for chunk in chunks:
            extracted = self.extractor.extract_key_sentences(chunk, tier=4)
            extracted_chunks.append(extracted)
        
        # Second level of extraction
        combined = '\n'.join(extracted_chunks)
        final_extraction = self.extractor.extract_key_sentences(combined, tier=4)
        
        # Final abstractive summary
        return self.generator.summarize(final_extraction)

    def process_documents(self, limit: Optional[int] = None, tier: Optional[int] = None) -> List[Dict[str, Any]]:
        """Process documents through the tiered pipeline and store results in database.
        
        Args:
            limit: Optional limit on number of documents to process
            tier: Optional tier to process (1: ≤600 words, 2: 601-2500 words, 3: 2501-20000 words, 4: >20000 words)
        """
        logger.info("Loading documents from database...")
        
        # Construct SQL query based on tier
        query = "SELECT * FROM processed_documents WHERE 1=1"
        params = []
        
        if tier == 1:
            query += " AND total_words <= 600 AND total_words > 0"
        elif tier == 2:
            query += " AND total_words > 600 AND total_words <= 2500"
        elif tier == 3:
            query += " AND total_words > 2500 AND total_words <= 20000"
        elif tier == 4:
            query += " AND total_words > 20000"
            
        if limit:
            query += " LIMIT ?"
            params.append(limit)
            
        # Connect to database and execute query
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        
        # Convert rows to Document objects
        documents = []
        for row in rows:
            doc = Document(
                id=row['id'],
                celex_number=row['celex_number'],
                html_url=row['html_url'],
                total_words=row['total_words'],
                summary=row['summary'],
                summary_word_count=row['summary_word_count'],
                compression_ratio=row['compression_ratio']
            )
            documents.append(doc)
            
        logger.info(f"Loaded {len(documents)} documents")
        
        if not documents:
            logger.warning("No documents found to process!")
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tables already exist, skipping creation
        
        processed_docs = []
        for doc in documents:
            try:
                logger.info(f"Processing document {doc.celex_number}")
                if not doc.total_words:
                    continue
                
                total_word_count = doc.total_words
                tier = self._get_document_tier(total_word_count)
                
                if tier == 1:  # Process entire document at once for Tier 1
                    logger.info(f"Processing Tier 1 document {doc.celex_number}")
                    
                    # Get document content from database
                    logger.info(f"Fetching content for document {doc.celex_number}")
                    cursor.execute("SELECT content FROM document_sections WHERE document_id = ?", (doc.id,))
                    sections = cursor.fetchall()
                    if not sections:
                        logger.warning(f"No sections found for document {doc.celex_number}")
                        continue
                        
                    # Combine all sections into one text
                    text = "\n\n".join(section[0] for section in sections)
                    summary = self._process_tier_1(text)
                    
                elif tier == 2:  # Two-step summarization for Tier 2
                    summary = self._process_tier_2(doc, cursor)
                    
                elif tier == 3:  # Hierarchical summarization for Tier 3
                    cursor.execute("SELECT content FROM document_sections WHERE document_id = ?", (doc.id,))
                    sections = cursor.fetchall()
                    if not sections:
                        logger.warning(f"No sections found for document {doc.celex_number}")
                        continue
                    text = "\n\n".join(section[0] for section in sections)
                    summary = self._process_tier_3(text)
                    
                elif tier == 4:  # Advanced hierarchical for Tier 4
                    cursor.execute("SELECT content FROM document_sections WHERE document_id = ?", (doc.id,))
                    sections = cursor.fetchall()
                    if not sections:
                        logger.warning(f"No sections found for document {doc.celex_number}")
                        continue
                    text = "\n\n".join(section[0] for section in sections)
                    summary = self._process_tier_4(text)
                    
                else:
                    logger.warning(f"Unknown tier {tier} for document {doc.celex_number}")
                    continue

                if not summary:
                    logger.warning(f"Failed to generate summary for document {doc.celex_number}")
                    continue
                    
                # Update document with summary
                doc.summary = summary
                doc.summary_word_count = len(summary.split())
                doc.compression_ratio = doc.summary_word_count / doc.total_words if doc.total_words > 0 else 0
                
                logger.info(f"Generated summary for {doc.celex_number} with {doc.summary_word_count} words (compression ratio: {doc.compression_ratio:.2f})")
                
                # Store document summary in database
                cursor.execute("""
                    UPDATE processed_documents
                    SET summary = ?,
                        summary_word_count = ?,
                        compression_ratio = ?
                    WHERE id = ?
                """, (doc.summary, doc.summary_word_count, doc.compression_ratio, doc.id))
                conn.commit()
                
                processed_docs.append({
                    'celex_number': doc.celex_number,
                    'total_words': doc.total_words,
                    'summary_word_count': doc.summary_word_count,
                    'compression_ratio': doc.compression_ratio
                })

            except Exception as e:
                logger.error(f"Error processing document {doc.celex_number}: {str(e)}")
                continue
                
        conn.close()
        return processed_docs
