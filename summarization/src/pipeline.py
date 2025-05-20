"""
Multi-tier summarization pipeline for EU legal documents.

This module implements the core summarization pipeline for the EU Legal Recommender system,
using a tiered approach based on document length:

- Tier 1 (0-600 words): Direct abstractive summarization
- Tier 2 (601-2,500 words): Two-step summarization with extractive then abstractive
- Tier 3 (2,501-20,000 words): Hierarchical summarization with section-aware processing
- Tier 4 (20,000+ words): Advanced hierarchical summarization with weighted extraction

The pipeline handles document retrieval from the database, processing through the appropriate
tier-specific summarization strategy, and storing the results back in the database.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import nltk

# Ensure nltk punkt is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Import system modules
import sys
import os

# Add the project root to the Python path to access all modules correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import local modules directly
from src.utils.database_utils import Document
from src.preprocessing.html_parser import LegalDocumentParser
from src.utils.text_chunking import chunk_text
from src.extractive.lexlm_wrapper import LexLMExtractor
from src.abstractive.bart_finetuner import BartFineTuner
from src.utils.config import get_config

# Import database utilities from project root
sys.path.insert(0, os.path.abspath(os.path.join(project_root, '..'))) 
from database_utils import get_db_connection, save_document_summary, get_document_sections

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SummarizationPipeline:
    def __init__(self, db_type: str = 'consolidated', config: Dict[str, Any] = None):
        """
        Initialize the summarization pipeline.
        
        Args:
            db_type: Type of database to use ('consolidated' or 'legacy')
            config: Configuration dictionary containing chunking parameters
                    If None, loads from centralized configuration system
        """
        self.db_type = db_type
        
        # Use provided config or load from centralized configuration system
        if config is None:
            config = get_config()
            logger.info("Using centralized configuration system")
            
        self.full_config = config  # Store full config
        self.config = config.get('chunking', {})
        
        # Initialize components
        self.html_parser = LegalDocumentParser(Path.cwd())
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

    def _process_tier_2(self, document: Document, conn: sqlite3.Connection) -> str:
        """Process Tier 2 document (600-2,500 words) - Two-step summarization.
        
        Process:
        1. Split text into chunks that fit within LexLM context (514 tokens)
        2. Extract K words where K = max(300, min(0.3 × D, 600))
        3. Generate final summary of 0.6K to 0.8K words
        """
        logger.info(f"Processing Tier 2 document {document.celex_number} by sections")
        
        # Get document sections from database using utility function
        sections = get_document_sections(document_id=document.id, db_type=self.db_type)
        if not sections:
            logger.warning(f"No sections found for document {document.celex_number}")
            return ""
            
        # Combine all sections into one text
        text = "\n\n".join(section['content'] for section in sections)
        
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
        
        # Get appropriate model parameters based on word count
        min_length, max_length = self.generator._get_tier_parameters(extracted_words)
        
        # Ensure extracted text fits within BART's context window (1024 tokens)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        tokens = tokenizer(extracted, truncation=True, max_length=1024, return_tensors="pt")
        truncated_text = tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
        
        # Generate summary using the tier2 model
        summary = self.generator.tier2_model(
            truncated_text,
            min_length=min_length,
            max_length=max_length,
            do_sample=False
        )[0]['summary_text']
        
        # Update document with summary statistics
        summary_words = len(word_tokenize(summary))
        document.summary = summary
        document.summary_word_count = summary_words
        document.compression_ratio = summary_words / word_count
        
        return summary

    def _process_tier_3(self, document: Document, conn: sqlite3.Connection) -> str:
        """Process Tier 3 document (2,500-20,000 words) - Hierarchical summarization.
        
        This implements the new section-aware hierarchical summarization process:
        1. Process each section to target length (~350 words)
        2. Apply weighted compression to reach intermediate target (600-750 words)
        3. Generate final abstractive summary (480-600 words)
        """
        from .utils.tier_processing import Section, process_tier3_section as process_section, refine_extraction
        from nltk.tokenize import word_tokenize
        
        logger.info(f"Processing Tier 3 document {document.celex_number} using section-aware approach")
        
        # Get document sections from database
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, title, content, section_type, section_order, word_count 
            FROM document_sections 
            WHERE document_id = ? 
            ORDER BY section_order""", 
            (document.id,)
        )
        sections_data = cursor.fetchall()
        
        if not sections_data:
            logger.warning(f"No sections found for document {document.celex_number}")
            return ""
            
        # Convert to Section objects
        sections = [
            Section(id=row[0], title=row[1], content=row[2], 
                   section_type=row[3], section_order=row[4], 
                   word_count=row[5])
            for row in sections_data
        ]
        
        # Step 1: Process each section
        logger.info(f"Processing {len(sections)} sections")
        all_chunks = []
        for section in sections:
            processed_chunks = process_section(section)
            all_chunks.extend(processed_chunks)
            
        # Step 2: Refine to reach target length (600-750 words)
        target_length = 700  # Target middle of 600-750 range
        logger.info(f"Refining extraction to target length of {target_length} words")
        
        chunk_extractions = refine_extraction(all_chunks, target_length)
        
        # Extract from each chunk using LexLM
        extracted_texts = []
        for chunk_text, extraction_percent in chunk_extractions:
            # Convert percentage to target word count
            chunk_words = len(chunk_text.split())
            target_words = int(chunk_words * extraction_percent)
            
            extracted = self.extractor.extract_key_sentences(
                chunk_text,
                target_length=target_words,
                tier=3
            )
            extracted_texts.append(extracted)
        
        # Combine extracted texts
        intermediate_text = ' '.join(extracted_texts)
        intermediate_words = len(word_tokenize(intermediate_text))
        logger.info(f"Generated intermediate text of {intermediate_words} words")
        
        # Step 3: Final abstractive summarization
        min_summary_length = 480
        max_summary_length = 600
        
        # Ensure text fits within BART's context window
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        tokens = tokenizer(intermediate_text, truncation=True, max_length=1024, return_tensors="pt")
        truncated_text = tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
        
        logger.info(f"Generating final summary (target length: {min_summary_length}-{max_summary_length} words)")
        summary = self.generator.tier3_model(
            truncated_text,
            min_length=min_summary_length,
            max_length=max_summary_length,
            do_sample=False
        )[0]['summary_text']
        
        # Update document with summary statistics
        summary_words = len(word_tokenize(summary))
        document.summary = summary
        document.summary_word_count = summary_words
        document.compression_ratio = summary_words / document.total_words
        
        return summary

    def _process_tier_4(self, text: str) -> str:
        """Process Tier 4 document (20,000+ words) - Advanced hierarchical.
        
        Step 1: Section-Based Pre-Summarization
        - Sections <750 words: Summarize to ≤350 words using BART
        - Sections 750-1500 words: Extract to 600 words, then summarize to ≤350
        - Sections >1500 words: Split into subsections, extract each to 600, summarize to ≤350
        
        Step 2: Global Dependent-Ratio Extraction
        - Compute compression factor f = target/total (minimum 0.15)
        - Apply weighted extraction across chunks with decreasing weights
        - Target final extraction of ~750 words
        
        Step 3: Final Abstractive Summarization
        - Generate 480-600 word summary from the ~750 word extraction
        """
        from nltk.tokenize import word_tokenize
        import re
        from .utils.tier_processing import (
            Section, ProcessedChunk, split_into_subsections, get_chunk_weights
        )
        
        def count_words(text: str) -> int:
            """Count words in text."""
            return len(word_tokenize(text))
            
        def split_into_sections(text: str, max_section_words: int = 3000) -> list:
            """Split text into sections of approximately equal size."""
            words = count_words(text)
            
            if words <= max_section_words:
                return [text]
                
            # Try to split on section markers
            section_markers = [
                r'\n\s*Article\s+\d+\s*\n',
                r'\n\s*Section\s+\d+\s*\n',
                r'\n\s*CHAPTER\s+\d+\s*\n',
                r'\n\s*TITLE\s+\d+\s*\n',
                r'\n\s*\d+\.\s+',  # Numbered paragraphs
                r'\n\n'  # Double line breaks as last resort
            ]
            
            sections = [text]
            
            for marker in section_markers:
                new_sections = []
                for section in sections:
                    if count_words(section) > max_section_words:
                        # Try to split on this marker
                        parts = re.split(marker, section)
                        if len(parts) > 1:
                            # Recombine with the markers
                            reconstructed = []
                            for i, part in enumerate(parts):
                                if i == 0:
                                    reconstructed.append(part)
                                else:
                                    match = re.search(marker, section)
                                    if match:
                                        marker_text = match.group(0)
                                        reconstructed.append(marker_text + part)
                                    else:
                                        reconstructed.append(part)
                            new_sections.extend(reconstructed)
                        else:
                            new_sections.append(section)
                    else:
                        new_sections.append(section)
                sections = new_sections
            return sections
            
        def process_section(section_text: str) -> str:
            """Process a single section based on its length."""
            words = count_words(section_text)
            
            if words <= 750:
                # Direct BART summarization
                return self.generator.summarize_tier4_section(section_text, max_length=350)
            elif words <= 1500:
                # Extract to 600, then summarize to 350
                extracted = self.extractor.extract_key_sentences(
                    section_text,
                    target_length=600,
                    tier=4
                )
                return self.generator.summarize_tier4_section(extracted, max_length=350)
            else:
                # Split into subsections using the utility function
                subsections = split_into_subsections(section_text)
                processed_subsections = []
                
                for i, subsec in enumerate(subsections):
                    # Extract from subsection
                    extracted = self.extractor.extract_key_sentences(
                        subsec,
                        target_length=600,
                        tier=4
                    )
                    # Summarize extracted text
                    summary = self.generator.summarize_tier4_section(extracted, max_length=350)
                    processed_subsections.append(summary)
                
                return '\n'.join(processed_subsections)
        
        # Step 1: Section-Based Pre-Summarization
        sections = split_into_sections(text)
        logger.info(f"Split document into {len(sections)} sections")
        
        chunks = []
        for i, section in enumerate(sections, 1):
            logger.info(f"Processing section {i}/{len(sections)}")
            chunk = process_section(section)
            chunks.append(chunk)
        
        # Step 2: Global Dependent-Ratio Extraction
        combined = '\n'.join(chunks)
        combined_words = count_words(combined)
        target_length = 750
        
        # Compute compression factor
        f = max(0.15, target_length / combined_words)
        
        # Define chunk weights
        def get_weight(index: int) -> float:
            if index == 0: return 1.2
            elif index == 1: return 1.0
            elif index == 2: return 0.8
            elif index == 3: return 0.6
            elif index == 4: return 0.5
            else: return 0.5
        
        # Extract from each chunk with weights
        weighted_chunks = []
        total_target_words = 0
        
        for i, chunk in enumerate(chunks):
            chunk_words = count_words(chunk)
            weight = get_weight(i)
            extraction_rate = max(0.15, min(0.35, f * weight))  # Clamp between 15-35%
            target_words = int(chunk_words * extraction_rate)
            total_target_words += target_words
            
            extracted = self.extractor.extract_key_sentences(
                chunk,
                target_length=target_words,
                tier=4
            )
            weighted_chunks.append(extracted)
        
        # Combine weighted extractions
        final_extraction = ' '.join(weighted_chunks)
        final_words = count_words(final_extraction)
        logger.info(f"Final extraction: {final_words} words")
        
        # Step 3: Final Abstractive Summarization (480-600 words)
        return self.generator.summarize(final_extraction)

    def process_documents(self, limit: Optional[int] = None, tier: Optional[int] = None) -> List[Dict[str, Any]]:
        """Process documents through the tiered pipeline and store results in database.
        
        Args:
            limit: Optional limit on number of documents to process
            tier: Optional tier to process (1: ≤600 words, 2: 601-2500 words, 3: 2501-20000 words, 4: >20000 words)
        """
        logger.info("Loading documents from database...")
        
        # Connect to database using utility function
        conn = get_db_connection(db_type=self.db_type)
        conn.row_factory = sqlite3.Row
        
        # Construct SQL query based on tier
        if tier == 1:
            tier_min, tier_max = 0, 600
        elif tier == 2:
            tier_min, tier_max = 601, 2500
        elif tier == 3:
            tier_min, tier_max = 2501, 20000
        elif tier == 4:
            tier_min, tier_max = 20001, 1000000  # Large upper bound for tier 4
        else:
            tier_min, tier_max = 0, 1000000  # Process all documents
        
        # Query for documents that need summarization
        if self.db_type == 'consolidated':
            query = """
                SELECT document_id as id, celex_number, html_url, total_words, 
                       summary, summary_word_count, compression_ratio
                FROM documents
                WHERE total_words BETWEEN ? AND ?
                AND summary IS NULL
                ORDER BY total_words
            """
        else:
            # Legacy database query
            query = """
                SELECT * FROM processed_documents 
                WHERE total_words BETWEEN ? AND ?
                AND summary IS NULL
                ORDER BY total_words
            """
            
        if limit:
            query += " LIMIT ?"
            params = [tier_min, tier_max, limit]
        else:
            params = [tier_min, tier_max]
            
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
        
        # We already have the database connection
        
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
                    sections = get_document_sections(document_id=doc.id, db_type=self.db_type)
                    if not sections:
                        logger.warning(f"No sections found for document {doc.celex_number}")
                        continue
                        
                    # Combine all sections into one text
                    text = "\n\n".join(section['content'] for section in sections)
                    summary = self._process_tier_1(text)
                    
                elif tier == 2:  # Two-step summarization for Tier 2
                    summary = self._process_tier_2(doc, conn)
                    
                elif tier == 3:  # Hierarchical summarization for Tier 3
                    summary = self._process_tier_3(doc, conn)
                    
                elif tier == 4:  # Advanced hierarchical for Tier 4
                    sections = get_document_sections(document_id=doc.id, db_type=self.db_type)
                    if not sections:
                        logger.warning(f"No sections found for document {doc.celex_number}")
                        continue
                    text = "\n\n".join(section['content'] for section in sections)
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
                
                # Store document summary in database using utility function
                save_document_summary(
                    celex_number=doc.celex_number,
                    summary=doc.summary,
                    summary_word_count=doc.summary_word_count,
                    total_words=doc.total_words,
                    compression_ratio=doc.compression_ratio,
                    tier=tier,
                    db_type=self.db_type
                )
                
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
