import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from summarization.src.utils.database_utils import load_documents, Document
from summarization.src.preprocessing.html_parser import LegalDocumentParser
from summarization.src.utils.text_chunking import chunk_text
from summarization.src.extractive.lexlm_wrapper import LexLMExtractor
from summarization.src.abstractive.bart_finetuner import BartFineTuner

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
        self.config = config['chunking']
        self.html_parser = LegalDocumentParser(Path(db_path).parent)
        self.extractor = LexLMExtractor()
        self.generator = BartFineTuner()
        
    def process_documents(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process documents through the pipeline:
        1. Load documents from database
        2. Parse HTML content
        3. Chunk the sections
        4. Generate summaries for each section
        
        Args:
            limit: Optional limit on number of documents to process
            
        Returns:
            List of processed documents with their chunks and summaries
        """
        # Load documents from database
        logger.info("Loading documents from database...")
        documents = load_documents(self.db_path, limit)
        logger.info(f"Loaded {len(documents)} documents")
        
        processed_docs = []
        for doc in documents:
            try:
                # Parse HTML content into sections
                logger.info(f"Processing document {doc.celex_number}")
                sections = self.html_parser.parse_html_content(doc.content_html)
                
                # Process each section
                processed_sections = []
                for section in sections:
                    # Chunk the section text
                    chunks = chunk_text(
                        section.content,
                        max_chunk_size=self.config['max_chunk_size']
                    )
                    
                    # Generate summary if section is long enough
                    summary = None
                    if len(chunks) > 1:
                        # Extract key sentences first
                        extracted = self.extractor.extract_key_sentences('\n'.join(chunks))
                        summary = self.generator.summarize(extracted)
                    elif len(chunks) == 1:
                        summary = self.generator.summarize(chunks[0])
                    
                    processed_sections.append({
                        'title': section.title,
                        'level': 1,  # We'll get proper level from section type later
                        'chunks': chunks,
                        'summary': summary
                    })
                
                processed_doc = {
                    'document_id': doc.document_id,
                    'celex_number': doc.celex_number,
                    'title': doc.title,
                    'sections': processed_sections
                }
                processed_docs.append(processed_doc)
                
            except Exception as e:
                logger.error(f"Error processing document {doc.celex_number}: {str(e)}")
                continue
                
        return processed_docs