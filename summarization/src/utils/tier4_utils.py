"""Utilities for processing Tier 4 documents (20,000-68,000 words).

This module implements the section-based pre-summarization approach for long documents:
1. Process each section based on length:
   - <750 words: Direct BART-base summary (max 350 words)
   - 750-1500 words: LexLM extraction to ~600 words, then BART-base summary (max 350 words)
   - >1500 words: Split into subsections, then process each as above
2. Apply weighted extraction across all chunks
3. Generate final summary using main BART model
"""

from typing import List, Tuple, Optional
import logging
from transformers import pipeline
import torch

from .tier_utils import (
    Section, ProcessedChunk, split_section_into_chunks,
    get_chunk_weights, compute_extraction_percentage, refine_extraction,
    split_into_subsections
)

logger = logging.getLogger(__name__)

class BartBaseSummarizer:
    """Wrapper for facebook/bart-base model for section-level summarization."""
    
    def __init__(self):
        """Initialize the BART-base model and tokenizer."""
        try:
            self.model = pipeline(
                'summarization',
                model="facebook/bart-base",
                device=-1  # Use CPU
            )
            logger.info("Loaded BART-base model")
        except Exception as e:
            logger.error(f"Error loading BART-base model: {str(e)}")
            raise
        
    def summarize(self, text: str, max_length: int = 350) -> str:
        """Generate a summary using BART-base.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary in words
            
        Returns:
            Generated summary
        """
        try:
            if not text:
                logger.warning("Empty input text")
                return text
                
            # Generate summary
            result = self.model(
                text,
                max_length=max_length,
                min_length=int(max_length * 0.8),
                truncation=True
            )
            
            if not result or len(result) == 0:
                logger.warning("No summary generated, returning original text")
                return text
                
            return result[0]['summary_text']
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return text  # Return original text on error

# This function is now imported from tier_utils

def process_section(section: Section, extractor, summarizer: BartBaseSummarizer) -> List[ProcessedChunk]:
    """Process a section based on its length.
    
    Args:
        section: Section to process
        extractor: LexLM extractor instance
        summarizer: BART-base summarizer instance
        
    Returns:
        List of processed chunks
    """
    logger.info(f"Processing section {section.id} ({section.word_count} words)")
    chunks = []
    
    if section.word_count < 750:
        # Direct BART-base summary
        summary = summarizer.summarize(section.content, max_length=350)
        from nltk.tokenize import word_tokenize
        summary_words = len(word_tokenize(summary))
        chunks.append(ProcessedChunk(
            text=summary,
            word_count=summary_words,
            weight=1.2,  # Higher weight for shorter sections
            original_section_id=section.id
        ))
    
    elif section.word_count <= 1500:
        # Extract then summarize
        extracted = extractor.extract_key_sentences(
            section.content,
            target_length=600,
            tier=4
        )
        summary = summarizer.summarize(extracted, max_length=350)
        summary_words = len(word_tokenize(summary))
        chunks.append(ProcessedChunk(
            text=summary,
            word_count=summary_words,
            weight=1.0,
            original_section_id=section.id
        ))
    
    else:
        # Split into subsections, extract and summarize each
        subsections = split_into_subsections(section.content)
        for i, subsec in enumerate(subsections):
            # Extract from subsection
            extracted = extractor.extract_key_sentences(
                subsec,
                target_length=600,
                tier=4
            )
            # Summarize extracted text
            summary = summarizer.summarize(extracted, max_length=350)
            summary_words = len(word_tokenize(summary))
            # Decrease weight for later subsections
            weight = max(0.5, 1.0 - (i * 0.1))
            chunks.append(ProcessedChunk(
                text=summary,
                word_count=summary_words,
                weight=weight,
                original_section_id=section.id
            ))
    
    return chunks

# This function is now imported from tier_utils
