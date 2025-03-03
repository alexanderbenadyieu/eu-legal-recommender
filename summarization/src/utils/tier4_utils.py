"""Utilities for processing Tier 4 documents (20,000-68,000 words).

This module implements the section-based pre-summarization approach for long documents:
1. Process each section based on length:
   - <750 words: Direct BART-base summary (max 350 words)
   - 750-1500 words: LexLM extraction to ~600 words, then BART-base summary (max 350 words)
   - >1500 words: Split into subsections, then process each as above
2. Apply weighted extraction across all chunks
3. Generate final summary using main BART model
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging
from transformers import pipeline
import torch
from nltk.tokenize import sent_tokenize, word_tokenize

from .text_chunking import chunk_text

logger = logging.getLogger(__name__)

@dataclass
class Section:
    """Represents a document section."""
    id: int
    title: str
    content: str
    section_type: str
    section_order: int
    word_count: int

@dataclass
class ProcessedChunk:
    """Represents a processed chunk of text."""
    text: str
    word_count: int
    weight: float
    section_id: int

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

def split_into_subsections(text: str, min_words: int = 750, max_words: int = 1500) -> List[str]:
    """Split text into subsections of 750-1500 words.
    
    Uses a similar approach to chunk_text but with different target sizes.
    Tries to split on natural boundaries (line breaks, periods, spaces).
    
    Args:
        text: Text to split
        min_words: Minimum words per subsection
        max_words: Maximum words per subsection
        
    Returns:
        List of subsections
    """
    # First split by double newlines to preserve section structure
    paragraphs = text.split("\n\n")
    current_subsection = []
    current_word_count = 0
    subsections = []
    
    for para in paragraphs:
        para_words = len(word_tokenize(para))
        
        # If adding this paragraph would exceed max_words, store current subsection
        if current_word_count + para_words > max_words and current_word_count >= min_words:
            subsections.append(" ".join(current_subsection))
            current_subsection = []
            current_word_count = 0
            
        current_subsection.append(para)
        current_word_count += para_words
    
    # Handle remaining text
    if current_subsection:
        # If it's too big, split further by sentences
        if current_word_count > max_words:
            sentences = sent_tokenize(" ".join(current_subsection))
            current_subsection = []
            current_word_count = 0
            
            for sent in sentences:
                sent_words = len(word_tokenize(sent))
                if current_word_count + sent_words > max_words and current_word_count >= min_words:
                    subsections.append(" ".join(current_subsection))
                    current_subsection = []
                    current_word_count = 0
                current_subsection.append(sent)
                current_word_count += sent_words
                
        if current_subsection:
            subsections.append(" ".join(current_subsection))
    
    return subsections

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
        summary_words = len(word_tokenize(summary))
        chunks.append(ProcessedChunk(
            text=summary,
            word_count=summary_words,
            weight=1.2,  # Higher weight for shorter sections
            section_id=section.id
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
            section_id=section.id
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
                section_id=section.id
            ))
    
    return chunks

def refine_extraction(chunks: List[ProcessedChunk], target_length: int) -> List[Tuple[str, float]]:
    """Refine the extraction using weighted compression.
    
    Args:
        chunks: List of processed chunks
        target_length: Target length in words
        
    Returns:
        List of (chunk_text, extraction_percentage) tuples
    """
    total_words = sum(chunk.word_count for chunk in chunks)
    base_compression = target_length / total_words
    
    # Initial extraction percentages
    extractions = []
    total_weighted_words = 0
    
    # First pass - apply weights and clamp
    for chunk in chunks:
        # Apply weight to base compression
        percentage = base_compression * chunk.weight
        # Clamp between 15% and 35%
        percentage = max(0.15, min(0.35, percentage))
        weighted_words = chunk.word_count * percentage
        total_weighted_words += weighted_words
        extractions.append((chunk.text, percentage))
    
    # Second pass - adjust to hit target length
    scale_factor = target_length / total_weighted_words
    
    # Apply scaling while respecting min/max bounds
    scaled_extractions = []
    total_scaled_words = 0
    
    for i, (text, pct) in enumerate(extractions):
        # Scale percentage but respect bounds
        scaled_pct = max(0.15, min(0.35, pct * scale_factor))
        
        # Calculate words after scaling
        scaled_words = chunks[i].word_count * scaled_pct
        total_scaled_words += scaled_words
        
        scaled_extractions.append((text, scaled_pct))
    
    # Final adjustment if still outside target range
    if abs(total_scaled_words - target_length) > target_length * 0.15:
        final_scale = target_length / total_scaled_words
        scaled_extractions = [
            (text, max(0.15, min(0.35, pct * final_scale)))
            for text, pct in scaled_extractions
        ]
    
    return scaled_extractions
