"""Common utilities for processing tiered documents.

This module provides shared functionality for processing documents across different tiers,
particularly for tier 3 and tier 4 documents which have similar processing patterns.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
from .text_chunking import chunk_text, get_chunk_size

@dataclass
class Section:
    """Represents a section in a document."""
    id: int
    title: Optional[str]
    content: str
    section_type: str
    section_order: int
    word_count: int

@dataclass
class ProcessedChunk:
    """Represents a processed chunk of text with its extraction weight."""
    text: str
    word_count: int
    weight: float
    original_section_id: int

def split_section_into_chunks(section: Section, max_chunk_words: int = 350) -> List[str]:
    """Split a section into chunks of approximately equal size.
    
    Args:
        section: Section object containing the text to split
        max_chunk_words: Maximum words per chunk
        
    Returns:
        List of text chunks
    """
    if section.word_count <= max_chunk_words:
        return [section.content]
        
    # Use existing chunking utility but with word-based size
    chunk_size = get_chunk_size(section.word_count, min_size=100)
    chunk_size = min(chunk_size, max_chunk_words)
    
    return chunk_text(section.content, chunk_size)

def get_chunk_weights(num_chunks: int) -> List[float]:
    """Get weight factors for chunks based on their position.
    
    Assigns higher weights to the beginning and end of the document,
    with a gradual decrease in the middle.
    
    Args:
        num_chunks: Number of chunks to generate weights for
        
    Returns:
        List of weight factors (sum = num_chunks)
    """
    if num_chunks <= 1:
        return [1.0]
    
    # Create a U-shaped weight distribution
    x = np.linspace(-1, 1, num_chunks)
    # Square function gives more weight to beginning and end
    weights = 1.5 - (x ** 2)
    
    # Normalize to ensure sum = num_chunks
    weights = weights * (num_chunks / weights.sum())
    
    return weights.tolist()

def compute_extraction_percentage(chunk_length: int, compression_factor: float, 
                                 weight: float, min_percent: float = 0.15, 
                                 max_percent: float = 0.35) -> float:
    """Compute the percentage of text to extract from a chunk.
    
    Args:
        chunk_length: Length of the chunk in words
        compression_factor: Overall compression factor
        weight: Weight factor for this chunk
        min_percent: Minimum extraction percentage
        max_percent: Maximum extraction percentage
        
    Returns:
        Extraction percentage (between min_percent and max_percent)
    """
    # Base percentage is the compression factor
    base_percentage = compression_factor
    
    # Apply weight factor
    weighted_percentage = base_percentage * weight
    
    # Ensure within bounds
    return max(min_percent, min(weighted_percentage, max_percent))

def refine_extraction(chunks: List[ProcessedChunk], target_length: int) -> List[Tuple[str, float]]:
    """Refine the extraction using weighted compression.
    
    Args:
        chunks: List of processed chunks
        target_length: Target length in words
        
    Returns:
        List of (chunk_text, extraction_percentage) tuples
    """
    total_words = sum(chunk.word_count for chunk in chunks)
    
    # Calculate base compression factor
    compression_factor = target_length / total_words if total_words > 0 else 0.5
    
    # Apply weights to determine extraction percentages
    extraction_info = []
    for chunk in chunks:
        extraction_percent = compute_extraction_percentage(
            chunk.word_count, 
            compression_factor,
            chunk.weight
        )
        extraction_info.append((chunk.text, extraction_percent))
    
    return extraction_info

def split_into_subsections(text: str, min_words: int = 750, max_words: int = 1500) -> List[str]:
    """Split text into subsections of specified word count range.
    
    Uses a similar approach to chunk_text but with different target sizes.
    Tries to split on natural boundaries (line breaks, periods, spaces).
    
    Args:
        text: Text to split
        min_words: Minimum words per subsection
        max_words: Maximum words per subsection
        
    Returns:
        List of subsections
    """
    from nltk.tokenize import sent_tokenize, word_tokenize
    
    # Count words in the text
    words = word_tokenize(text)
    word_count = len(words)
    
    if word_count <= max_words:
        return [text]
    
    # Split into sentences
    sentences = sent_tokenize(text)
    
    # Group sentences into subsections
    subsections = []
    current_section = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = len(word_tokenize(sentence))
        
        # If adding this sentence would exceed max_words and we're above min_words,
        # start a new subsection
        if current_word_count + sentence_words > max_words and current_word_count >= min_words:
            subsections.append(" ".join(current_section))
            current_section = [sentence]
            current_word_count = sentence_words
        else:
            current_section.append(sentence)
            current_word_count += sentence_words
    
    # Add the last subsection if it's not empty
    if current_section:
        subsections.append(" ".join(current_section))
    
    return subsections
