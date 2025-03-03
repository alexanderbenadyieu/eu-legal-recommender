"""Utilities for Tier 3 document summarization process."""

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
    
    Args:
        num_chunks: Number of chunks to generate weights for
        
    Returns:
        List of weights where earlier chunks have higher weights
    """
    if num_chunks <= 0:
        return []
        
    # Define base weights for first 5 chunks
    base_weights = [1.2, 1.0, 0.8, 0.6, 0.5]
    
    if num_chunks <= len(base_weights):
        return base_weights[:num_chunks]
    
    # For additional chunks, use 0.5
    return base_weights + [0.5] * (num_chunks - len(base_weights))

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
    raw_percent = compression_factor * weight
    return np.clip(raw_percent, min_percent, max_percent)

def process_section(section: Section, target_length: int = 350) -> List[ProcessedChunk]:
    """Process a section according to its length category.
    
    Args:
        section: Section to process
        target_length: Target length for processed section
        
    Returns:
        List of processed chunks with their weights
    """
    chunks = []
    
    if section.word_count <= 350:
        # Use section as is
        chunks = [section.content]
    elif section.word_count <= 750:
        # Split into chunks ≤350 words
        chunks = split_section_into_chunks(section, max_chunk_words=350)
    elif section.word_count <= 1500:
        # Process as whole but aim for 350 words
        chunks = [section.content]
    else:
        # Split into subsections
        subsections = split_section_into_chunks(section, max_chunk_words=750)
        for subsec in subsections:
            if len(subsec.split()) > 350:
                chunks.extend(split_section_into_chunks(
                    Section(section.id, None, subsec, section.section_type, 
                           section.section_order, len(subsec.split())),
                    max_chunk_words=350
                ))
            else:
                chunks.append(subsec)
    
    # Get weights for chunks
    weights = get_chunk_weights(len(chunks))
    
    # Create ProcessedChunk objects
    return [
        ProcessedChunk(
            text=chunk,
            word_count=len(chunk.split()),
            weight=weight,
            original_section_id=section.id
        )
        for chunk, weight in zip(chunks, weights)
    ]

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
    if total_weighted_words < target_length * 0.85:
        # Increase extraction if too low
        scale_factor = target_length / total_weighted_words
        extractions = [
            (text, min(0.35, pct * scale_factor))
            for text, pct in extractions
        ]
    elif total_weighted_words > target_length * 1.15:
        # Decrease extraction if too high
        scale_factor = target_length / total_weighted_words
        extractions = [
            (text, max(0.15, pct * scale_factor))
            for text, pct in extractions
        ]
    
    return extractions
