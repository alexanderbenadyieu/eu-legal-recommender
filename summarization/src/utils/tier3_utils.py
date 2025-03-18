"""Utilities for Tier 3 document summarization process."""

from typing import List, Tuple, Dict, Optional
import numpy as np
from .tier_utils import (
    Section, ProcessedChunk, split_section_into_chunks, 
    get_chunk_weights, compute_extraction_percentage, refine_extraction,
    split_into_subsections
)

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

# This function is now imported from tier_utils
