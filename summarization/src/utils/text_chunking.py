"""Utilities for chunking text in a way that respects document structure."""

import re
from typing import List

def split_by_separator(text: str, separator: str, max_chunk_size: int) -> List[str]:
    """Split text by a given separator pattern."""
    # Use regex split
    segments = re.split(f'({separator})', text)
    
    # Recombine segments with their separators
    segments = [segments[i] + (segments[i+1] if i+1 < len(segments) else '') 
               for i in range(0, len(segments), 2)]
    
    # Combine segments while respecting max_chunk_size
    chunks = []
    current_chunk = ""
    
    for segment in segments:
        if len(current_chunk) + len(segment) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = segment
        else:
            current_chunk += segment
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def chunk_text(text: str, max_chunk_size: int) -> List[str]:
    """Split text into chunks using a priority list of separators.
    
    Args:
        text: Text to split
        max_chunk_size: Maximum size of each chunk in characters
    
    Returns:
        List of text chunks
    """
    # If text is already small enough, return it as is
    if len(text) <= max_chunk_size:
        return [text]
    
    # List of separators in order of priority
    separators = [
        "\n",      # Newlines first
        "\t",      # Then tabs
        "[...]",   # Special markers
        ".\\s+",    # Sentences (period followed by whitespace)
        "(?<=[?!])",  # Question/exclamation marks (lookbehind)
        "...",     # Ellipsis
        ";",       # Semicolons
        ":",       # Colons
        " - "      # Dashes with spaces
    ]
    
    # Try each separator in order until we get chunks that are small enough
    for separator in separators:
        chunks = split_by_separator(text, separator, max_chunk_size)
        if all(len(chunk) <= max_chunk_size for chunk in chunks):
            return chunks
    
    # If no separator worked well, fall back to character-level chunking
    chunks = []
    for i in range(0, len(text), max_chunk_size):
        chunks.append(text[i:i + max_chunk_size])
    return chunks

def get_chunk_size(text_length: int) -> int:
    """Determine appropriate chunk size based on text length.
    
    We use a maximum of 300 words per chunk to ensure we stay well under the
    514 token limit (assuming average English word is ~1.3 tokens).
    """
    return min(300, text_length)  # Always use 300 words max per chunk
