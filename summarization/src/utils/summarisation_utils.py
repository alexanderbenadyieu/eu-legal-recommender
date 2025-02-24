"""
Utility functions for summarisation.

This module provides helper functions for text processing and dummy summarisation. In production, they can be replaced or integrated with real model API calls.
"""

def clean_text(text: str) -> str:
    """Clean text by removing duplicates and unnecessary whitespace.
    
    Args:
        text: Text to clean
        
    Returns:
        str: Cleaned text
    """
    # Split into lines and remove empty lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Remove duplicated lines while preserving order
    seen = set()
    unique_lines = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)
    
    # Join lines back together
    return '\n'.join(unique_lines)


def get_word_count(text: str) -> int:
    """Return the number of words in the text."""
    return len(text.split())


def abstractive_summarise(text: str, min_length: int, max_length: int, model: str = "BART") -> str:
    """Simulate abstractive summarisation by returning a truncated version of the text.
    In production, this would call the appropriate model API (e.g., BART or LongformerBART).
    """
    words = text.split()
    summary_words = words[:min_length] if len(words) >= min_length else words
    return f"[{model} summary ({min_length}-{max_length} words)]: " + ' '.join(summary_words) + "..."


def extractive_summarise(text: str, target_length: int) -> str:
    """Simulate extractive summarisation by selecting the first target_length words.
    In production, a more sophisticated extraction would be applied.
    """
    words = text.split()
    return ' '.join(words[:target_length])
