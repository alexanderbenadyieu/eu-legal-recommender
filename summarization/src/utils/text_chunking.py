"""Utilities for chunking text in a way that respects document structure."""

import re
from typing import List
import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK's punkt tokenizer."""
    return sent_tokenize(text)

def get_section_boundaries(text: str) -> List[int]:
    """Find potential section boundaries in legal text.
    
    Looks for common legal document section markers like:
    - Article X
    - Section X
    - Chapter X
    - Numbered lists (1., 2., etc.)
    """
    patterns = [
        r'Article \d+',
        r'Section \d+',
        r'Chapter \d+',
        r'\n\d+\.',  # Numbered lists
        r'\([a-z]\)',  # (a), (b), etc.
        r'\n[A-Z][A-Z\s]+[A-Z]\n'  # All caps headers
    ]
    
    boundaries = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            boundaries.append(match.start())
    
    return sorted(list(set(boundaries)))

def chunk_text(text: str, target_chunk_size: int, overlap: int = 50, max_words: int = 300) -> List[str]:
    """Split text into chunks respecting sentence and section boundaries.
    
    Args:
        text: Text to split
        target_chunk_size: Target size of each chunk in words
        overlap: Number of words to overlap between chunks
    
    Returns:
        List of text chunks
    """
    # Get sentences and their word counts
    sentences = split_into_sentences(text)
    sentence_word_counts = [len(sent.split()) for sent in sentences]
    
    # Get section boundaries
    section_starts = get_section_boundaries(text)
    
    # Create chunks
    chunks = []
    current_chunk = []
    current_word_count = 0
    last_section_start = 0
    
    for i, (sentence, word_count) in enumerate(zip(sentences, sentence_word_counts)):
        # Check if sentence starts a new section
        sentence_start = text.find(sentence, last_section_start)
        if any(abs(section_start - sentence_start) < 10 for section_start in section_starts):
            # If we have a non-empty chunk and hit a section boundary, start a new chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                # Add overlap by keeping some sentences from previous chunk
                overlap_sentences = []
                overlap_words = 0
                for sent in reversed(current_chunk):
                    if overlap_words + len(sent.split()) > overlap:
                        break
                    overlap_sentences.insert(0, sent)
                    overlap_words += len(sent.split())
                current_chunk = overlap_sentences
                current_word_count = overlap_words
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_word_count += word_count
        
        # If chunk is full or would exceed max words, start a new one
        if current_word_count >= target_chunk_size or current_word_count >= max_words:
            chunks.append(" ".join(current_chunk))
            # Add overlap
            overlap_sentences = []
            overlap_words = 0
            for sent in reversed(current_chunk):
                if overlap_words + len(sent.split()) > overlap:
                    break
                overlap_sentences.insert(0, sent)
                overlap_words += len(sent.split())
            current_chunk = overlap_sentences
            current_word_count = overlap_words
        
        last_section_start = sentence_start + len(sentence)
    
    # Add final chunk if non-empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def get_chunk_size(text_length: int) -> int:
    """Determine appropriate chunk size based on text length.
    
    We use a maximum of 300 words per chunk to ensure we stay well under the
    514 token limit (assuming average English word is ~1.3 tokens).
    """
    return min(300, text_length)  # Always use 300 words max per chunk
