"""Utilities for chunking text in a way that respects document structure."""

import re
from typing import List
import nltk
from transformers import AutoTokenizer

# Initialize tokenizer once for efficiency
_tokenizer = None

def get_tokenizer():
    """Get or initialize the LexLM tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained('lexlm/legal-roberta-base')
    return _tokenizer

def chunk_text(text: str, max_tokens: int = 514) -> List[str]:
    """Split text into chunks that fit within model context window while preserving document structure.
    
    The chunking process follows this priority:
    1. Document sections (marked by headers, e.g., 'Article', 'Section', etc.)
    2. Paragraph breaks (double newlines)
    3. Sentence boundaries
    4. Word boundaries (if needed)
    
    Args:
        text: Text to split
        max_tokens: Maximum number of tokens per chunk (default: 514 for LexLM)
        
    Returns:
        List of text chunks that preserve document structure
    """
    tokenizer = get_tokenizer()
    
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))
    
    if not text:
        return []
        
    # Check if text fits in one chunk
    if count_tokens(text) <= max_tokens:
        return [text]
    
    # First try to split by document sections
    section_patterns = [
        r'(?:\n|^)\s*(?:SECTION|Article|ARTICLE|Chapter|CHAPTER)\s+\d+[:\.]',
        r'(?:\n|^)\s*\d+\.\s+[A-Z]',  # Numbered sections
        r'(?:\n|^)\s*[A-Z][A-Z\s]+(?:\n|$)'  # ALL CAPS headers
    ]
    
    for pattern in section_patterns:
        sections = [s.strip() for s in re.split(pattern, text) if s.strip()]
        if len(sections) > 1:
            chunks = []
            for section in sections:
                if count_tokens(section) > max_tokens:
                    # Recursively process long sections
                    chunks.extend(chunk_text(section, max_tokens))
                else:
                    chunks.append(section)
            return chunks
    
    # Try paragraph breaks
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if len(paragraphs) > 1:
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = count_tokens(para)
            if current_tokens + para_tokens <= max_tokens:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para
                current_tokens += para_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if para_tokens > max_tokens:
                    # Recursively process long paragraphs
                    chunks.extend(chunk_text(para, max_tokens))
                else:
                    current_chunk = para
                    current_tokens = para_tokens
        
        if current_chunk:
            chunks.append(current_chunk)
        return chunks
    
    # Fall back to sentence boundaries
    try:
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sent in sentences:
            sent_tokens = count_tokens(sent)
            if current_tokens + sent_tokens <= max_tokens:
                current_chunk = current_chunk + " " + sent if current_chunk else sent
                current_tokens += sent_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if sent_tokens > max_tokens:
                    # Split long sentences at space boundaries
                    words = sent.split()
                    current_chunk = ""
                    current_tokens = 0
                    for word in words:
                        word_tokens = count_tokens(word)
                        if current_tokens + word_tokens <= max_tokens:
                            current_chunk = current_chunk + " " + word if current_chunk else word
                            current_tokens += word_tokens
                        else:
                            chunks.append(current_chunk)
                            current_chunk = word
                            current_tokens = word_tokens
                else:
                    current_chunk = sent
                    current_tokens = sent_tokens
        
        if current_chunk:
            chunks.append(current_chunk)
        return chunks
        
    except Exception as e:
        # If NLTK fails, fall back to simple space-based chunking
        words = text.split()
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for word in words:
            word_tokens = count_tokens(word)
            if current_tokens + word_tokens <= max_tokens:
                current_chunk = current_chunk + " " + word if current_chunk else word
                current_tokens += word_tokens
            else:
                chunks.append(current_chunk)
                current_chunk = word
                current_tokens = word_tokens
        
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

def get_chunk_size(text_length: int, min_size: int = 100) -> int:
    """Determine appropriate chunk size based on text length.
    
    Args:
        text_length: Total length of text in words
        min_size: Minimum chunk size in words (default 100)
        
    Returns:
        Appropriate chunk size in words
        
    The function aims to create chunks that are:
    - At least min_size words (unless text is shorter)
    - At most 400 words (to stay under token limits)
    - Balanced across the text
    """
    if text_length <= min_size:
        return text_length  # If text is shorter than min_size, use full text
        
    # For longer texts, try to create balanced chunks
    max_size = 400  # Maximum chunk size
    
    # Calculate number of chunks needed
    num_chunks = max(1, text_length // max_size)
    
    # Get balanced chunk size
    chunk_size = text_length // num_chunks
    
    # Ensure chunk size is between min and max
    return max(min_size, min(chunk_size, max_size))
