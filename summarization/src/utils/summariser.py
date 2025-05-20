"""
Summariser module for tiers 1 to 4 based on the multi-tier summarisation strategy.

Tiers:
    Tier 1: Single-Step Abstractive Summarisation (0-600 words) using adaptive compression ratios.
    Tier 2: Two-Step Summarisation (600-2500 words) with extractive then abstractive summarisation.
    Tier 3: Hierarchical Summarisation (2500-20,000 words) with fixed and dependent extraction followed by chained abstractive summarisation.
    Tier 4: Hierarchical Summarisation (20,000-68,000 words) with similar process but adjusted target lengths and models.
"""

import os
import math
import yaml
from summarization.src.utils.summarisation_utils import get_word_count
from nltk.tokenize import sent_tokenize

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config/summarisation_config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

from summarization.src.abstractive.bart_finetuner import BartFineTuner
from summarization.src.extractive.lexlm_wrapper import LexLMExtractor

# Initialize models
bart_model = BartFineTuner()
lexlm_model = LexLMExtractor()

def get_summary_range(word_count: int, tier_config: dict) -> tuple[int, int]:
    """Get min and max summary lengths based on document length and tier config."""
    if 'summary_ranges' in tier_config:  # Tier 1 style
        if word_count <= tier_config['thresholds']['short']:
            return tuple(tier_config['summary_ranges']['short'])
        elif word_count <= tier_config['thresholds']['medium']:
            return tuple(tier_config['summary_ranges']['medium'])
        else:
            return tuple(tier_config['summary_ranges']['long'])
    else:  # Tier 2-4 style
        return (tier_config['final_abstractive']['min'],
                tier_config['final_abstractive']['max'])

def summarise_tier1(text: str) -> str:
    """Tier 1 summarisation for texts up to 600 words.
    
    For very short texts (<=600 words), we skip the extractive step
    and directly use BART for abstractive summarization.
    """
    word_count = get_word_count(text)
    min_len, max_len = get_summary_range(word_count, config['tier1'])
    return bart_model.summarize(text, min_length=min_len, max_length=max_len)

def summarise_tier2(text: str) -> str:
    """Tier 2 summarisation for texts between 600 and 2500 words.
    
    Process:
    1. Split text into chunks that fit within LexLM context (514 tokens)
    2. Extract K words where K = max(300, min(0.3 × D, 600))
    3. Generate final summary of 0.6K to 0.8K words
    """
    from summarization.src.utils.text_chunking import chunk_text
    
    # Step 1: Split into chunks that fit within context length
    chunks = chunk_text(text)
    
    # Step 2: Extractive summarization
    word_count = get_word_count(text)
    target_extraction = max(300, min(int(0.3 * word_count), 600))  # K = max(300, min(0.3D, 600))
    
    # Extract key sentences to reach target length
    extracted = lexlm_model.extract_key_sentences(
        text,
        target_length=target_extraction
    )
    extracted_words = get_word_count(extracted)
    
    # Step 3: Abstractive summarization
    min_summary_length = int(0.6 * extracted_words)  # 0.6K
    max_summary_length = int(0.8 * extracted_words)  # 0.8K
    
    return bart_model.summarize(
        extracted,
        min_length=min_summary_length,
        max_length=max_summary_length
    )

def apply_flexible_extraction(text: str, target_percentage: float, variance: float = 0.25) -> str:
    """Apply extraction with flexible target percentage.
    
    Args:
        text: Text to extract from
        target_percentage: Target extraction percentage (e.g., 0.34 for 34%)
        variance: Allowed variance as fraction (default 0.25 means ±25% of target)
        
    Returns:
        Extracted text preserving sentence boundaries
    """
    # Calculate allowed range
    min_pct = target_percentage * (1 - variance)
    max_pct = target_percentage * (1 + variance)
    
    # Extract with flexible target, letting LexLM handle sentence boundaries
    return lexlm_model.extract_key_sentences(
        text,
        extraction_percentage=target_percentage,
        min_ratio=min_pct,
        max_ratio=max_pct
    )

def apply_dependent_extraction(text: str, target_length: int, min_ratio: float = 0.15) -> str:
    """Apply dependent extraction to reduce text length.
    
    Args:
        text: Text to reduce
        target_length: Target length in words
        min_ratio: Minimum extraction ratio (default 0.15)
        
    Returns:
        Extracted text preserving sentence boundaries
    """
    current_length = get_word_count(text)
    target_ratio = target_length / current_length
    
    # Allow 25% variance around target ratio
    return lexlm_model.extract_key_sentences(
        text,
        extraction_percentage=max(min_ratio, target_ratio),
        min_ratio=max(min_ratio, target_ratio * 0.75),
        max_ratio=target_ratio * 1.25
    )

def summarise_tier3(text: str) -> str:
    """Tier 3 summarisation for texts between 2500 and 20,000 words using hierarchical summarisation.
    
    Process:
    1. Split into chunks and apply fixed-ratio extraction
    2. Apply dependent extraction passes to reach target length
    3. Generate final summary using BART
    """
    # First split into chunks and apply fixed-ratio extraction
    chunks = split_into_chunks(text)
    percentages = config['extraction']['percentages']
    default_pct = config['extraction']['default_percentage']
    
    # First pass: Extract from each chunk with flexible percentages
    extracted_chunks = []
    for i, chunk in enumerate(chunks):
        target_pct = percentages[i] if i < len(percentages) else default_pct
        extracted = apply_flexible_extraction(chunk, target_pct)
        extracted_chunks.append(extracted)
    
    # Combine chunks
    combined = " ".join(extracted_chunks)
    
    # Apply dependent extraction passes if needed
    target = config['tier3']['final_extraction_target']
    while get_word_count(combined) > target * 1.5:  # Allow 50% variance
        combined = apply_dependent_extraction(combined, target)
    
    # Final abstractive summarization
    cfg = config['tier3']
    return bart_model.summarize(
        combined,
        min_length=cfg['final_abstractive']['min'],
        max_length=cfg['final_abstractive']['max']
    )

def summarise_tier4(text: str) -> str:
    """Tier 4 summarisation for texts between 20,000 and 68,000 words using hierarchical summarisation.
    
    Process:
    1. Split into chunks and apply fixed-ratio extraction
    2. Apply dependent extraction passes to reach target length
    3. Generate final summary using LongformerBART for longer context
    """
    # First split into chunks and apply fixed-ratio extraction
    chunks = split_into_chunks(text)
    percentages = config['extraction']['percentages']
    default_pct = config['extraction']['default_percentage']
    
    # First pass: Extract from each chunk with flexible percentages
    extracted_chunks = []
    for i, chunk in enumerate(chunks):
        target_pct = percentages[i] if i < len(percentages) else default_pct
        extracted = apply_flexible_extraction(chunk, target_pct)
        extracted_chunks.append(extracted)
    
    # Combine chunks
    combined = " ".join(extracted_chunks)
    
    # Apply dependent extraction passes if needed
    target = config['tier4']['final_extraction_target']
    while get_word_count(combined) > target * 1.5:  # Allow 50% variance
        combined = apply_dependent_extraction(combined, target)
    
    # Final abstractive summarization using LongformerBART for longer context
    cfg = config['tier4']
    return bart_model.summarize(
        combined,
        min_length=cfg['final_abstractive']['min'],
        max_length=cfg['final_abstractive']['max'],
        use_longformer=True  # Use LongformerBART for longer context
    )

def split_into_chunks(text: str, chunk_size: int = None) -> list[str]:
    """Split text into chunks of specified size.
    
    Args:
        text: Text to split
        chunk_size: Size of each chunk in words (default from config)
        
    Returns:
        List of text chunks
    """
    if chunk_size is None:
        chunk_size = config['chunking']['max_chunk_size']
        
    # Use NLTK to split into sentences first
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_words = get_word_count(sentence)
        if current_size + sentence_words <= chunk_size:
            current_chunk.append(sentence)
            current_size += sentence_words
        else:
            if current_chunk:  # Add completed chunk
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_words
            
    if current_chunk:  # Add final chunk
        chunks.append(' '.join(current_chunk))
        
    return chunks

def summarise(text: str) -> str:
    """Dispatch summarisation to the correct tier based on document length."""
    # First calculate total word count
    word_count = get_word_count(text)
    
    # Dispatch to appropriate tier based on length
    if word_count <= 600:
        return summarise_tier1(text)
    elif word_count <= 2500:
        return summarise_tier2(text)
    elif word_count <= 20000:
        return summarise_tier3(text)
    elif word_count <= 68000:
        return summarise_tier4(text)
    else:
        return "Document exceeds the supported length for tiers 1-4 summarisation."

if __name__ == '__main__':
    # Example test
    sample_text = "This is a test sentence. " * 300
    print(f"Input word count: {get_word_count(sample_text)}")
    summary = summarise(sample_text)
    print("Summary:", summary)
