"""
Summariser module for tiers 1 to 4 based on the multi-tier summarisation strategy.

Tiers:
    Tier 1: Single-Step Abstractive Summarisation (0-600 words) using adaptive compression ratios.
    Tier 2: Two-Step Summarisation (600-2500 words) with extractive then abstractive summarisation.
    Tier 3: Hierarchical Summarisation (2500-20,000 words) with fixed and dependent extraction followed by chained abstractive summarisation.
    Tier 4: Hierarchical Summarisation (20,000-68,000 words) with similar process but adjusted target lengths and models (e.g., LongformerBART).

Note: This is a dummy implementation that simulates the summarisation with placeholder functions.
"""

import os
import math
import yaml
from utils.summarisation_utils import get_word_count

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config/summarisation_config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

from abstractive.bart_finetuner import bart_summarise
from extractive.lexlm_wrapper import lexlm_extract

# Fallback to dummy implementations if real ones are not available
from utils.summarisation_utils import abstractive_summarise as dummy_abstractive, extractive_summarise as dummy_extractive

# Define wrappers that choose the real function if available

def call_abstractive(text: str, min_length: int, max_length: int, tier: int = 1) -> str:
    """Call appropriate abstractive model based on tier.
    
    Args:
        text: Text to summarize
        min_length: Minimum summary length
        max_length: Maximum summary length
        tier: Document tier (1-4)
    
    Returns:
        str: Generated summary
    """
    if bart_summarise is not None:
        return bart_summarise(text, min_length, max_length, tier=tier)
    
    # Fallback to dummy implementation if models aren't available
    return dummy_abstractive(text, min_length, max_length, "BART")


def call_extractive(text: str, target_length: int) -> str:
    if lexlm_extract is not None:
        # Assuming lexlm_extract has the same signature
        return lexlm_extract(text, target_length)
    else:
        return dummy_extractive(text, target_length)


def summarise_tier1(text: str) -> str:
    """Tier 1 summarisation for texts up to 600 words."""
    count = get_word_count(text)
    tier_config = config['tier1']
    
    if count <= tier_config['thresholds']['short']:
        min_length, max_length = tier_config['summary_ranges']['short']
    elif count <= tier_config['thresholds']['medium']:
        min_length, max_length = tier_config['summary_ranges']['medium']
    else:
        min_length, max_length = tier_config['summary_ranges']['long']
    
    return call_abstractive(text, min_length, max_length, tier=1)


def summarise_tier2(text: str) -> str:
    """Tier 2 summarisation for texts between 600 and 2500 words."""
    D = get_word_count(text)
    tier_config = config['tier2']
    
    # Extract content according to configured ratio and bounds
    extraction_config = tier_config['extraction']
    K = max(extraction_config['min'], 
            min(int(extraction_config['ratio'] * D), extraction_config['max']))
    
    extraction = call_extractive(text, K)
    
    # Calculate final summary length using configured multipliers
    multiplier_config = tier_config['summary_multiplier']
    min_summary = int(multiplier_config['min'] * K)
    max_summary = int(multiplier_config['max'] * K)
    
    return call_abstractive(extraction, min_summary, max_summary, tier=2)


def summarise_tier3(text: str) -> str:
    """Tier 3 summarisation for texts between 2500 and 20,000 words using hierarchical summarisation."""
    from utils.text_chunking import chunk_text, get_chunk_size
    
    # Get appropriate chunk size and create chunks
    text_length = get_word_count(text)
    chunk_size = get_chunk_size(text_length)
    chunks = chunk_text(text, chunk_size, overlap=30)  # Use 30-word overlap for smaller chunks
    
    # Extract from each chunk using configured percentages
    extracted_chunks = []
    extraction_percentages = config['extraction_percentages']
    default_percentage = config['default_extraction_percentage']
    
    for i, chunk in enumerate(chunks):
        percentage = extraction_percentages[i] if i < len(extraction_percentages) else default_percentage
        current_word_count = get_word_count(chunk)
        target_words = max(1, int(current_word_count * percentage))
        extracted = call_extractive(chunk, target_words)
        extracted_chunks.append(extracted)
    
    L_e = " ".join(extracted_chunks)
    L_e_count = get_word_count(L_e)
    
    # Apply dependent extraction if needed
    tier_config = config['tier3']
    target_length = tier_config['final_extraction_target']
    
    if L_e_count > target_length:
        f = target_length / L_e_count
        f = max(f, 0.15)  # minimum extraction rate
        target = int(L_e_count * f)
        L2 = call_extractive(L_e, target)
    else:
        L2 = L_e
    
    # Final abstractive summarisation
    abstractive_config = tier_config['final_abstractive']
    return call_abstractive(L2, abstractive_config['min'], abstractive_config['max'], tier=3)


def summarise_tier4(text: str) -> str:
    """Tier 4 summarisation for texts between 20,000 and 68,000 words using hierarchical summarisation with LongformerBART."""
    from utils.text_chunking import chunk_text, get_chunk_size
    
    # Get appropriate chunk size and create chunks
    text_length = get_word_count(text)
    chunk_size = get_chunk_size(text_length)
    chunks = chunk_text(text, chunk_size, overlap=30)  # Use 30-word overlap for smaller chunks
    
    # Extract from each chunk using configured percentages
    extracted_chunks = []
    extraction_percentages = config['extraction_percentages']
    default_percentage = config['default_extraction_percentage']
    
    for i, chunk in enumerate(chunks):
        percentage = extraction_percentages[i] if i < len(extraction_percentages) else default_percentage
        current_word_count = get_word_count(chunk)
        target_words = max(1, int(current_word_count * percentage))
        extracted = call_extractive(chunk, target_words)
        extracted_chunks.append(extracted)
    
    L_e = " ".join(extracted_chunks)
    L_e_count = get_word_count(L_e)
    
    # Apply dependent extraction if needed
    tier_config = config['tier4']
    target_length = tier_config['final_extraction_target']
    
    if L_e_count > target_length:
        f = target_length / L_e_count
        f = max(f, 0.15)  # minimum extraction rate
        target = int(L_e_count * f)
        L2 = call_extractive(L_e, target)
    else:
        L2 = L_e
    
    # Final abstractive summarisation using LongformerBART
    abstractive_config = tier_config['final_abstractive']
    return call_abstractive(L2, abstractive_config['min'], abstractive_config['max'], tier=4)


import re
from typing import List, Set

def normalize_text(text: str) -> str:
    """Normalize text by removing extra whitespace and standardizing separators."""
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,;!?])', r'\1', text)
    return text.strip()

def is_contained_in(shorter: str, longer: str) -> bool:
    """Check if shorter text is semantically contained within longer text."""
    # Normalize both texts
    shorter = normalize_text(shorter)
    longer = normalize_text(longer)
    
    # If it's an exact match or the shorter text is fully contained
    if shorter == longer or shorter in longer:
        return True
    
    # Calculate similarity ratio for fuzzy matching
    shorter_words = set(shorter.split())
    longer_words = set(longer.split())
    
    # If most words from shorter are in longer, consider it contained
    common_words = shorter_words.intersection(longer_words)
    similarity_ratio = len(common_words) / len(shorter_words)
    
    return similarity_ratio > 0.9  # 90% threshold

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences more accurately."""
    # Common abbreviations to avoid splitting on
    abbreviations = {'Mr.', 'Mrs.', 'Dr.', 'Ph.D.', 'e.g.', 'i.e.', 'etc.', 'vs.', 'Art.'}
    
    # Split on sentence endings, preserving the delimiter
    splits = re.split(r'([.!?]+(?=\s|$))', text)
    
    sentences = []
    current_sentence = ''
    
    for i in range(0, len(splits)-1, 2):
        current_part = splits[i]
        delimiter = splits[i+1] if i+1 < len(splits) else ''
        
        # Check if this is actually a sentence boundary
        is_abbreviation = any(current_part.strip().endswith(abbr) for abbr in abbreviations)
        
        if is_abbreviation:
            current_sentence += current_part + delimiter
        else:
            sentences.append(normalize_text(current_sentence + current_part + delimiter))
            current_sentence = ''
    
    # Add any remaining text
    if current_sentence or (len(splits) % 2 == 1 and splits[-1]):
        remaining = current_sentence + (splits[-1] if len(splits) % 2 == 1 else '')
        if remaining.strip():
            sentences.append(normalize_text(remaining))
    
    return [s for s in sentences if s.strip()]

def clean_legal_text(text: str) -> str:
    """Clean legal document specific elements.
    
    Removes:
    1. URLs and ELI references
    2. Journal header information
    3. ISSN and edition information
    4. Other metadata and boilerplate text
    """
    # Remove all instances of the journal header and metadata
    patterns_to_remove = [
        # Header patterns
        r'Official Journal\s*of\s*the European Union',
        r'EN\s*[LS]\s*series',
        r'\d{4}/\d+\s*\d{1,2}\.\d{1,2}\.\d{4}',
        r'L series',
        r'\d{4}/\d+',
        r'\d{1,2}\.\d{1,2}\.\d{4}',
        
        # URLs and references
        r'ELI:\s*https?://[^\s]+',
        r'https?://[^\s]+',
        
        # ISSN and edition info
        r'ISSN\s*\d{4}-\d{4}',
        r'\(electronic edition\)',
        r'electronic edition',
        
        # Other metadata
        r'L \d+/\d+',
        r'Top',
        r'^EN$',  # Standalone EN
        r'^L series$',  # Standalone L series
    ]
    
    # Apply all cleanup patterns
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove multiple spaces and clean up
    text = re.sub(r'\s+', ' ', text)
    
    # Remove any lines that are just numbers or very short
    lines = [line.strip() for line in text.split('\n')]
    lines = [line for line in lines if line and not re.match(r'^[\d\s.]+$', line) and len(line.split()) > 1]
    
    return '\n'.join(lines).strip()


def fix_sentence_boundaries(text: str) -> str:
    """Fix common sentence boundary issues in legal texts."""
    # Fix missing space after period between sentences
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    
    # Fix cases where we have title/name followed by actual content
    text = re.sub(r'([a-z]\))([A-Z])', r'\1 \2', text)  # Fix e.g., "a)The" -> "a) The"
    text = re.sub(r'([a-z])([A-Z][a-z]+)', r'\1. \2', text)  # Fix e.g., "directiveThe" -> "directive. The"
    
    # Fix missing periods in common legal document patterns
    text = re.sub(r'\)(\s*[A-Z])', r'). \1', text)  # After closing parenthesis
    text = re.sub(r'([^.!?])\s+(The\s+Agreement|This\s+|Section\s+|Regulation\s+|Directive\s+)', r'\1. \2', text)
    
    # Don't split on Article references
    text = re.sub(r'\s+Article\s+\d+', lambda m: m.group().lower(), text)
    text = re.sub(r'article\s+(\d+)', r'Article \1', text)  # Then restore capitalization
    
    # Handle specific cases where we know there should be a sentence break
    text = re.sub(r'\s*(Notice concerning.*?\([^)]+\))\s*([A-Z])', r'\1. \2', text)
    
    return text


def preprocess_text(text: str) -> str:
    """Preprocess text before summarization.
    
    This function applies various preprocessing steps including:
    1. Splitting text into proper sentences
    2. Removing duplicate and contained sentences while preserving order
    3. Cleaning and normalizing text
    
    Args:
        text: Raw input text
        
    Returns:
        str: Preprocessed text ready for summarization
    """
    # First clean legal-specific elements
    text = clean_legal_text(text)
    
    # Then normalize the text
    text = normalize_text(text)
    
    # Fix sentence boundaries
    text = fix_sentence_boundaries(text)
    
    # Split into sentences
    sentences = split_into_sentences(text)
    
    # Remove duplicates and contained sentences while preserving order
    unique_sentences = []
    seen_content: Set[str] = set()
    
    for sentence in sentences:
        # Skip if sentence is empty after normalization
        if not sentence.strip():
            continue
            
        # Check if this sentence is new or contains more information
        is_new = True
        for seen in list(seen_content):  # Create a copy to modify during iteration
            # If current sentence is contained in a previous one, skip it
            if is_contained_in(sentence, seen):
                is_new = False
                break
            # If a previous sentence is contained in current one, remove it
            elif is_contained_in(seen, sentence):
                seen_content.remove(seen)
                unique_sentences = [s for s in unique_sentences if normalize_text(s) != seen]
        
        if is_new:
            seen_content.add(sentence)
            unique_sentences.append(sentence)
    
    # Join sentences back together
    return ' '.join(unique_sentences)


def summarise(text: str) -> str:
    """Dispatch summarisation to the correct tier based on document length."""
    # First preprocess the text
    processed_text = preprocess_text(text)
    
    # Then get word count and dispatch to appropriate tier
    count = get_word_count(processed_text)
    if count <= 600:
        return summarise_tier1(processed_text)
    elif count <= 2500:
        return summarise_tier2(processed_text)
    elif count <= 20000:
        return summarise_tier3(processed_text)
    elif count <= 68000:
        return summarise_tier4(processed_text)
    else:
        return "Document exceeds the supported length for tiers 1-4 summarisation."


if __name__ == '__main__':
    # Example test
    sample_text = "This is a test sentence. " * 300
    print(f"Input word count: {get_word_count(sample_text)}")
    summary = summarise(sample_text)
    print("Summary:", summary)
