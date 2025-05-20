"""LexLM-based extractive summarization for EU legal documents.

This module provides a wrapper for the LexLM model (legal-roberta-large) to perform
extractive summarization on EU legal documents. It uses sentence embeddings to identify
the most important sentences in a document based on semantic similarity.

The extractive summarization is used as a preprocessing step for longer documents
before applying abstractive summarization, helping to reduce the input size while
preserving the most important information.
"""

from transformers import AutoTokenizer, AutoModel
import torch
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Any, Optional

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Model name
model_name = "lexlms/legal-roberta-large"

# Initialize model and tokenizer lazily
tokenizer = None
model = None

def get_model_and_tokenizer():
    """Get the model and tokenizer, initializing if needed."""
    global tokenizer, model
    if tokenizer is None:
        print("Loading LexLM RoBERTa tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model is None:
        print("Loading LexLM RoBERTa model...")
        model = AutoModel.from_pretrained(model_name)
    return model, tokenizer

class LexLMExtractor:
    def __init__(self, model_name="lexlms/legal-roberta-large", config=None):
        """Initialize LexLM extractor with a legal domain model."""
        self.model, self.tokenizer = get_model_and_tokenizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Device set to use {self.device}")
        
        # Set default config if none provided
        self.config = config or {
            'extraction': {
                'percentages': [0.34, 0.30, 0.245, 0.20, 0.165],
                'default_percentage': 0.125
            }
        }
    
    def score_sentences(self, sentences: List[str]) -> List[float]:
        """Score sentences based on their legal relevance."""
        scores = []
        
        for sentence in sentences:
            # Skip empty sentences
            if not sentence.strip():
                scores.append(0.0)
                continue
                
            # Tokenize sentence
            inputs = self.tokenizer(
                sentence,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Get model output
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use the last layer's [CLS] token embedding as sentence representation
                last_hidden = outputs.last_hidden_state
                cls_embedding = last_hidden[:, 0, :]
                # Use L2 norm of CLS embedding as importance score
                score = torch.norm(cls_embedding).item()
                scores.append(score)
        
        return scores

    def _get_extraction_percentage(self, chunk_number: int) -> float:
        """Get extraction percentage for a given chunk number.
        
        Args:
            chunk_number: Which chunk we're extracting (1-based)
            
        Returns:
            Percentage to extract
        """
        # Load from config
        percentages = self.config['extraction']['percentages']
        default = self.config['extraction']['default_percentage']
        
        # Return appropriate percentage (0-based index)
        if chunk_number <= len(percentages):
            return percentages[chunk_number - 1]
        return default
    
    def _get_tier_extraction_target(self, word_count: int, chunk_number: int = 1) -> int:
        """Get extraction target based on document tier.
        
        Args:
            word_count: Number of words in document
            chunk_number: Which chunk we're extracting (1-based)
            
        Returns:
            Target number of words for extraction
        """
        # Tier 1: 0-600 words - No extraction needed
        if word_count <= 600:
            return word_count
            
        # For other tiers, use configured percentages
        percentage = self._get_extraction_percentage(chunk_number)
        return int(word_count * percentage)
    
    def extract_key_sentences(self, text: str, chunk_number: int = 1, tier: int = None, 
                           target_length: int = None) -> str:
        """Extract key sentences from text using LexLM scoring.
        
        Args:
            text: Text to extract from
            chunk_number: Which chunk this is (affects extraction percentage)
            tier: Document tier (affects target length)
            target_length: Specific target length in words (overrides percentage-based)
            
        Returns:
            Extracted text with length close to target
        """
        # Split text into sentences
        sentences = sent_tokenize(text)
        if not sentences:
            return text
            
        word_count = len(text.split())
        
        # Score sentences
        scores = self.score_sentences(sentences)
        
        # Create list of (index, sentence, score, words) tuples
        sentence_data = [
            (i, sent, score, len(sent.split()))
            for i, (sent, score) in enumerate(zip(sentences, scores))
        ]
        
        # Sort by score (highest first)
        sentence_data.sort(key=lambda x: x[2], reverse=True)
        
        # Determine target word count
        if target_length is not None:
            target_words = target_length
        elif tier:
            target_words = self._get_tier_extraction_target(word_count, tier)
        else:
            percentage = self._get_extraction_percentage(chunk_number)
            target_words = int(word_count * percentage)
        
        # Calculate bounds
        min_words = int(target_words * 0.9)  # Allow 10% under
        max_words = int(target_words * 1.1)  # Allow 10% over
        
        # First pass: Add highest scoring sentences until close to target
        selected_indices = set()
        current_words = 0
        
        for idx, sent, score, sent_words in sentence_data:
            if current_words + sent_words > max_words:
                continue
            selected_indices.add(idx)
            current_words += sent_words
            
            # Stop if we're close enough to target
            if current_words >= min_words:
                break
        
        # Second pass: Optimize selection
        if abs(current_words - target_words) > 0.1 * target_words:
            # If under target, try to add more sentences
            if current_words < target_words:
                for idx, sent, score, sent_words in sentence_data:
                    if idx not in selected_indices:
                        if current_words + sent_words <= max_words:
                            selected_indices.add(idx)
                            current_words += sent_words
                            if current_words >= min_words:
                                break
            
            # If over target, try to remove lowest scoring sentences
            else:
                # Sort by score (lowest first)
                sentence_data.sort(key=lambda x: x[2])
                for idx, sent, score, sent_words in sentence_data:
                    if idx in selected_indices:
                        if current_words - sent_words >= min_words:
                            selected_indices.remove(idx)
                            current_words -= sent_words
                            if current_words <= max_words:
                                break
        
        # Return sentences in original order
        selected = sorted((idx, sent) for idx, sent, _, _ in sentence_data if idx in selected_indices)
        return ' '.join(sent for _, sent in selected)