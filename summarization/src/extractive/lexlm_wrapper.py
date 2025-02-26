# summarization/src/extractive/lexlm_wrapper.py
from transformers import AutoTokenizer, AutoModel
import torch
import nltk
from nltk.tokenize import sent_tokenize
from typing import List

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
    
    def extract_key_sentences(self, text: str, chunk_number: int = 1) -> str:
        """Extract most relevant sentences from text using LexLM model.
        
        Args:
            text: Input text to extract from
            chunk_number: Which chunk we're extracting (1-based)
            
        Returns:
            str: Extracted text containing most relevant sentences
        """
        # Split text into sentences
        sentences = sent_tokenize(text)
        word_count = len(text.split())
        
        # Score sentences
        scores = self.score_sentences(sentences)
        
        # Sort sentences by score
        sentence_scores = list(zip(sentences, scores))
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get target word count (this is just a guideline)
        target_words = self._get_tier_extraction_target(word_count, chunk_number)
        
        # Calculate flexible bounds
        # Allow more variation for smaller chunks (earlier chunks)
        variation = 0.25 if chunk_number <= 2 else 0.2
        min_words = int(target_words * (1 - variation))
        max_words = int(target_words * (1 + variation))
        
        # Select sentences until we reach a good stopping point
        selected_sentences = []
        current_words = 0
        
        for sentence, score in sentence_scores:
            sentence_words = len(sentence.split())
            
            # Always add the first sentence regardless of length
            if not selected_sentences:
                selected_sentences.append(sentence)
                current_words += sentence_words
                continue
            
            # If we're in the target range, only add high-scoring sentences
            if current_words >= min_words:
                # If the score is low or we're over max, stop here
                if score < scores[0] * 0.5 or current_words >= max_words:
                    break
            
            # Add the sentence if it doesn't make us go too far over
            if current_words + sentence_words <= max_words * 1.1:
                selected_sentences.append(sentence)
                current_words += sentence_words
        
        # Sort selected sentences by their original order
        selected_sentences.sort(key=lambda x: sentences.index(x))
        
        return ' '.join(selected_sentences)