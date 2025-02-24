# summarization/src/extractive/lexlm_wrapper.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            output_hidden_states=True,
            output_attentions=True
        )
    return model, tokenizer

class LexLMExtractor:
    def __init__(self, model_name="lexlms/legal-roberta-large"):
        """Initialize LexLM extractor with a legal domain model."""
        self.model, self.tokenizer = get_model_and_tokenizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
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
                outputs = self.model(**inputs, output_hidden_states=True)
                # Use the last hidden state as a representation of sentence importance
                hidden_states = outputs.hidden_states[-1]
                # Average the hidden states to get a sentence score
                score = hidden_states.mean().item()
                scores.append(score)
        
        return scores

    def extract_key_sentences(self, text: str, chunk_index: int = 0) -> str:
        """Extract most relevant sentences from text using LexLM model.
        
        Args:
            text: Input text to extract from
            chunk_index: Index of the current chunk (0-based) to determine extraction percentage
            
        Returns:
            str: Extracted text containing most relevant sentences
        """
        # Extraction percentages based on chunk index
        extraction_percentages = [0.34, 0.30, 0.245, 0.20, 0.165]
        default_percentage = 0.125
        
        # Get appropriate extraction percentage
        extraction_percentage = extraction_percentages[chunk_index] if chunk_index < len(extraction_percentages) else default_percentage
        
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        # Score sentences
        scores = self.score_sentences(sentences)
        
        # Sort sentences by score
        sentence_scores = list(zip(sentences, scores))
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top sentences based on chunk-specific percentage
        num_sentences = max(1, int(len(sentences) * extraction_percentage))
        selected_sentences = [s[0] for s in sentence_scores[:num_sentences]]
        
        # Sort selected sentences by their original order
        selected_sentences.sort(key=lambda x: sentences.index(x))
        
        return ' '.join(selected_sentences)