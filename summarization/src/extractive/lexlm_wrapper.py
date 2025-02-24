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

def lexlm_extract(text: str, target_length: int) -> str:
    """Extract most relevant sentences from text using LexLM model.
    
    Args:
        text: Input text to extract from
        target_length: Target length of extraction in words
    
    Returns:
        str: Extracted text containing most relevant sentences
    """
    print(f"\nExtractive summarization debug:")
    print(f"Input text length: {len(text.split())} words")
    
    # Split text into sentences
    sentences = sent_tokenize(text)
    print(f"Number of sentences: {len(sentences)}")
    
    # Initialize extractor
    extractor = LexLMExtractor()
    
    # Score sentences
    scores = extractor.score_sentences(sentences)
    print(f"Sentence scores range: {min(scores):.4f} to {max(scores):.4f}")
    
    # Sort sentences by score
    sentence_scores = list(zip(sentences, scores))
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Select sentences until we reach target length
    selected_sentences = []
    current_length = 0
    
    for sentence, score in sentence_scores:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= target_length:
            selected_sentences.append(sentence)
            current_length += sentence_length
            print(f"Selected sentence (score={score:.4f}): {sentence[:100]}...")
        else:
            break
    
    print(f"Selected {len(selected_sentences)} sentences, total length: {current_length} words")
    
    # Sort selected sentences by their original order
    selected_sentences.sort(key=lambda x: sentences.index(x))
    
    # Join sentences
    return ' '.join(selected_sentences)
    
    return " ".join(selected_sentences)