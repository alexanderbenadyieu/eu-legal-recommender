# summarization/src/abstractive/bart_finetuner.py
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

class BartFineTuner:
    def __init__(self):
        """Initialize BART model for summarization."""
        try:
            self.summarizer = pipeline(
                'summarization',
                model="facebook/bart-large-cnn",
                device=-1  # Use CPU
            )
        except Exception as e:
            logger.error(f"Error loading BART model: {str(e)}")
            raise
    
    def summarize(self, text: str) -> str:
        """Generate a summary of the input text.
        
        Args:
            text: Text to summarize
            
        Returns:
            Generated summary
        """
        try:
            # Calculate target length based on input length
            input_words = len(text.split())
            target_length = max(30, min(int(input_words * 0.3), 150))
            
            # Generate summary
            result = self.summarizer(
                text,
                max_length=target_length,
                min_length=20,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
                truncation=True
            )
            
            return result[0]['summary_text']
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return ""  # Return empty string on error