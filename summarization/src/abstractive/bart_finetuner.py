"""BART-based abstractive summarization for EU legal documents.

This module provides a wrapper for fine-tuned BART models specialized for
summarizing EU legal documents. It includes models optimized for different
document lengths and tiers in the summarization pipeline.

The models are fine-tuned on EU legal document datasets to better capture
the specific language and structure of legal texts.
"""

from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

class BartFineTuner:
    """BART-based abstractive summarization for EU legal documents.
    
    This class provides a wrapper for fine-tuned BART models specialized for
    different tiers of the summarization pipeline. It loads and manages multiple
    models optimized for different document lengths and summarization requirements.
    
    The class handles model loading, text generation, and error handling for the
    abstractive summarization component of the pipeline.
    """
    
    def __init__(self):
        """Initialize BART models for different tiers of summarization.
        
        Loads specialized models for each tier:
        - Tier 1: Direct summarization for short documents (0-600 words)
        - Tier 2: Two-step summarization for medium documents (600-2500 words)
        - Tier 3/4: Hierarchical summarization for long documents (2500+ words)
        """
        try:
            # Tier 1 model (0-600 words)
            self.tier1_model = pipeline(
                'summarization',
                model="MikaSie/BART_no_extraction_V2",
                device=-1  # Use CPU
            )
            
            # Tier 2 model (600-2500 words)
            self.tier2_model = pipeline(
                'summarization',
                model="MikaSie/LexLM_BART_hybrid_V1",
                device=-1
            )
            
            # Tier 3/4 model for final summaries (2500+ words)
            self.tier3_model = pipeline(
                'summarization',
                model="MikaSie/LexLM_BART_hybrid_V1",  # Same as Tier 2
                device=-1
            )
            
            # Tier 4 section-level model
            self.tier4_section_model = pipeline(
                'summarization',
                model="facebook/bart-base",
                device=-1
            )
            
        except Exception as e:
            logger.error(f"Error loading BART models: {str(e)}")
            raise
    
    def _get_tier_parameters(self, input_words: int) -> tuple[int, int]:
        """Get target summary length based on input length tier.
        
        Args:
            input_words: Number of words in input text
            
        Returns:
            Tuple of (min_length, max_length) for summary
        """
        # Tier 1: 0-600 words
        if input_words <= 600:
            if input_words <= 150:
                return 15, 50
            elif input_words <= 300:
                return 50, 100
            else:  # 301-600
                return 100, 200
        
        # Tier 2: 600-2,500 words
        elif input_words <= 2500:
            # Note: Extraction should be handled at pipeline level
            # Here we handle the abstractive part: 0.6-0.8 × K
            # where K = max(300, min(0.3×D, 600))
            K = max(300, min(int(0.3 * input_words), 600))
            return int(0.6 * K), int(0.8 * K)
        
        # Tier 3: 2,500-20,000 words
        elif input_words <= 20000:
            return 480, 600
        
        # Tier 4: 20,000-68,000 words
        else:
            return 600, 800

    def summarize(self, text: str) -> str:
        """Generate a summary of the input text following the tiered approach.
        
        Args:
            text: Text to summarize
            
        Returns:
            Generated summary
        """
        try:
            if not text:
                logger.warning("Empty input text")
                return text

            # Calculate word count and get tier parameters
            input_words = len(text.split())
            min_length, max_length = self._get_tier_parameters(input_words)
            
            logger.info(f"Input length: {input_words} words, Target summary length: {min_length}-{max_length} words")
            
            # Select appropriate model based on input length
            if input_words <= 600:
                model = self.tier1_model
            elif input_words <= 2500:
                model = self.tier2_model
            else:
                model = self.tier3_model
                
            # Get tokenizer from model pipeline
            tokenizer = model.tokenizer
            
            # Generate summary
            result = model(
                text,
                max_length=max_length,
                min_length=min_length,
                truncation=True
            )
            
            if not result or len(result) == 0:
                logger.warning("No summary generated, returning original text")
                return text
                
            return result[0]['summary_text']
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return text  # Return original text on error
            
    def summarize_tier4_section(self, text: str, max_length: int = 350) -> str:
        """Generate section-level summary for Tier 4 using BART-base.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary in words
            
        Returns:
            Generated summary
        """
        try:
            if not text:
                logger.warning("Empty input text for Tier 4 section")
                return text

            # Generate summary using BART-base
            result = self.tier4_section_model(
                text,
                min_length=int(max_length * 0.8),
                max_length=max_length,
                truncation=True
            )
            
            if not result or len(result) == 0:
                logger.warning("No Tier 4 section summary generated, returning original text")
                return text
                
            return result[0]['summary_text']
            
        except Exception as e:
            logger.error(f"Error generating Tier 4 section summary: {str(e)}")
            return text  # Return original text on error