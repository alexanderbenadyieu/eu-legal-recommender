"""Unit tests for Tier 4 document processing utilities."""

import unittest
from unittest.mock import Mock, patch

# Mock transformers pipeline
class MockPipeline:
    def __init__(self, *args, **kwargs):
        pass
        
    def __call__(self, text, **kwargs):
        return [{'summary_text': 'Test summary'}]

from summarization.src.utils.tier4_utils import (
    Section,
    ProcessedChunk,
    BartBaseSummarizer,
    split_into_subsections,
    process_section,
    refine_extraction
)

class TestTier4Utils(unittest.TestCase):
    """Test cases for Tier 4 document processing utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock BART summarizer
        self.mock_summarizer = Mock()
        self.mock_summarizer.summarize.return_value = "Summarized text"
        
        # Create mock LexLM extractor
        self.mock_extractor = Mock()
        self.mock_extractor.extract_key_sentences.return_value = "Extracted text"
        
        # Sample text sections
        self.short_text = "This is a short section." * 20  # ~100 words
        self.medium_text = "This is a medium length section." * 100  # ~500 words
        self.long_text = "This is a long section that needs splitting." * 400  # ~2000 words
        self.very_long_text = "This is a very long section that needs multiple splits." * 800  # ~4000 words
    
    def test_split_into_subsections(self):
        """Test splitting text into subsections."""
        # Test with short text (should return single section)
        subsections = split_into_subsections(self.short_text)
        self.assertEqual(len(subsections), 1)
        
        # Test with medium text (should return single section)
        subsections = split_into_subsections(self.medium_text)
        self.assertEqual(len(subsections), 1)
        
        # Test with long text (should split into 2 sections)
        subsections = split_into_subsections(self.long_text)
        self.assertGreater(len(subsections), 1)
        for subsec in subsections:
            words = len(subsec.split())
            self.assertLessEqual(words, 1500)
            self.assertGreaterEqual(words, 750)
            
        # Test with very long text (should split into multiple sections)
        subsections = split_into_subsections(self.very_long_text)
        self.assertGreater(len(subsections), 2)
        for subsec in subsections:
            words = len(subsec.split())
            self.assertLessEqual(words, 1500)
            
    def test_process_section_short(self):
        """Test processing short sections (<750 words)."""
        section = Section(
            id=1,
            title="Short Section",
            content=self.short_text,
            section_type="article",
            section_order=1,
            word_count=100
        )
        
        chunks = process_section(section, self.mock_extractor, self.mock_summarizer)
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.section_id, 1)
        self.assertGreater(chunk.weight, 1.0)  # Higher weight for short sections
        
        # Should use direct summarization without extraction
        self.mock_extractor.extract_key_sentences.assert_not_called()
        self.mock_summarizer.summarize.assert_called_once()
        
    def test_process_section_medium(self):
        """Test processing medium sections (750-1500 words)."""
        section = Section(
            id=2,
            title="Medium Section",
            content=self.long_text,
            section_type="article",
            section_order=2,
            word_count=1000
        )
        
        chunks = process_section(section, self.mock_extractor, self.mock_summarizer)
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.section_id, 2)
        self.assertEqual(chunk.weight, 1.0)  # Standard weight
        
        # Should use both extraction and summarization
        self.mock_extractor.extract_key_sentences.assert_called_once()
        self.mock_summarizer.summarize.assert_called_once()
        
    def test_process_section_long(self):
        """Test processing long sections (>1500 words)."""
        section = Section(
            id=3,
            title="Long Section",
            content=self.very_long_text,
            section_type="article",
            section_order=3,
            word_count=2000
        )
        
        chunks = process_section(section, self.mock_extractor, self.mock_summarizer)
        self.assertGreater(len(chunks), 1)
        
        # Check decreasing weights
        weights = [chunk.weight for chunk in chunks]
        self.assertEqual(len(weights), len(chunks))
        for i in range(1, len(weights)):
            self.assertLessEqual(weights[i], weights[i-1])
            
        # Should use both extraction and summarization multiple times
        self.assertGreater(self.mock_extractor.extract_key_sentences.call_count, 1)
        self.assertGreater(self.mock_summarizer.summarize.call_count, 1)
        
    def test_refine_extraction(self):
        """Test refining extraction with weighted compression."""
        chunks = [
            ProcessedChunk(text="Chunk 1", word_count=200, weight=1.2, section_id=1),
            ProcessedChunk(text="Chunk 2", word_count=300, weight=1.0, section_id=1),
            ProcessedChunk(text="Chunk 3", word_count=400, weight=0.8, section_id=2),
            ProcessedChunk(text="Chunk 4", word_count=500, weight=0.6, section_id=2)
        ]
        
        target_length = 750
        extractions = refine_extraction(chunks, target_length)
        
        # Check output format
        self.assertEqual(len(extractions), len(chunks))
        for text, pct in extractions:
            self.assertIsInstance(text, str)
            self.assertIsInstance(pct, float)
            self.assertGreaterEqual(pct, 0.15)  # Min extraction
            self.assertLessEqual(pct, 0.35)  # Max extraction
            
        # Calculate total extracted words
        total_words = sum(
            chunk.word_count * pct 
            for (_, pct), chunk in zip(extractions, chunks)
        )
        # Allow 15% margin around target
        self.assertGreater(total_words, target_length * 0.85)
        self.assertLess(total_words, target_length * 1.15)
        
    def test_bart_base_summarizer(self):
        """Test BART-base summarizer initialization and usage."""
        with patch('transformers.pipeline', MockPipeline):
            # Initialize summarizer
            summarizer = BartBaseSummarizer()
            
            # Test summarization
            summary = summarizer.summarize("Test text", max_length=350)
            self.assertEqual(summary, "Test summary")

if __name__ == '__main__':
    unittest.main()
