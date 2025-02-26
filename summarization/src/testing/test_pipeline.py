import pytest
import logging
import yaml
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
from transformers import PreTrainedTokenizerFast

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.pipeline import SummarizationPipeline
from src.extractive.lexlm_wrapper import LexLMExtractor
from src.abstractive.bart_finetuner import BartFineTuner

# Mock classes for testing
class MockTensor:
    def __init__(self, data):
        self.data = data
    
    def to(self, device):
        return self
    
    def __len__(self):
        return len(self.data)

class MockTokenizerOutput:
    def __init__(self, input_ids):
        self.input_ids = MockTensor(input_ids)
    
    def to(self, device):
        return self

class MockTokenizer:
    def __init__(self):
        self.padding_side = 'right'
        self.model_max_length = 512
        self.vocab_size = 50265
        self.all_special_tokens = ['<s>', '</s>', '<pad>']
        self.special_tokens_map = {'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>'}
        
    def encode(self, text, *args, **kwargs):
        # Simple mock that returns 1 token per word
        return [1] * len(text.split())
        
    def decode(self, token_ids, *args, **kwargs):
        # Simple mock that returns a space for each token
        return ' ' * len(token_ids)
        
    def __call__(self, text, *args, **kwargs):
        input_ids = self.encode(text)
        return MockTokenizerOutput([input_ids])

class MockPipeline:
    def __init__(self, *args, **kwargs):
        self.tokenizer = MockTokenizer()
        
    def __call__(self, text, *args, **kwargs):
        # Return a shorter version of the input text
        words = text.split()
        summary_words = words[:len(words)//2]  # Take first half of words
        return [{'summary_text': ' '.join(summary_words)}]

@pytest.fixture(autouse=True)
def mock_models(monkeypatch):
    # Mock get_model_and_tokenizer
    def mock_get_model_and_tokenizer():
        return MagicMock(), MockTokenizer()
    monkeypatch.setattr('src.extractive.lexlm_wrapper.get_model_and_tokenizer', mock_get_model_and_tokenizer)
    
    # Mock pipeline creation
    def mock_pipeline(*args, **kwargs):
        return MockPipeline()
    monkeypatch.setattr('transformers.pipeline', mock_pipeline)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def config():
    return {
        'chunking': {
            'max_chunk_size': 1000,
            'overlap': 100
        },
        'extraction': {
            'percentages': [0.34, 0.30, 0.245, 0.20, 0.165],
            'default_percentage': 0.125
        }
    }

@pytest.fixture
def pipeline(config):
    db_path = str(Path(__file__).parent.parent / 'data' / 'test.db')
    return SummarizationPipeline(db_path, config)

def test_pipeline_initialization(pipeline):
    assert pipeline is not None
    assert pipeline.extractor is not None
    assert pipeline.generator is not None
    assert pipeline.config == pipeline.full_config['chunking']

@patch('src.utils.text_chunking._tokenizer', new_callable=MockTokenizer)
def test_tier_2_processing(mock_tokenizer, pipeline):
    # Test text with ~1000 words (Tier 2)
    section_text = """
    SECTION 1. Environmental Protection
    The European Union has established comprehensive regulations regarding environmental protection and climate action.
    These regulations aim to reduce greenhouse gas emissions, promote renewable energy, and ensure sustainable development.
    Member states are required to implement these directives through national legislation and report their progress regularly.
    
    SECTION 2. Implementation and Monitoring
    The European Commission monitors compliance and can initiate infringement procedures against non-compliant states.
    Key objectives include achieving climate neutrality by 2050, protecting biodiversity, and promoting circular economy.
    Member states must submit annual reports detailing their progress towards these environmental objectives.
    
    SECTION 3. Future Objectives
    Looking ahead, the EU aims to strengthen its environmental policies further.
    This includes new initiatives for renewable energy, waste reduction, and sustainable agriculture.
    The success of these programs depends on cooperation between member states and EU institutions.
    """
    # Repeat the text to get ~1000 words
    test_text = (section_text.strip() + '\n\n') * 20
    
    # Process text
    from src.utils.text_chunking import chunk_text
    with patch('src.utils.text_chunking.get_tokenizer', return_value=mock_tokenizer):
        chunks = chunk_text(test_text)
        assert len(chunks) > 1, "Text should be split into multiple chunks"
        
        # Process each chunk
        extracted_chunks = []
        for i, chunk in enumerate(chunks, 1):
            chunk_extracted = pipeline.extractor.extract_key_sentences(chunk, chunk_number=i)
            extracted_chunks.append(chunk_extracted)
            # Verify each chunk's extraction
            assert chunk_extracted is not None
            assert len(chunk_extracted.split()) <= len(chunk.split())
        
        # Combine and verify final extraction
        extracted = ' '.join(extracted_chunks)
        assert extracted is not None
        assert len(extracted.split()) < len(test_text.split())
        
        # Generate and verify summary
        summary = pipeline.generator.summarize(extracted)
        assert summary is not None
        assert len(summary.split()) < len(extracted.split())

def test_extraction_percentages(pipeline):
    extractor = pipeline.extractor
    
    # Test extraction percentages for different chunk numbers
    assert extractor._get_extraction_percentage(1) == 0.34
    assert extractor._get_extraction_percentage(2) == 0.30
    assert extractor._get_extraction_percentage(10) == 0.125  # Default

def test_tier_parameters(pipeline):
    generator = pipeline.generator
    
    # Test Tier 2 parameters (600-2500 words)
    min_len, max_len = generator._get_tier_parameters(1000)
    K = max(300, min(int(0.3 * 1000), 600))
    assert min_len == int(0.6 * K)
    assert max_len == int(0.8 * K)