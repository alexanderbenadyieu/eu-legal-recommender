"""Test cases for Tier 3 document summarization."""

import unittest
import sqlite3
from pathlib import Path
import tempfile
import os

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.pipeline import SummarizationPipeline
from src.utils.database_utils import Document
from src.utils.tier3_utils import Section, ProcessedChunk, process_section, refine_extraction

class TestTier3Processing(unittest.TestCase):
    def setUp(self):
        """Set up test environment with a temporary database."""
        # Create temporary database
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        
        # Create test database
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Create necessary tables
        self.cursor.executescript("""
            CREATE TABLE processed_documents (
                id INTEGER PRIMARY KEY,
                celex_number TEXT UNIQUE,
                title TEXT,
                content TEXT,
                total_words INTEGER,
                summary TEXT,
                summary_word_count INTEGER,
                compression_ratio REAL
            );
            
            CREATE TABLE document_sections (
                id INTEGER PRIMARY KEY,
                document_id INTEGER,
                title TEXT,
                content TEXT,
                section_type TEXT,
                section_order INTEGER,
                word_count INTEGER,
                summary TEXT,
                summary_word_count INTEGER,
                compression_ratio REAL,
                tier INTEGER,
                FOREIGN KEY (document_id) REFERENCES processed_documents(id)
            );
        """)
        
        # Initialize pipeline
        self.config = {
            'chunking': {
                'max_chunk_size': 500
            },
            'extraction': {
                'percentages': [0.34, 0.30, 0.245, 0.20, 0.165],
                'default_percentage': 0.125
            }
        }
        self.pipeline = SummarizationPipeline(self.db_path, self.config)
        
    def tearDown(self):
        """Clean up temporary files."""
        self.conn.close()
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_section_processing(self):
        """Test processing of individual sections."""
        # Test short section (≤350 words)
        short_section = Section(
            id=1,
            title="Short Section",
            content="This is a short test section. It contains few words.",
            section_type="ARTICLE",
            section_order=1,
            word_count=10
        )
        chunks = process_section(short_section)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, short_section.content)
        
        # Test medium section (350-750 words)
        medium_text = " ".join(["word"] * 500)
        medium_section = Section(
            id=2,
            title="Medium Section",
            content=medium_text,
            section_type="ARTICLE",
            section_order=2,
            word_count=500
        )
        chunks = process_section(medium_section)
        self.assertTrue(all(len(chunk.text.split()) <= 350 for chunk in chunks))
        
        # Test long section (>1500 words)
        long_text = " ".join(["word"] * 2000)
        long_section = Section(
            id=3,
            title="Long Section",
            content=long_text,
            section_type="ARTICLE",
            section_order=3,
            word_count=2000
        )
        chunks = process_section(long_section)
        self.assertTrue(all(len(chunk.text.split()) <= 750 for chunk in chunks))
        
    def test_extraction_refinement(self):
        """Test the weighted extraction refinement process."""
        # Create test chunks
        chunks = [
            ProcessedChunk("First chunk " * 50, 100, 1.2, 1),
            ProcessedChunk("Second chunk " * 100, 200, 1.0, 1),
            ProcessedChunk("Third chunk " * 150, 300, 0.8, 1),
            ProcessedChunk("Fourth chunk " * 200, 400, 0.6, 1)
        ]
        
        # Test refinement to target length
        target_length = 500
        extractions = refine_extraction(chunks, target_length)
        
        # Check that total extracted length is close to target
        total_words = sum(
            chunk.word_count * pct
            for (text, pct), chunk in zip(extractions, chunks)
        )
        self.assertTrue(
            abs(total_words - target_length) <= target_length * 0.15,
            f"Extraction total {total_words} too far from target {target_length}"
        )
        
    def test_full_tier3_processing(self):
        """Test the complete Tier 3 processing pipeline."""
        # Create test document
        doc = Document(
            id=1,
            celex_number="TEST123",
            html_url="http://test.com",
            total_words=5000,
            summary=None,
            summary_word_count=None,
            compression_ratio=None
        )
        
        # Create test sections
        sections = [
            (1, "Introduction", "First section " * 200, "INTRO", 1, 400),
            (1, "Article 1", "Second section " * 300, "ARTICLE", 2, 600),
            (1, "Article 2", "Third section " * 400, "ARTICLE", 3, 800),
            (1, "Conclusion", "Final section " * 1600, "CONCLUSION", 4, 3200)
        ]
        
        # Insert test data
        self.cursor.execute(
            "INSERT INTO processed_documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (doc.id, doc.celex_number, doc.title, doc.content, doc.total_words, 
             doc.summary, doc.summary_word_count, doc.compression_ratio)
        )
        
        self.cursor.executemany(
            """INSERT INTO document_sections 
               (document_id, title, content, section_type, section_order, word_count)
               VALUES (?, ?, ?, ?, ?, ?)""",
            sections
        )
        self.conn.commit()
        
        # Process document
        summary = self.pipeline._process_tier_3(doc, self.cursor)
        
        # Verify summary
        self.assertIsNotNone(summary)
        summary_words = len(summary.split())
        self.assertTrue(
            480 <= summary_words <= 600,
            f"Summary length {summary_words} outside target range 480-600"
        )
        
if __name__ == '__main__':
    unittest.main()
