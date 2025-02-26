"""
Test the HTML parser on a single document
"""
import sqlite3
from pathlib import Path
import sys

# Add parent directory to path to import parser
sys.path.append(str(Path(__file__).parent.parent))
from summarization.src.preprocessing.html_parser import LegalDocumentParser

def test_single_document(html_file: str):
    """Process a single document and print its sections"""
    
    # Initialize parser
    parser = LegalDocumentParser(Path('.'))
    
    try:
        # Read HTML content from file
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Parse HTML content
        print(f"\nProcessing document {html_file}...")
        sections = parser.parse_html_content(html_content)
        
        print("\nExtracted sections:")
        print("==================")
        for i, section in enumerate(sections, 1):
            print(f"\nSection {i}:")
            print(f"Type: {section.section_type}")
            print(f"Title: {section.title}")
            print(f"Content:\n{section.content}")
            print(f"Word count: {len(section.content.split())}")
            print("-" * 80)
            
    finally:
        pass

if __name__ == "__main__":
    test_single_document('test_doc.html')
