"""
Parse and preprocess HTML content from legal documents
"""
import json
from pathlib import Path
from bs4 import BeautifulSoup
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentSection:
    """Represents a section of the legal document"""
    title: str
    content: str
    section_type: str

class LegalDocumentParser:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and special characters"""
        # Remove multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text)
        # Keep all characters except truly unwanted ones
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        return text.strip()
    
    def _remove_unwanted_elements(self, soup: BeautifulSoup) -> None:
        """Remove unwanted elements from the soup"""
        unwanted_classes = [
            'oj-signatory',  # Signature section
            'oj-note',      # Notes/footnotes
            'oj-hd-lg',     # Language indicator
            'oj-final',     # Final section
            'oj-hd-ti',     # Title header
            'oj-hd-coll',   # Collection header
            'oj-hd-uniq',   # Unique identifier
            'oj-hd-date',   # Date header
        ]
        
        for class_name in unwanted_classes:
            for element in soup.find_all(class_=class_name):
                element.decompose()

    def _extract_table_content(self, table) -> str:
        """Extract meaningful text content from a table"""
        content_parts = []
        
        # Process each row
        for row in table.find_all('tr'):
            row_parts = []
            # Get all cells (both header and data cells)
            cells = row.find_all(['td', 'th'])
            
            for cell in cells:
                # Skip cells that only contain numbers or are empty
                cell_text = self._clean_text(cell.get_text())
                if cell_text and not cell_text.isdigit():
                    # Check if it's a paragraph cell
                    if cell.find('p', class_='oj-normal'):
                        row_parts.append(cell_text)
                    # For definition-style tables, include both number and content
                    elif len(cells) > 1:
                        row_parts.append(cell_text)
            
            if row_parts:
                content_parts.append(' '.join(row_parts))
        
        return '\n'.join(content_parts)
    
    def parse_html_content(self, html_content: str) -> List[DocumentSection]:
        """Parse HTML content and extract structured sections"""
        soup = BeautifulSoup(html_content, 'html.parser')
        sections = []
        
        # Remove unwanted elements
        self._remove_unwanted_elements(soup)
        
        # Extract main title
        main_title = soup.find('div', class_='eli-main-title')
        if main_title:
            sections.append(DocumentSection(
                title="Document Title",
                content=self._clean_text(main_title.get_text()),
                section_type="header"
            ))
        
        # Find all section titles (including articles)
        section_titles = soup.find_all(['p', 'div'], class_=['oj-ti-grseq-1', 'oj-ti-art'])
        
        for title_elem in section_titles:
            title_text = self._clean_text(title_elem.get_text())
            content_parts = []
            
            # Get all following siblings until the next section title
            current = title_elem.find_next_sibling()
            while current and not (current.name == 'p' and 
                                 ('oj-ti-grseq-1' in current.get('class', []) or 
                                  'oj-ti-art' in current.get('class', []))):
                if current.name == 'p' and 'oj-normal' in current.get('class', []):
                    text = self._clean_text(current.get_text())
                    if text:
                        content_parts.append(text)
                elif current.name == 'table':
                    table_content = self._extract_table_content(current)
                    if table_content:
                        content_parts.append(table_content)
                current = current.find_next_sibling()
            
            if content_parts:  # Only add sections with content
                sections.append(DocumentSection(
                    title=title_text,
                    content='\n'.join(content_parts),
                    section_type="article" if 'oj-ti-art' in title_elem.get('class', []) else "section"
                ))
        
        return sections
    
    def process_file(self, file_path: str) -> Optional[Dict]:
        """Process a single JSON file and extract structured content"""
        full_path = self.base_dir / file_path
        
        try:
            with open(full_path, 'r') as f:
                data = json.load(f)
            
            if 'content_html' not in data:
                logger.warning(f"No HTML content found in {file_path}")
                return None
            
            # Parse the HTML content
            sections = self.parse_html_content(data['content_html'])
            
            # Create structured output with only celex and sections
            return {
                'celex_number': data['metadata']['celex_number'],
                'sections': [
                    {
                        'title': section.title,
                        'content': section.content,
                        'type': section.section_type
                    }
                    for section in sections
                ]
            }
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return None

def main():
    # Initialize parser
    base_dir = Path("/Users/alexanderbenady/DataThesis/eu-legal-recommender/scraper/data")
    parser = LegalDocumentParser(base_dir)
    
    # Read list of files to process
    with open("/Users/alexanderbenady/DataThesis/eu-legal-recommender/summarization/src/preprocessing/files_with_html.txt", 'r') as f:
        files = f.read().splitlines()
    
    # Create output directory
    output_dir = Path("/Users/alexanderbenady/DataThesis/eu-legal-recommender/summarization/src/preprocessing/processed")
    output_dir.mkdir(exist_ok=True)
    
    # Process all files
    total_files = len(files)
    processed = 0
    errors = 0
    
    logger.info(f"Processing {total_files} files...")
    
    for file_path in tqdm(files, desc="Processing files"):
        try:
            result = parser.process_file(file_path)
            if result:
                # Save using celex number as filename
                output_file = output_dir / f"{result['celex_number']}.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                processed += 1
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            errors += 1
    
    logger.info(f"Processing complete!")
    logger.info(f"Successfully processed: {processed}")
    logger.info(f"Errors: {errors}")
    logger.info(f"Total files: {total_files}")

if __name__ == "__main__":
    main()
