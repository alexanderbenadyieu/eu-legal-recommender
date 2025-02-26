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
    
    def _is_regulatory_annex(self, element) -> bool:
        """Check if an element is a regulatory annex section.
        
        Regulatory annexes are identified by:
        1. Having class 'oj-doc-ti'
        2. Title matching pattern 'ANNEX [IVX]+'
        """
        if 'oj-doc-ti' not in element.get('class', []):
            return False
            
        title = element.get_text().strip()
        return bool(re.match(r'^ANNEX [IVX]+$', title))

    def _is_valid_section(self, text: str) -> bool:
        """Check if a section contains meaningful content.
        
        A section is considered invalid if it:
        1. Is empty or only whitespace
        2. Contains only numbers and punctuation
        3. Has no actual words (just special characters)
        """
        if not text or not text.strip():
            return False
            
        # Check if text contains at least one word (sequence of letters)
        has_words = bool(re.search(r'[a-zA-Z]{2,}', text))
        if not has_words:
            return False
            
        # Check if text is just numbers and punctuation
        only_nums_punct = bool(re.match(r'^[\d\s.,;:!?()\[\]{}"\`~@#$%^&*+=|\\/<>-]*$', text))
        if only_nums_punct:
            return False
            
        return True

    def _clean_text(self, text: str) -> str:
        """Clean text by removing URLs and unwanted characters while preserving basic structure."""
        if not text:
            return ""
            
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove unwanted control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

        # Remove multiple whitespace while preserving newlines
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'\n\s+', '\n', text)

        cleaned = text.strip()
        return cleaned if self._is_valid_section(cleaned) else ""
    
    def _remove_unwanted_elements(self, soup: BeautifulSoup) -> None:
        """Remove unwanted elements from the soup"""
        unwanted_classes = [
            'oj-signatory',  # Signature section
            'oj-note',      # Notes/footnotes
            'oj-hd-lg',     # Language indicator
            'oj-final',     # Final section
            'oj-hd-coll',   # Collection header
            'oj-hd-uniq',   # Unique identifier
            'oj-hd-date',   # Date header
        ]
        
        for class_name in unwanted_classes:
            for element in soup.find_all(class_=class_name):
                element.decompose()

    def _extract_numbered_paragraph(self, table) -> str:
        """Extract content from a table containing a numbered paragraph."""
        # Find the number in the first cell
        number_cell = table.find('td')
        if not number_cell:
            return ""
            
        number = number_cell.get_text().strip()
        
        # Find the content in the second cell
        content_cell = table.find_all('td')[1] if len(table.find_all('td')) > 1 else None
        if not content_cell:
            return ""
            
        # Remove any footnote references
        for note in content_cell.find_all(['a', 'span'], class_='oj-note-tag'):
            note.decompose()
            
        content = content_cell.get_text().strip()
        
        if content:
            return f"{number} {content}"
        return ""

    def _extract_table_content(self, table) -> str:
        """Extract meaningful text content from a table"""
        # Check if this is a numbered paragraph table (has 2 columns, first is narrow)
        cols = table.find_all('col')
        if len(cols) == 2 and any('4%' in col.get('width', '') for col in cols):
            return self._extract_numbered_paragraph(table)
            
        # Otherwise process as regular table
        content_parts = []
        for row in table.find_all('tr'):
            row_parts = []
            cells = row.find_all(['td', 'th'])
            
            for cell in cells:
                # Remove any footnote references
                for note in cell.find_all(['a', 'span'], class_='oj-note-tag'):
                    note.decompose()
                    
                cell_text = self._clean_text(cell.get_text())
                if cell_text and not cell_text.isdigit():
                    if cell.find('p', class_='oj-normal') or cell.find('span'):
                        row_parts.append(cell_text)
                    elif len(cells) > 1:
                        row_parts.append(cell_text)
            
            if row_parts:
                content_parts.append(' '.join(row_parts))
        
        return '\n'.join(content_parts)
    
    def parse_html_content(self, html_content: str, save_path: Optional[Path] = None) -> List[DocumentSection]:
        """Parse HTML content and extract structured sections.
        
        Args:
            html_content: Raw HTML content to parse
            save_path: Optional path to save preprocessed text
            
        Returns:
            List of DocumentSection objects containing valid sections only
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted elements first
        self._remove_unwanted_elements(soup)
        
        # Process sections in order of document structure
        processed_sections = []
        
        # 1. Main title
        main_title = soup.find('div', class_='eli-main-title')
        if main_title:
            title_text = self._clean_text(main_title.get_text())
            if title_text:  # Only add if valid
                title_section = DocumentSection(
                    title="Document Title",
                    content=title_text,
                    section_type="header"
                )
                processed_sections.append(title_section)
        
        # 2. Preamble
        preamble = soup.find('div', class_='eli-preamble')
        if preamble:
            preamble_text = self._clean_text(preamble.get_text())
            if preamble_text:  # Only add if valid
                preamble_section = DocumentSection(
                    title="Preamble",
                    content=preamble_text,
                    section_type="preamble"
                )
                processed_sections.append(preamble_section)
        
        # 3. Process eli-container and its subdivisions
        eli_container = soup.find('div', class_='eli-container')
        if eli_container:
            content_parts = []
            
            # Process all eli-subdivisions
            # Process enumerated paragraphs
            for enum_div in eli_container.find_all('div', class_='oj-enumeration-spacing', recursive=True):
                # Get all inline text
                text_parts = []
                for p in enum_div.find_all('p', style='display: inline;'):
                    text = self._clean_text(p.get_text())
                    if text:
                        text_parts.append(text)
                if text_parts:
                    content_parts.append(' '.join(text_parts))

            # Process subdivisions as before
            for subdivision in eli_container.find_all('div', class_='eli-subdivision', recursive=True):
                # Skip footnotes
                if subdivision.find(class_='oj-note'):
                    continue
                    
                # Process direct oj-normal paragraphs
                for p in subdivision.find_all('p', class_='oj-normal', recursive=False):
                    content = self._clean_text(p.get_text())
                    if content:
                        content_parts.append(content)
                
                # Process tables (which might contain numbered paragraphs)
                for table in subdivision.find_all('table', recursive=False):
                    table_content = self._extract_table_content(table)
                    if table_content:
                        content_parts.append(table_content)
                        
                # Process direct span elements (sometimes used for content)
                for span in subdivision.find_all('span', recursive=False):
                    if not span.find_parent(class_='oj-note-tag'):
                        content = self._clean_text(span.get_text())
                        if content:
                            content_parts.append(content)
            
            # Also process any direct oj-normal paragraphs in the eli-container
            for p in eli_container.find_all('p', class_='oj-normal', recursive=False):
                content = self._clean_text(p.get_text())
                if content:
                    content_parts.append(content)
            
            if content_parts:  # Only create section if we found content
                content = '\n'.join(content_parts)
                if self._is_valid_section(content):
                    section = DocumentSection(
                        title="Main Content",
                        content=content,
                        section_type="main"
                    )
                    processed_sections.append(section)
        
        # 4. Main sections and articles (excluding regulatory annexes)
        processed_elements = set()  # Keep track of elements we've processed
        
        # Find all top-level sections
        for section_type in ['oj-ti-grseq-1', 'oj-ti-art', 'oj-ti-section', 'oj-ti-chapter']:
            for section_elem in soup.find_all(class_=section_type):
                # Skip if already processed or is a regulatory annex
                if section_elem in processed_elements or self._is_regulatory_annex(section_elem):
                    continue
                    
                # Get section title and ensure it's valid
                title = self._clean_text(section_elem.get_text())
                if not title:  # Skip sections with invalid titles
                    continue
                    
                # Get section content until next section
                content_parts = []
                current = section_elem.find_next_sibling()
                
                while current and not any(cls in current.get('class', []) for cls in 
                                         ['oj-ti-grseq-1', 'oj-ti-art', 'oj-ti-section', 'oj-ti-chapter']):
                    
                    # Skip if this element has already been processed or is a nested section
                    if current in processed_elements or current.find_parent(class_=['oj-ti-grseq-1', 'oj-ti-art', 'oj-ti-section', 'oj-ti-chapter']):
                        current = current.find_next_sibling()
                        continue
                        
                    # Mark this element as processed
                    processed_elements.add(current)
                    
                    # Handle tables specially
                    if current.name == 'table':
                        table_text = self._extract_table_content(current)
                        if table_text:  # Only add valid table content
                            content_parts.append(table_text)
                    else:
                        # Check if this is an AR section
                        is_ar = any('AR' in node.strip() for node in current.stripped_strings)
                        
                        # Get text from this element and its children
                        for text_node in current.stripped_strings:
                            text = self._clean_text(text_node)
                            if text and (not is_ar or not any(p.strip().startswith('AR') for p in content_parts)):
                                content_parts.append(text)
                                
                    current = current.find_next_sibling()
                
                # Create section only if we have valid content
                if content_parts:
                    # Remove duplicates while preserving order
                    seen = set()
                    deduped_parts = []
                    for part in content_parts:
                        if part not in seen:
                            seen.add(part)
                            deduped_parts.append(part)
                    
                    content = '\n'.join(filter(None, deduped_parts))  # Remove any empty strings
                    if self._is_valid_section(content):  # Validate combined content
                        section = DocumentSection(
                            title=title,
                            content=content,
                            section_type=section_type
                        )
                        processed_sections.append(section)
                        
                # Mark the section element itself as processed
                processed_elements.add(section_elem)
        
        # Save preprocessed text if path provided
        if save_path:
            preprocessed_text = {
                'sections': [
                    {
                        'title': section.title,
                        'content': section.content,
                        'type': section.section_type
                    } for section in processed_sections
                ]
            }
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(preprocessed_text, f, indent=2, ensure_ascii=False)
        
        return processed_sections
    
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
