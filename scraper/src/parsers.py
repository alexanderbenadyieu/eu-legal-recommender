"""
Document Parsing Module for EUR-Lex Web Scraper

This module provides comprehensive parsing capabilities for 
extracting metadata and content from EUR-Lex documents. It offers 
robust parsing mechanisms for handling various document structures 
and extracting critical information.

Key Features:
- Structured document metadata extraction
- Flexible parsing of HTML and text content
- Advanced date parsing and normalization
- Comprehensive error handling
- Support for multiple document types

Parsing Capabilities:
- Metadata extraction (dates, authors, identifiers)
- Full text and URL parsing
- EuroVoc descriptor and subject matter extraction
- Directory code and description parsing
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from loguru import logger

from exceptions import ParseError


@dataclass
class DocumentMetadata:
    """
    Structured representation of EUR-Lex document metadata.

    Provides a comprehensive and type-safe container for document 
    metadata, capturing essential information about EUR-Lex documents.

    Attributes:
        celex_number (str): Unique CELEX identifier for the document
        title (str): Official document title
        identifier (str): Document-specific identifier
        eli_uri (str): European Legislation Identifier URI
        adoption_date (Optional[datetime]): Date of document adoption
        effect_date (Optional[datetime]): Date when document takes effect
        end_validity_date (Optional[datetime]): Date when document becomes invalid
        authors (List[str]): List of document authors or institutions
        form (str): Document type or legal form
        eurovoc_descriptors (List[str]): EuroVoc thematic descriptors
        subject_matters (List[str]): Subject matter classifications
        directory_codes (List[str]): Official directory classification codes
        directory_descriptions (List[str]): Descriptions of directory codes

    Notes:
        - Supports optional date fields for flexibility
        - Provides comprehensive document classification information
    """
    celex_number: str
    title: str
    identifier: str
    eli_uri: str
    adoption_date: Optional[datetime]
    effect_date: Optional[datetime]
    end_validity_date: Optional[datetime]
    authors: List[str]
    form: str
    eurovoc_descriptors: List[str]
    subject_matters: List[str]
    directory_codes: List[str]
    directory_descriptions: List[str]


class DocumentContent:
    """
    Container for parsed document content and associated metadata.

    Manages the full text, HTML, and PDF URLs for a EUR-Lex document, 
    with optional metadata attachment.

    Attributes:
        full_text (str): Complete text content of the document
        html_url (str): URL to the HTML version of the document
        pdf_url (str): URL to the PDF version of the document
        metadata (Optional[DocumentMetadata]): Parsed document metadata

    Notes:
        - Supports lazy loading of metadata
        - Provides access to multiple document representations
    """
    
    def __init__(self, full_text: str, html_url: str, pdf_url: str):
        """
        Initialize a document content container.

        Args:
            full_text (str): Complete text content of the document
            html_url (str): URL to the HTML version of the document
            pdf_url (str): URL to the PDF version of the document

        Notes:
            - Initializes with document content and URL references
            - Metadata can be added later via the metadata attribute
        """
        self.full_text = full_text
        self.html_url = html_url
        self.pdf_url = pdf_url
        self.metadata = None


class MetadataParser:
    """
    Advanced parser for extracting structured metadata from EUR-Lex documents.

    Provides sophisticated parsing capabilities to extract and normalize 
    document metadata from BeautifulSoup parsed HTML content.

    Parsing Capabilities:
    - Date extraction and normalization
    - Comprehensive metadata field parsing
    - Robust error handling
    - Support for multiple document structures

    Notes:
        - Uses BeautifulSoup for HTML parsing
        - Handles variations in document metadata presentation
        - Provides detailed logging for parsing activities
    """
    
    def _extract_dates(self, soup: BeautifulSoup) -> Dict[str, str]:
        """
        Extract and normalize dates from document metadata.

        Parses various date fields from the BeautifulSoup parsed HTML, 
        handling different date representations and structures.

        Args:
            soup (BeautifulSoup): Parsed HTML content

        Returns:
            Dict[str, str]: Dictionary of date labels and their string representations

        Notes:
            - Supports multiple date field extractions
            - Handles potential missing or inconsistent date formats
            - Prepares dates for further parsing and normalization
        """
        dates = {
            'Date of document': '',
            'Date of effect': '',
            'Date of end of validity': ''
        }
        
        try:
            dates_dl = soup.find('dl', {'class': 'NMetadata'})
            if dates_dl:
                dts = dates_dl.find_all('dt')
                dds = dates_dl.find_all('dd')
                
                for dt, dd in zip(dts, dds):
                    label = dt.text.strip().lower()
                    value = dd.text.strip().split(';')[0]  # Take first part before semicolon
                    
                    if 'date of document' in label:
                        dates['Date of document'] = value
                    elif 'date of effect' in label:
                        dates['Date of effect'] = value
                    elif 'date of end of validity' in label:
                        dates['Date of end of validity'] = value
            
            return dates
            
        except Exception as e:
            logger.error(f"Failed to extract dates: {str(e)}")
            return dates
    
    def _extract_directory_info(self, soup: BeautifulSoup) -> Tuple[List[str], List[str]]:
        """
        Extract directory code and description.

        Parses directory code and description from the BeautifulSoup 
        parsed HTML content.

        Args:
            soup (BeautifulSoup): Parsed HTML content

        Returns:
            Tuple[List[str], List[str]]: Tuple of directory codes and descriptions

        Notes:
            - Supports extraction of multiple directory codes and descriptions
            - Handles variations in directory code and description presentation
        """
        try:
            directory_section = soup.find('dt', string=lambda s: s and 'Directory code' in s)
            if directory_section:
                dd = directory_section.find_next('dd')
                codes = []
                descriptions = []
                
                # Find all list items in the directory section
                for li in dd.find_all('li'):
                    # Extract the code (first part before the first link)
                    code = li.get_text().strip().split()[0]
                    
                    # Extract the full description by joining all link texts with '/'
                    desc_parts = [span.get_text(strip=True) for span in li.find_all('span')]
                    description = ' / '.join(desc_parts)
                    
                    codes.append(code)
                    descriptions.append(description)
                
                return codes, descriptions
            return [], []
            
        except Exception as e:
            logger.error(f"Failed to extract directory info: {str(e)}")
            return [], []
    
    def _extract_list_items(self, soup: BeautifulSoup, section_label: str) -> List[str]:
        """
        Extract items from a list section.

        Parses list items from the BeautifulSoup parsed HTML content.

        Args:
            soup (BeautifulSoup): Parsed HTML content
            section_label (str): Label of the section to extract

        Returns:
            List[str]: List of extracted items

        Notes:
            - Supports extraction of multiple list items
            - Handles variations in list item presentation
        """
        items = []
        try:
            section = soup.find('dt', string=lambda s: s and section_label in s)
            if section:
                dd = section.find_next('dd')
                items = [li.get_text(strip=True) for li in dd.find_all('li')]
            return items
            
        except Exception as e:
            logger.error(f"Failed to extract list items: {str(e)}")
            return items

    def parse_metadata(self, html_content: str) -> Dict[str, Any]:
        """
        Parse metadata from document HTML.

        Extracts and normalizes metadata from the BeautifulSoup parsed 
        HTML content.

        Args:
            html_content (str): Raw HTML content of the document

        Returns:
            Dict[str, Any]: Dictionary of parsed metadata

        Raises:
            ParseError: If metadata cannot be parsed
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract basic metadata
            celex = soup.find('p', {'class': 'DocumentTitle'}).text.strip().split()[-1]
            title = soup.find('p', {'id': 'title'}).text.strip()
            
            # Find identifier - it's in the p tag right after the title
            title_elem = soup.find('p', {'id': 'title'})
            identifier = None
            if title_elem:
                # Skip the hidden original title element
                next_elem = title_elem.find_next_sibling('p', {'id': 'originalTitle'})
                if next_elem:
                    next_elem = next_elem.find_next_sibling('p')
                else:
                    next_elem = title_elem.find_next_sibling('p')
                    
                if next_elem:
                    identifier = next_elem.text.strip()
                
            if not identifier:
                logger.warning("Could not find document identifier, using title as fallback")
                identifier = title
                
            eli_uri = soup.find('a', href=lambda h: h and h.startswith('http://data.europa.eu/eli/')).get('href')
            
            # Extract dates
            dates = self._extract_dates(soup)
            
            # Extract form, authors, and responsible body
            misc_section = soup.find('dt', string=lambda s: s and 'Form' in s)
            if misc_section:
                dd = misc_section.find_next('dd')
                form = dd.text.strip()
            else:
                form = ''
                
            authors_section = soup.find('dt', string=lambda s: s and 'Author' in s)
            if authors_section:
                dd = authors_section.find_next('dd')
                authors = [a.strip() for a in dd.text.split(',')]
            else:
                authors = []
            
            responsible_section = soup.find('dt', string=lambda s: s and 'Responsible body' in s)
            if responsible_section:
                dd = responsible_section.find_next('dd')
                responsible_body = dd.text.strip()
            else:
                responsible_body = ''
                
            # Extract EUROVOC descriptors and subject matters
            eurovoc_section = soup.find('dt', string=lambda s: s and 'EUROVOC descriptor' in s)
            eurovoc = []
            if eurovoc_section:
                dd = eurovoc_section.find_next('dd')
                for li in dd.find_all('li'):
                    label = li.get_text(strip=True)
                    eurovoc.append(label)
            
            subjects = self._extract_list_items(soup, 'Subject matter')
            
            # Extract directory info
            dir_codes, dir_descs = self._extract_directory_info(soup)
            
            return {
                'celex_number': celex,
                'title': title,
                'identifier': identifier,
                'eli_uri': eli_uri,
                'dates': dates,
                'authors': authors,
                'responsible_body': responsible_body,
                'form': form,
                'eurovoc_descriptors': eurovoc,
                'subject_matters': subjects,
                'directory_codes': dir_codes,
                'directory_descriptions': dir_descs
            }
            
        except Exception as e:
            logger.error(f"Failed to parse metadata: {str(e)}")
            raise ParseError(f"Failed to parse document metadata") from e


class DocumentParser:
    """
    Comprehensive parser for extracting full document content from EUR-Lex.

    Manages the parsing of document content, including text extraction, 
    URL resolution, and integration with metadata parsing.

    Parsing Capabilities:
    - Full text extraction from HTML
    - URL resolution and normalization
    - Integration with metadata parsing
    - Error handling and logging

    Attributes:
        base_url (str): Base URL for resolving relative document URLs

    Notes:
        - Supports parsing of various document types
        - Provides robust URL handling
        - Integrates with metadata extraction
    """
    
    def __init__(self, base_url: str):
        """
        Initialize the document content parser.

        Args:
            base_url (str): Base URL for resolving relative document URLs

        Notes:
            - Sets up the base URL for URL resolution
            - Prepares parser for document content extraction
        """
        self.base_url = base_url
    
    def parse_document(self, html_content: str, doc_id: str) -> DocumentContent:
        """
        Parse a complete document from HTML content.

        Extracts full document content, resolves URLs, and prepares 
        a comprehensive document representation.

        Args:
            html_content (str): Raw HTML content of the document
            doc_id (str): Unique document identifier

        Returns:
            DocumentContent: Parsed document with content and URLs

        Notes:
            - Handles full document parsing
            - Supports error handling and logging
            - Integrates metadata and content parsing
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find the main content div - try different possible IDs
            content_div = soup.find('div', {'id': 'document-content'})
            if not content_div:
                content_div = soup.find('div', {'id': 'TexteOnly'})
            if not content_div:
                content_div = soup.find('div', {'id': 'text'})
            if not content_div:
                raise ParseError("Could not find main content div")
                
            # Extract all text content
            full_text = []
            
            # Extract text from all paragraphs
            for p in content_div.find_all(['p', 'div', 'table'], recursive=True):
                # Skip empty paragraphs and navigation/metadata elements
                if not p.get_text(strip=True) or p.get('class', [None])[0] in ['hidden-print', 'navigation', 'metadata']:
                    continue
                full_text.append(p.get_text(strip=True))
            
            # Join all text with newlines
            full_text = '\n'.join(filter(None, full_text))
            
            # Get URLs
            html_url = f"{self.base_url}/legal-content/EN/TXT/?uri=OJ:L_{doc_id}"
            pdf_url = f"{self.base_url}/legal-content/EN/TXT/PDF/?uri=OJ:L_{doc_id}"
            
            return DocumentContent(
                full_text=full_text,
                html_url=html_url,
                pdf_url=pdf_url
            )
            
        except Exception as e:
            logger.error(f"Failed to parse document content: {str(e)}")
            raise ParseError("Failed to parse document content") from e
