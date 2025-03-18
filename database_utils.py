#!/usr/bin/env python3
"""
Database Utility Module

This module provides a unified interface for accessing the consolidated database.
It handles the transition from separate databases to the unified database structure.
"""

import sqlite3
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database paths
CONSOLIDATED_DB_PATH = Path('/Users/alexanderbenady/DataThesis/eu-legal-recommender/scraper/data/eurlex.db')
LEGACY_SUMMARY_DB_PATH = Path('/Users/alexanderbenady/DataThesis/eu-legal-recommender/summarization/data/processed_documents.db')

def get_db_connection(db_type='consolidated', read_only=False):
    """
    Get a connection to the appropriate database based on type.
    
    Args:
        db_type: Type of database to connect to ('consolidated', 'scraper', or 'summary')
        read_only: Whether to open the database in read-only mode
    
    Returns:
        SQLite connection object
    """
    if db_type in ['consolidated', 'scraper']:
        db_path = CONSOLIDATED_DB_PATH
    elif db_type == 'summary':
        # For backward compatibility, check if legacy database still exists
        if LEGACY_SUMMARY_DB_PATH.exists():
            logger.warning(
                "Using legacy summary database. Consider updating your code to use the consolidated database."
            )
            db_path = LEGACY_SUMMARY_DB_PATH
        else:
            logger.info("Legacy summary database not found, using consolidated database.")
            db_path = CONSOLIDATED_DB_PATH
    else:
        raise ValueError(f"Unknown database type: {db_type}")
    
    # Create URI for read-only mode if needed
    if read_only:
        uri = f"file:{db_path}?mode=ro"
        return sqlite3.connect(uri, uri=True)
    else:
        return sqlite3.connect(db_path)

def get_document_by_celex(celex_number, db_type='consolidated', include_summary=False):
    """
    Get a document by its CELEX number.
    
    Args:
        celex_number: CELEX number of the document
        db_type: Type of database to use
        include_summary: Whether to include summary information
    
    Returns:
        Document data as a dictionary
    """
    conn = get_db_connection(db_type, read_only=True)
    cursor = conn.cursor()
    
    try:
        if db_type == 'summary' and LEGACY_SUMMARY_DB_PATH.exists():
            # Using legacy summary database
            query = """
            SELECT id, celex_number, html_url, total_words, summary, 
                   summary_word_count, compression_ratio, tier
            FROM processed_documents
            WHERE celex_number = ?
            """
        else:
            # Using consolidated database
            if include_summary:
                query = """
                SELECT document_id, celex_number, html_url, title, identifier, 
                       eli_uri, pdf_url, date_of_document, date_of_effect, 
                       date_of_end_validity, total_words, summary, 
                       summary_word_count, compression_ratio, tier
                FROM documents
                WHERE celex_number = ?
                """
            else:
                query = """
                SELECT document_id, celex_number, html_url, title, identifier, 
                       eli_uri, pdf_url, date_of_document, date_of_effect, 
                       date_of_end_validity
                FROM documents
                WHERE celex_number = ?
                """
        
        cursor.execute(query, (celex_number,))
        result = cursor.fetchone()
        
        if result:
            if db_type == 'summary' and LEGACY_SUMMARY_DB_PATH.exists():
                return {
                    'id': result[0],
                    'celex_number': result[1],
                    'html_url': result[2],
                    'total_words': result[3],
                    'summary': result[4],
                    'summary_word_count': result[5],
                    'compression_ratio': result[6],
                    'tier': result[7]
                }
            else:
                if include_summary:
                    return {
                        'document_id': result[0],
                        'celex_number': result[1],
                        'html_url': result[2],
                        'title': result[3],
                        'identifier': result[4],
                        'eli_uri': result[5],
                        'pdf_url': result[6],
                        'date_of_document': result[7],
                        'date_of_effect': result[8],
                        'date_of_end_validity': result[9],
                        'total_words': result[10],
                        'summary': result[11],
                        'summary_word_count': result[12],
                        'compression_ratio': result[13],
                        'tier': result[14]
                    }
                else:
                    return {
                        'document_id': result[0],
                        'celex_number': result[1],
                        'html_url': result[2],
                        'title': result[3],
                        'identifier': result[4],
                        'eli_uri': result[5],
                        'pdf_url': result[6],
                        'date_of_document': result[7],
                        'date_of_effect': result[8],
                        'date_of_end_validity': result[9]
                    }
        return None
    finally:
        conn.close()

def get_document_sections(document_id=None, celex_number=None, db_type='consolidated'):
    """
    Get sections for a document by document_id or celex_number.
    
    Args:
        document_id: ID of the document
        celex_number: CELEX number of the document
        db_type: Type of database to use
    
    Returns:
        List of document sections
    """
    if document_id is None and celex_number is None:
        raise ValueError("Either document_id or celex_number must be provided")
    
    conn = get_db_connection(db_type, read_only=True)
    cursor = conn.cursor()
    
    try:
        if db_type == 'summary' and LEGACY_SUMMARY_DB_PATH.exists():
            # Using legacy summary database
            if celex_number:
                query = """
                SELECT ds.id, ds.document_id, ds.title, ds.content, ds.section_type, 
                       ds.section_order, ds.word_count, ds.summary, ds.summary_word_count, 
                       ds.compression_ratio, ds.tier
                FROM document_sections ds
                JOIN processed_documents pd ON ds.document_id = pd.id
                WHERE pd.celex_number = ?
                ORDER BY ds.section_order
                """
                cursor.execute(query, (celex_number,))
            else:
                query = """
                SELECT id, document_id, title, content, section_type, section_order, 
                       word_count, summary, summary_word_count, compression_ratio, tier
                FROM document_sections
                WHERE document_id = ?
                ORDER BY section_order
                """
                cursor.execute(query, (document_id,))
        else:
            # Using consolidated database
            if celex_number:
                query = """
                SELECT ds.id, ds.document_id, ds.title, ds.content, ds.section_type, 
                       ds.section_order, ds.word_count, ds.summary, ds.summary_word_count, 
                       ds.compression_ratio, ds.tier
                FROM document_sections ds
                JOIN documents d ON ds.document_id = d.document_id
                WHERE d.celex_number = ?
                ORDER BY ds.section_order
                """
                cursor.execute(query, (celex_number,))
            else:
                query = """
                SELECT id, document_id, title, content, section_type, section_order, 
                       word_count, summary, summary_word_count, compression_ratio, tier
                FROM document_sections
                WHERE document_id = ?
                ORDER BY section_order
                """
                cursor.execute(query, (document_id,))
        
        results = cursor.fetchall()
        
        sections = []
        for result in results:
            sections.append({
                'id': result[0],
                'document_id': result[1],
                'title': result[2],
                'content': result[3],
                'section_type': result[4],
                'section_order': result[5],
                'word_count': result[6],
                'summary': result[7],
                'summary_word_count': result[8],
                'compression_ratio': result[9],
                'tier': result[10]
            })
        
        return sections
    finally:
        conn.close()

def get_document_keywords(document_id=None, celex_number=None, db_type='consolidated'):
    """
    Get keywords for a document by document_id or celex_number.
    
    Args:
        document_id: ID of the document
        celex_number: CELEX number of the document
        db_type: Type of database to use
    
    Returns:
        List of document keywords with scores
    """
    if document_id is None and celex_number is None:
        raise ValueError("Either document_id or celex_number must be provided")
    
    conn = get_db_connection(db_type, read_only=True)
    cursor = conn.cursor()
    
    try:
        if db_type == 'summary' and LEGACY_SUMMARY_DB_PATH.exists():
            # Using legacy summary database
            if celex_number:
                query = """
                SELECT dk.id, dk.document_id, dk.keyword, dk.score
                FROM document_keywords dk
                JOIN processed_documents pd ON dk.document_id = pd.id
                WHERE pd.celex_number = ?
                ORDER BY dk.score DESC
                """
                cursor.execute(query, (celex_number,))
            else:
                query = """
                SELECT id, document_id, keyword, score
                FROM document_keywords
                WHERE document_id = ?
                ORDER BY score DESC
                """
                cursor.execute(query, (document_id,))
        else:
            # Using consolidated database
            if celex_number:
                query = """
                SELECT dk.id, dk.document_id, dk.keyword, dk.score
                FROM document_keywords dk
                JOIN documents d ON dk.document_id = d.document_id
                WHERE d.celex_number = ?
                ORDER BY dk.score DESC
                """
                cursor.execute(query, (celex_number,))
            else:
                query = """
                SELECT id, document_id, keyword, score
                FROM document_keywords
                WHERE document_id = ?
                ORDER BY score DESC
                """
                cursor.execute(query, (document_id,))
        
        results = cursor.fetchall()
        
        keywords = []
        for result in results:
            keywords.append({
                'id': result[0],
                'document_id': result[1],
                'keyword': result[2],
                'score': result[3]
            })
        
        return keywords
    finally:
        conn.close()

def save_document_summary(celex_number, summary, summary_word_count, total_words, 
                         compression_ratio, tier, db_type='consolidated'):
    """
    Save or update a document summary.
    
    Args:
        celex_number: CELEX number of the document
        summary: Summary text
        summary_word_count: Word count of the summary
        total_words: Total word count of the original document
        compression_ratio: Compression ratio
        tier: Summarization tier used
        db_type: Type of database to use
    
    Returns:
        True if successful, False otherwise
    """
    conn = get_db_connection(db_type)
    cursor = conn.cursor()
    
    try:
        if db_type == 'summary' and LEGACY_SUMMARY_DB_PATH.exists():
            # Using legacy summary database
            query = """
            UPDATE processed_documents
            SET summary = ?, summary_word_count = ?, total_words = ?, 
                compression_ratio = ?, tier = ?
            WHERE celex_number = ?
            """
            cursor.execute(query, (summary, summary_word_count, total_words, 
                                 compression_ratio, tier, celex_number))
            
            if cursor.rowcount == 0:
                # Document doesn't exist, insert it
                query = """
                INSERT INTO processed_documents (celex_number, summary, summary_word_count, 
                                              total_words, compression_ratio, tier)
                VALUES (?, ?, ?, ?, ?, ?)
                """
                cursor.execute(query, (celex_number, summary, summary_word_count, 
                                     total_words, compression_ratio, tier))
        else:
            # Using consolidated database
            query = """
            UPDATE documents
            SET summary = ?, summary_word_count = ?, total_words = ?, 
                compression_ratio = ?, tier = ?
            WHERE celex_number = ?
            """
            cursor.execute(query, (summary, summary_word_count, total_words, 
                                 compression_ratio, tier, celex_number))
        
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error saving document summary: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def save_document_section(document_id, title, content, section_type, section_order, 
                         word_count, summary=None, summary_word_count=None, 
                         compression_ratio=None, tier=None, db_type='consolidated'):
    """
    Save or update a document section.
    
    Args:
        document_id: ID of the document
        title: Section title
        content: Section content
        section_type: Type of section
        section_order: Order of the section in the document
        word_count: Word count of the section
        summary: Summary of the section
        summary_word_count: Word count of the summary
        compression_ratio: Compression ratio
        tier: Summarization tier used
        db_type: Type of database to use
    
    Returns:
        ID of the inserted or updated section
    """
    conn = get_db_connection(db_type)
    cursor = conn.cursor()
    
    try:
        # Check if section already exists
        query = """
        SELECT id FROM document_sections
        WHERE document_id = ? AND section_order = ?
        """
        cursor.execute(query, (document_id, section_order))
        result = cursor.fetchone()
        
        if result:
            # Update existing section
            section_id = result[0]
            query = """
            UPDATE document_sections
            SET title = ?, content = ?, section_type = ?, word_count = ?,
                summary = ?, summary_word_count = ?, compression_ratio = ?, tier = ?
            WHERE id = ?
            """
            cursor.execute(query, (title, content, section_type, word_count,
                                 summary, summary_word_count, compression_ratio, tier,
                                 section_id))
        else:
            # Insert new section
            query = """
            INSERT INTO document_sections (document_id, title, content, section_type,
                                        section_order, word_count, summary,
                                        summary_word_count, compression_ratio, tier)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(query, (document_id, title, content, section_type,
                                 section_order, word_count, summary,
                                 summary_word_count, compression_ratio, tier))
            section_id = cursor.lastrowid
        
        conn.commit()
        return section_id
    except Exception as e:
        logger.error(f"Error saving document section: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()

def save_document_keyword(document_id, keyword, score, db_type='consolidated'):
    """
    Save or update a document keyword.
    
    Args:
        document_id: ID of the document
        keyword: Keyword text
        score: Keyword score
        db_type: Type of database to use
    
    Returns:
        ID of the inserted or updated keyword
    """
    conn = get_db_connection(db_type)
    cursor = conn.cursor()
    
    try:
        # Check if keyword already exists
        query = """
        SELECT id FROM document_keywords
        WHERE document_id = ? AND keyword = ?
        """
        cursor.execute(query, (document_id, keyword))
        result = cursor.fetchone()
        
        if result:
            # Update existing keyword
            keyword_id = result[0]
            query = """
            UPDATE document_keywords
            SET score = ?
            WHERE id = ?
            """
            cursor.execute(query, (score, keyword_id))
        else:
            # Insert new keyword
            query = """
            INSERT INTO document_keywords (document_id, keyword, score)
            VALUES (?, ?, ?)
            """
            cursor.execute(query, (document_id, keyword, score))
            keyword_id = cursor.lastrowid
        
        conn.commit()
        return keyword_id
    except Exception as e:
        logger.error(f"Error saving document keyword: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()

def get_document_id_by_celex(celex_number, db_type='consolidated'):
    """
    Get document ID by CELEX number.
    
    Args:
        celex_number: CELEX number of the document
        db_type: Type of database to use
    
    Returns:
        Document ID or None if not found
    """
    conn = get_db_connection(db_type, read_only=True)
    cursor = conn.cursor()
    
    try:
        if db_type == 'summary' and LEGACY_SUMMARY_DB_PATH.exists():
            # Using legacy summary database
            query = "SELECT id FROM processed_documents WHERE celex_number = ?"
            cursor.execute(query, (celex_number,))
        else:
            # Using consolidated database
            query = "SELECT document_id FROM documents WHERE celex_number = ?"
            cursor.execute(query, (celex_number,))
        
        result = cursor.fetchone()
        return result[0] if result else None
    finally:
        conn.close()
