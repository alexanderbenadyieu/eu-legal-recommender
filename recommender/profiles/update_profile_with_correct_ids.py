#!/usr/bin/env python3
"""
Script to update the renewable energy client profile with the correct document IDs from the database.
"""

import os
import sys
import json
import sqlite3
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_document_mappings(db_path, celex_numbers):
    """
    Get document ID to CELEX number mappings from the database.
    
    Args:
        db_path: Path to the SQLite database
        celex_numbers: List of CELEX numbers to look up
        
    Returns:
        Dictionary mapping CELEX numbers to document IDs
    """
    logger.info(f"Getting document mappings for {len(celex_numbers)} CELEX numbers")
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create a mapping dictionary
    mapping = {}
    
    # For each CELEX number, get the corresponding document ID
    for celex in celex_numbers:
        cursor.execute("SELECT document_id FROM documents WHERE celex_number = ?", (celex,))
        result = cursor.fetchone()
        
        if result:
            doc_id = result[0]
            mapping[celex] = doc_id
            logger.info(f"Found document ID {doc_id} for CELEX number {celex}")
        else:
            logger.warning(f"No document found for CELEX number {celex}")
    
    # Close the connection
    conn.close()
    
    return mapping

def update_profile(profile_path, output_path, mapping):
    """
    Update the profile with the correct document IDs.
    
    Args:
        profile_path: Path to the original profile JSON file
        output_path: Path to write the updated profile
        mapping: Dictionary mapping CELEX numbers to document IDs
    """
    logger.info(f"Updating profile at {profile_path}")
    
    # Load the profile
    with open(profile_path, 'r') as f:
        profile = json.load(f)
    
    # Get the historical documents
    historical_docs = profile.get('profile', {}).get('historical_documents', [])
    
    if not historical_docs:
        logger.error(f"No historical documents found in profile {profile_path}")
        return
    
    logger.info(f"Found {len(historical_docs)} historical documents in profile")
    
    # Create an updated list of historical documents using document IDs instead of CELEX numbers
    updated_docs = []
    for celex in historical_docs:
        if celex in mapping:
            updated_docs.append(str(mapping[celex]))  # Convert document ID to string
            logger.info(f"Mapped CELEX {celex} to document ID {mapping[celex]}")
        else:
            logger.warning(f"No mapping found for CELEX {celex}")
    
    # Update the profile
    profile['profile']['historical_documents'] = updated_docs
    
    # Write the updated profile
    with open(output_path, 'w') as f:
        json.dump(profile, f, indent=2)
    
    logger.info(f"Updated profile written to {output_path}")
    
    return profile

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Update renewable energy client profile with correct document IDs")
    parser.add_argument("--profile", type=str, default="profiles/renewable_energy_client.json",
                      help="Path to the profile JSON file")
    parser.add_argument("--output", type=str, default="profiles/renewable_energy_client_updated.json",
                      help="Path to write the updated profile")
    parser.add_argument("--db-path", type=str, default="../scraper/data/eurlex.db",
                      help="Path to the SQLite database")
    args = parser.parse_args()
    
    # Resolve the database path
    db_path = args.db_path
    if not os.path.isabs(db_path):
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', db_path))
    
    # Make sure the database exists
    if not os.path.exists(db_path):
        logger.error(f"Database not found at {db_path}")
        # Try alternative paths
        alt_path = os.path.join('/Users/alexanderbenady/DataThesis/eu-legal-recommender', 'scraper', 'data', 'eurlex.db')
        if os.path.exists(alt_path):
            logger.info(f"Using alternative database path: {alt_path}")
            db_path = alt_path
        else:
            logger.error("Cannot find database")
            sys.exit(1)
    
    # Load the profile
    with open(args.profile, 'r') as f:
        profile = json.load(f)
    
    # Get the historical documents
    historical_docs = profile.get('profile', {}).get('historical_documents', [])
    
    if not historical_docs:
        logger.error(f"No historical documents found in profile {args.profile}")
        sys.exit(1)
    
    # Get the document mappings
    mapping = get_document_mappings(db_path, historical_docs)
    
    # Update the profile
    update_profile(args.profile, args.output, mapping)
    
    logger.info("Done!")

if __name__ == "__main__":
    main()
