import sqlite3
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import random

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))
from utils.summarisation_utils import get_word_count

def get_document_lengths(db_path: str) -> Dict[str, int]:
    """Get word count for each document in the database.
    
    Returns:
        Dict mapping document_id to word count
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all documents
    cursor.execute("SELECT document_id, content FROM documents WHERE content IS NOT NULL")
    
    # Calculate word counts
    doc_lengths = {}
    for doc_id, content in cursor:
        length = get_word_count(content)
        if length <= 500000:  # Exclude extremely long documents
            doc_lengths[doc_id] = length
    
    conn.close()
    return doc_lengths

def find_tier_examples(doc_lengths: Dict[str, int], num_per_tier: int = 3) -> Dict[str, List[str]]:
    """Find example documents for each tier.
    
    Args:
        doc_lengths: Dict mapping document_id to word count
        num_per_tier: Number of examples to find per tier
    
    Returns:
        Dict mapping tier name to list of document IDs
    """
    tier_ranges = {
        'tier1': (0, 600),
        'tier2': (601, 2500),
        'tier3': (2501, 20000),
        'tier4': (20001, 68000)
    }
    
    # Group documents by tier
    tier_docs = {tier: [] for tier in tier_ranges}
    for doc_id, length in doc_lengths.items():
        for tier, (min_len, max_len) in tier_ranges.items():
            if min_len <= length <= max_len:
                tier_docs[tier].append((doc_id, length))
                break
    
    # Select random examples for each tier
    examples = {}
    for tier, docs in tier_docs.items():
        if docs:
            # Sort by length to get a spread
            docs.sort(key=lambda x: x[1])
            step = len(docs) // num_per_tier
            if step == 0:
                selected = docs
            else:
                # Take evenly spaced samples
                selected = docs[::step][:num_per_tier]
            examples[tier] = [(doc_id, length) for doc_id, length in selected]
        else:
            examples[tier] = []
    
    return examples

def main():
    db_path = "../scraper/data/eurlex.db"
    
    print("Loading documents and calculating lengths...")
    doc_lengths = get_document_lengths(db_path)
    print(f"Found {len(doc_lengths)} documents (excluding those > 500,000 words)")
    
    # Get length statistics
    lengths = list(doc_lengths.values())
    print(f"\nDocument length statistics:")
    print(f"Min length: {min(lengths)} words")
    print(f"Max length: {max(lengths)} words")
    print(f"Average length: {sum(lengths)/len(lengths):.0f} words")
    
    # Find examples for each tier
    print("\nFinding example documents for each tier...")
    examples = find_tier_examples(doc_lengths)
    
    # Print examples
    for tier, docs in examples.items():
        print(f"\n{tier.upper()}:")
        if docs:
            for doc_id, length in docs:
                print(f"  - Document {doc_id}: {length} words")
        else:
            print("  No documents found in this range")
    
    # Save example IDs to file
    output_file = "tier_examples.txt"
    with open(output_file, "w") as f:
        for tier, docs in examples.items():
            f.write(f"{tier}:\n")
            for doc_id, length in docs:
                f.write(f"{doc_id},{length}\n")
            f.write("\n")
    
    print(f"\nExample document IDs saved to {output_file}")

if __name__ == "__main__":
    main()
