#!/usr/bin/env python3
"""
Script to generate test_data.json from client profiles.
"""
import json
from pathlib import Path

# Directories
PROFILES_DIR = Path(__file__).parent / '../profiles'
OUTPUT_FILE = Path(__file__).parent / 'test_data.json'


def main():
    profiles_path = PROFILES_DIR.resolve()
    output_path = OUTPUT_FILE.resolve()
    test_data = {}

    for file in profiles_path.glob('*_client.json'):
        data = json.loads(file.read_text())
        user_id = data.get('user_id')
        docs = data.get('profile', {}).get('historical_documents', [])
        if not user_id or not isinstance(docs, list):
            continue
        # ground truth
        test_data[user_id] = {
            'query': user_id,
            'relevant_docs': docs,
            'relevance_scores': {doc: 1.0 for doc in docs}
        }

    # ensure directory exists
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(test_data, indent=2))
    print(f"Generated test data at {output_path}")


if __name__ == '__main__':
    main()
