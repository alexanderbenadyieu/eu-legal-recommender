# Embedding Generation Utilities

This directory contains utilities for generating document embeddings and storing them in Pinecone.

## Overview

The scripts in this directory help you:

1. Generate embeddings for document summaries using sentence-transformers
2. Upload these embeddings to a Pinecone vector database
3. Organize documents by tier (based on document length)
4. Support multiple database sources through a flexible connector system

## Prerequisites

Before using these scripts, you need:

1. A Pinecone API key (sign up at https://www.pinecone.io/)
2. The processed documents database (from the summarization pipeline)
3. Python dependencies installed (see requirements.txt in the recommender directory)

## Database Connectors

The system supports multiple database sources through a flexible connector system:

1. **SQLite Connector**: Connects to a local SQLite database
2. **API Connector**: Connects to a remote API endpoint (for future use)

You can easily extend the system by adding new connectors in the `db_connector.py` file.

## Usage

### Generate Embeddings from SQLite Database (Default)

```bash
python generate_embeddings.py \
  --pinecone-api-key YOUR_PINECONE_API_KEY \
  --tier 1
```

### Generate Embeddings from API Source

```bash
python generate_embeddings.py \
  --db-type api \
  --api-url https://your-api-endpoint.com/documents \
  --db-api-key YOUR_API_KEY \
  --pinecone-api-key YOUR_PINECONE_API_KEY \
  --tier 1
```

### Additional Options

#### Database Source Options
- `--db-type`: Type of database connector to use (`sqlite` or `api`, default: `sqlite`)
- `--db-path`: Path to the SQLite database (default: `../../summarization/data/processed_documents.db`)
- `--api-url`: URL for the API (required for api db-type)
- `--db-api-key`: API key for the database API (optional for api db-type)

#### Pinecone Options
- `--pinecone-api-key`: Pinecone API key (required)
- `--index-name`: Name of the Pinecone index (default: "eu-legal-documents")

Note: The system uses the Pinecone serverless GCP starter environment by default.

#### Other Options
- `--batch-size`: Number of embeddings to upload in each batch (default: 100)
- `--model`: Sentence transformer model to use (default: "all-MiniLM-L6-v2")

### Example for Processing All Tiers

```bash
# Process tier 1 documents
python generate_embeddings.py --pinecone-api-key YOUR_API_KEY --tier 1

# Process tier 2 documents
python generate_embeddings.py --pinecone-api-key YOUR_API_KEY --tier 2

# Process tier 3 documents
python generate_embeddings.py --pinecone-api-key YOUR_API_KEY --tier 3

# Process tier 4 documents
python generate_embeddings.py --pinecone-api-key YOUR_API_KEY --tier 4
```

## Implementation Details

- The scripts use the `BERTEmbedder` class from the recommender system to generate embeddings
- For each document, we combine the summary text and keywords to create a rich embedding
- Documents are identified by their CELEX number in the Pinecone database
- Metadata (tier, word count, etc.) is stored alongside each embedding for filtering during search

## Extending the System

### Adding a New Database Connector

To add support for a new database source:

1. Open `db_connector.py`
2. Create a new class that inherits from `DocumentDBConnector`
3. Implement the required methods:
   - `fetch_tier_documents(tier)`
   - `fetch_document_keywords(document_id)`
   - `close()`
4. Update the `get_connector()` factory function to support your new connector

Example:

```python
class MyNewConnector(DocumentDBConnector):
    def __init__(self, connection_string):
        # Initialize your connector
        self.connection_string = connection_string
        # Connect to your database
        
    def fetch_tier_documents(self, tier):
        # Implement fetching documents from your source
        # Return a list of document dictionaries
        
    def fetch_document_keywords(self, document_id):
        # Implement fetching keywords for a document
        # Return a list of keywords
        
    def close(self):
        # Close any connections
```

Then update the factory function:

```python
def get_connector(connector_type, **kwargs):
    # Existing code...
    elif connector_type.lower() == 'my_new_type':
        if 'connection_string' not in kwargs:
            raise ValueError("connection_string is required for my_new_type connector")
        return MyNewConnector(kwargs['connection_string'])
    # Existing code...
```
