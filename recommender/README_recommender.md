# EU Legal Document Recommender System

A sophisticated hybrid recommender system designed to match EU legal documents with user interests based on semantic similarity, document features, and user preferences. The system supports both standard and personalized recommendations, including document similarity search with client preference weighting.

## Project Structure

```
recommender/
├── run_recommender.py          # Main entry point script for recommendations
├── src/
│   ├── config.py               # Centralized configuration
│   ├── models/
│   │   ├── embeddings.py       # BERT-based embeddings generator
│   │   ├── features.py         # Categorical feature processing
│   │   ├── pinecone_recommender.py  # Core recommender implementation
│   │   ├── personalized_recommender.py  # Personalized recommender
│   │   ├── user_profile.py     # User profile management
│   │   └── similarity.py       # Similarity calculation functions
│   ├── utils/
│   │   ├── logging.py          # Logging utilities
│   │   ├── exceptions.py       # Custom exceptions
│   │   └── data_processing.py  # Data processing utilities
│   └── cli/
│       └── cli.py              # Comprehensive CLI for advanced operations
├── streamlit_app/              # Web-based user interface
│   ├── app.py                  # Streamlit application with user-friendly interface
│   ├── document_cache.py       # Caching for documents and recommendations
│   └── components/             # Modular UI components
│       ├── ui.py               # Core UI components
│       └── visualization.py    # Data visualization components
├── examples/                   # Example usage scripts
├── profiles/                   # User profile definitions
├── evaluation/                 # Evaluation scripts and results
├── results/                    # Optimization results and tuned parameters
└── tests/                      # Unit and integration tests
```

## Web Interface

The recommender system includes a web-based user interface built with Streamlit that provides an intuitive way to interact with the system without writing code.

### Features

1. **Multiple Recommendation Modes**:
   - Query-based recommendations
   - Document similarity recommendations
   - Profile-based recommendations

2. **User Profile Management**:
   - Loading and displaying client profiles
   - Applying profile preferences to recommendations

3. **Visualization and Analytics**:
   - Document similarity scores
   - Year and subject matter distributions
   - Document similarity network graphs

4. **Configuration Options**:
   - API key management
   - Recommendation parameters (top-k, filters)
   - Temporal boosting for recency

### Running the Web Interface

```bash
cd recommender
pip install streamlit
streamlit run streamlit_app/app.py
```

## Architecture Overview

### Core Components

1. **Text Embedding (embeddings.py)**: Uses Legal-BERT for domain-specific semantic understanding with efficient batch processing, caching, and GPU support.

2. **Feature Processing (features.py)**: Handles categorical document attributes with one-hot encoding and dynamic feature adaptation.

3. **Similarity Computation**: Combines text embedding similarity (70%) and categorical feature similarity (30%) using Pinecone for efficient vector search.

4. **Recommender System**:
   - **Base Recommender**: Provides configurable ranking with hybrid search and metadata filtering.
   - **Personalized Recommender**: Extends the base recommender with user profiles combining expert descriptions, historical documents, and categorical preferences.

## Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

The recommender system requires a Pinecone API key for vector similarity search. Create a `.env` file in the recommender directory with the following variables:

```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=gcp-starter  # or your preferred environment
DEVICE=cpu  # or cuda if you have GPU support
```

Alternatively, you can pass these values as command-line arguments when using the `run_recommender.py` script.

## Usage

### Command-Line Usage

The simplest way to use the recommender system is through the command-line interface provided by `run_recommender.py`:

```bash
# Get recommendations based on a query
python run_recommender.py --query "Environmental regulations for reducing greenhouse gas emissions" --top-k 5

# Get similar documents based on a document ID
python run_recommender.py --document-id 32018L2001 --top-k 5

# Get personalized recommendations using a user profile
python run_recommender.py --query "renewable energy policy" --profile profiles/renewable_energy_client.json

# Use custom weights for text and categorical similarity
python run_recommender.py --query "climate change" --text-weight 0.8 --categorical-weight 0.2
```

### Programmatic Usage

```python
# Basic usage
from recommender.src.models.embeddings import BERTEmbedder
from recommender.src.models.features import FeatureProcessor
from recommender.src.models.pinecone_recommender import PineconeRecommender

# Initialize components
embedder = BERTEmbedder(model_name='nlpaueb/legal-bert-base-uncased')
feature_processor = FeatureProcessor({
    'type': ['regulation', 'directive', 'decision', 'other'],
    'subject': ['environment', 'energy', 'finance', 'health']
})

# Create recommender
recommender = PineconeRecommender(
    api_key='your_pinecone_api_key',
    index_name='eu-legal-documents-legal-bert',
    embedder=embedder,
    feature_processor=feature_processor
)

# Get recommendations by query
recommendations = recommender.get_recommendations(
    query_text="Climate change policies in the EU",
    top_k=5
)

# Advanced configuration
recommender = PineconeRecommender(
    api_key='your_pinecone_api_key',
    index_name='eu-legal-documents-legal-bert',
    embedder=embedder,
    feature_processor=feature_processor,
    text_weight=0.8,  # Customize weights
    categorical_weight=0.2
)

# With metadata filtering
recommendations = recommender.get_recommendations(
    query_text="Renewable energy directives",
    metadata_filter={
        'document_type': 'directive',
        'year': {'$gte': 2020}  # Only documents from 2020 or later
    },
    top_k=10
)
```

## Performance Optimization

1. **Batch Processing**
   - Documents are processed in batches
   - Configurable batch size for memory management
   - Parallel processing where applicable

2. **Caching System**
   - Embeddings cache
   - Feature encoding cache
   - FAISS index persistence

3. **GPU Acceleration**
   - BERT model can run on GPU
   - FAISS GPU support available
   - Batch size optimization for GPU memory

## Personalized Recommendations

The system supports personalized recommendations through user profiles that combine expert descriptions, historical document interactions, and categorical preferences.

```python
# Create and configure a user profile
from recommender.src.models.user_profile import UserProfile
from recommender.src.models.personalized_recommender import PersonalizedRecommender

# Initialize personalized recommender
personalized_recommender = PersonalizedRecommender(
    api_key='your_pinecone_api_key',
    index_name='eu-legal-documents-legal-bert'
)

# Get or create a user profile
user_profile = personalized_recommender.get_user_profile("renewable_energy_client")

# Configure the profile (or load from JSON file)
user_profile.create_expert_profile(
    "Expert in renewable energy regulations, particularly interested in solar and wind energy policies."
)
user_profile.add_historical_document("32018L2001", engagement_score=0.9)  # Renewable Energy Directive
user_profile.set_categorical_preferences({
    "document_type": {"regulation": 0.8, "directive": 0.9},
    "subject_matters": {"energy": 1.0, "environment": 0.9}
})

# Get personalized recommendations
recommendations = personalized_recommender.get_personalized_recommendations(
    user_id="renewable_energy_client",
    query_text="offshore wind energy development",
    top_k=5
)
```

The system includes a sample renewable energy client profile at `profiles/renewable_energy_client.json` that demonstrates a complete profile with expert descriptions, historical documents, and categorical preferences.

### Command-Line Interface

The system provides two command-line interfaces:

1. **Basic CLI** (`run_recommender.py`): For common recommendation tasks

```bash
# Query-based recommendations
python recommender/run_recommender.py --query "climate change" --top-k 5

# Document similarity recommendations
python recommender/run_recommender.py --document-id "32018L2001" --top-k 5

# Personalized recommendations with user profile
python recommender/run_recommender.py --query "renewable energy" \
    --profile "recommender/profiles/renewable_energy_client.json" --top-k 5
```

2. **Advanced CLI** (`src/cli/cli.py`): For system management

```bash
# Index documents into Pinecone
python src/cli/cli.py index --db-type consolidated --tiers 1,2,3,4

# Manage user profiles
python src/cli/cli.py profile create --user-id "new_client" \
    --description "Client interested in environmental regulations"
```

### Results Directory

The `results/` directory contains optimization results and tuned parameters for the recommender system:

- **optimized_weights.json**: Contains the results of hyperparameter optimization experiments for component weights in the personalized recommender (expert profile, historical documents, categorical preferences). These weights are determined through evaluation on a validation set using metrics like NDCG@10.

These optimized parameters can be used as defaults for the recommender system to provide the best performance across different use cases.

### Scripts Directory

The `scripts/` directory contains utility scripts for various recommender system operations:

- **setup_environment.py**: Sets up the environment for the recommender system by creating necessary directories, checking dependencies, and configuring environment variables.

- **run_benchmarks.py**: Comprehensive benchmarking tool that measures performance (query time, embedding generation time, memory usage) and recommendation quality metrics (precision, recall, NDCG).

- **index_documents.py**: Script for indexing documents into Pinecone, converting document data to embeddings and storing them in the vector database.

- **recreate_embeddings.py**: Regenerates embeddings for all documents, useful when updating the embedding model or fixing corrupted embeddings.

- **debug_category_preferences.py**: Tool for debugging and testing categorical preference handling in user profiles.

- **direct_test_optimizer.py**: Script for testing and optimizing component weights for the recommender system.

- **patch_all_dictionaries.py**: Utility for updating metadata across all document dictionaries in the database.

These scripts provide essential utilities for maintaining, optimizing, and evaluating the recommender system.



## How It Works

The EU Legal Recommender system uses a hybrid approach combining semantic similarity and categorical feature matching to provide relevant document recommendations.

### Recommendation Modes

1. **Query-based**: Converts a text query to an embedding, searches for similar documents, and applies categorical filters.

2. **Document-based**: Finds documents similar to a reference document by comparing their embeddings and categorical features.

3. **Personalized**: Enhances recommendations by incorporating user profiles:
   - Combines expert descriptions, historical documents, and categorical preferences
   - Blends the user profile with the query for personalized results
   - Applies preference weights to boost scores for preferred document types and subjects

All modes use the Pinecone vector database for efficient similarity search and support metadata filtering for more targeted results.