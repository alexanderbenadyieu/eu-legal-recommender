# EU Legal Recommender System

A sophisticated recommender system for EU legal documents that provides personalized recommendations based on semantic similarity and categorical features.

## Features

- **Text-Based Recommendations**: Get recommendations based on text queries using state-of-the-art language models
- **Personalized Recommendations**: Create user profiles with expert descriptions, historical documents, and categorical preferences
- **Hybrid Approach**: Combine semantic similarity with categorical feature matching for better recommendations
- **Flexible Configuration**: Customize weights, features, and models to suit your needs
- **Caching**: Efficient caching of embeddings and results for improved performance

## Installation

```bash
# Install from PyPI
pip install eu-legal-recommender

# Or install from source
git clone https://github.com/your-org/eu-legal-recommender.git
cd eu-legal-recommender
pip install -e .
```

## Quick Start

```python
from src.models.pinecone_recommender import PineconeRecommender

# Initialize the recommender
recommender = PineconeRecommender(
    api_key="your_pinecone_api_key",
    index_name="eu-legal-documents"
)

# Get recommendations based on a text query
recommendations = recommender.get_recommendations(
    query_text="renewable energy regulations",
    top_k=5
)

# Print the recommendations
for doc in recommendations:
    print(f"{doc['id']}: {doc['title']} (Score: {doc['score']})")
```

## CLI Usage

The package includes a command-line interface for common tasks:

```bash
# Get recommendations
python -m recommender.cli recommend "renewable energy regulations" --top-k 5

# Create a user profile
python -m recommender.cli create-profile --name "energy-expert" --description "Expert in renewable energy regulations"

# Get personalized recommendations
python -m recommender.cli recommend-personalized "energy regulations" --profile "energy-expert" --top-k 5
```

## Documentation

For more detailed documentation, see:

- [User Guides](docs/guides/README.md): Getting started and usage guides
- [API Documentation](docs/api/README.md): Detailed API reference
- [Development Documentation](docs/development/README.md): Contributing to the project
- [Examples](examples/README.md): Example scripts and notebooks

## Project Structure

The project follows a standardized structure as outlined in [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md). Key components include:

- `src/`: Core implementation of the recommender system
  - `models/`: Recommender algorithms and components
  - `utils/`: Utility functions and helpers
  - `api/`: API implementation (if applicable)
- `scripts/`: Utility scripts for maintenance and operations
- `tests/`: Test suite for ensuring code quality
- `examples/`: Example usage scripts
- `docs/`: Documentation

## Testing

```bash
# Run all tests
python -m tests.run_tests

# Run tests with coverage
python -m tests.run_tests --cov --html
```

## License

[MIT License](LICENSE)

## Acknowledgements

This project uses the following open-source libraries:

- [sentence-transformers](https://github.com/UKPLab/sentence-transformers): For text embeddings
- [Pinecone](https://www.pinecone.io/): For vector similarity search
- [scikit-learn](https://scikit-learn.org/): For machine learning utilities
- [numpy](https://numpy.org/): For numerical operations
