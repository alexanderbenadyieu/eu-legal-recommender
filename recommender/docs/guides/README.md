# User Guides

This directory contains user guides for the EU Legal Recommender system.

## Contents

- **Getting Started**: Basic setup and configuration
- **Basic Usage**: How to use the recommender for simple queries
- **Personalized Recommendations**: Setting up and using personalized recommendations
- **User Profiles**: Creating and managing user profiles
- **Advanced Features**: Advanced usage scenarios and configurations

## Getting Started

To get started with the EU Legal Recommender system:

1. Install the package using pip:
   ```
   pip install eu-legal-recommender
   ```

2. Set up your environment variables:
   ```
   PINECONE_API_KEY=your_api_key
   PINECONE_ENVIRONMENT=your_environment
   ```

3. Initialize the recommender:
   ```python
   from src.models.pinecone_recommender import PineconeRecommender
   
   recommender = PineconeRecommender(
       api_key=os.getenv("PINECONE_API_KEY"),
       index_name="eu-legal-documents"
   )
   ```

4. Make your first recommendation query:
   ```python
   recommendations = recommender.get_recommendations(
       query_text="renewable energy regulations",
       top_k=5
   )
   ```

For more detailed guides, see the specific documentation files in this directory.
