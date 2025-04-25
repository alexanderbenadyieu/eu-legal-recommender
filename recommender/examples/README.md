# EU Legal Recommender System Examples

This directory contains example scripts demonstrating how to use the EU Legal Recommender System.

## Directory Structure

- **basic/**: Simple examples demonstrating core functionality
  - `document_similarity_example.py`: Shows how to calculate similarity between documents

- **personalized/**: Examples of personalized recommendation features
  - `personalized_recommender_example.py`: Demonstrates personalized recommendations based on user profiles

- **profiles/**: Examples related to user profile creation and usage
  - `try_renewable_client.py`: Example of a renewable energy client profile

## Running Examples

To run any example, navigate to the recommender directory and execute:

```bash
python -m examples.<directory>.<example_name>
```

For instance:

```bash
# Run the document similarity example
python -m examples.basic.document_similarity_example

# Run the personalized recommender example
python -m examples.personalized.personalized_recommender_example

# Try the renewable energy client profile
python -m examples.profiles.try_renewable_client
```

## Creating New Examples

When creating new examples, please follow these guidelines:

1. Place your example in the appropriate subdirectory based on its functionality
2. Include detailed comments explaining what the example demonstrates
3. Make sure the example is self-contained and can run independently
4. Add proper error handling and logging
5. Update this README if you add a new example category
