# EU Legal Recommender System

A recommender system for EU legal documents that provides personalized recommendations based on semantic similarity and categorical features. The system includes a complete pipeline from data scraping to interactive recommendation delivery via a Streamlit web application.

## Thesis Context

This repository is part of my undergraduate capstone project for the **Bachelor in Data and Business Analytics (BDBA)** at [IE University](https://www.ie.edu/). The project was conducted in collaboration with **Vinces Consulting**, a public affairs firm based in Madrid.

**Thesis Title:**  
*Automating EU Legislative Monitoring Using NLP Techniques and Content-Based Recommender Systems*

The goal of the project was to design and implement a modular recommender system that uses domain-adapted NLP techniques and hybrid similarity scoring to match EU legislative texts to user-defined regulatory profiles.

**Thesis PDF:**  
[Download the full thesis here](https://drive.google.com/file/d/19ykNQ0NkmGmjD347BEzN4o9dUdvmBP41/view?usp=sharing)

**Partner Organisation:**  
[Vinces Consulting](https://www.vincesconsulting.com/)

## System Overview

The EU Legal Recommender system consists of three independent but complementary components, each with its own specialized functionality:

### 1. Scraper
- **Automated Document Collection**: Extract legislative documents from EUR-Lex with comprehensive error handling
- **Rich Metadata Extraction**: Capture document identifiers, directory codes, dates, authors, and more
- **Efficient Storage Structure**: Organize documents in a hierarchical year/month/day structure
- **Robust Error Handling**: Implement retries, rate limiting, and recovery mechanisms

### 2. Summarization
- **Multi-Tier Processing**: Adapt summarization approach based on document length and complexity
- **Hybrid Techniques**: Combine extractive and abstractive summarization methods
- **Keyword Extraction**: Identify key terms and concepts using domain-specific models
- **Section-Aware Processing**: Handle document structure for improved summary quality

### 3. Recommender
- **Text-Based Recommendations**: Find relevant documents using state-of-the-art language models (Legal-BERT)
- **Personalized Recommendations**: Tailor results using expert profiles and categorical preferences
- **Interactive Web Interface**: Explore recommendations through a user-friendly Streamlit application
- **Client Profiles**: Pre-configured industry-specific profiles for demonstration purposes

## Project Components

The EU Legal Recommender system consists of three main components, each with its own detailed README file:

1. **[Scraper](scraper/README_scraper.md)**: Extracts legislative documents from EUR-Lex and stores them in a structured database
   - Automated document scraping with comprehensive error handling
   - Rich metadata extraction (CELEX numbers, directory codes, publication dates, etc.)
   - Efficient hierarchical storage structure (year/month/day)
   - Focuses on the Official Journal L series (Legislation)

2. **[Summarization](summarization/README_summariser.md)**: Processes EU legal documents to generate concise summaries and extract keywords
   - Multi-tier summarization strategy based on document length
   - Tier 1 (0-600 words): Direct abstractive summarization
   - Tier 2 (601-2,500 words): Two-step summarization with extractive then abstractive
   - Tier 3 (2,501-20,000 words): Hierarchical summarization with section-aware processing
   - Tier 4 (20,000+ words): Advanced hierarchical summarization with weighted extraction

3. **[Recommender](recommender/README_recommender.md)**: Provides personalized document recommendations based on semantic similarity and categorical features
   - Hybrid approach combining text embedding similarity (70%) and categorical feature matching (30%)
   - Legal-BERT for domain-specific semantic understanding
   - User profile management with expert descriptions, historical documents, and categorical preferences
   - Interactive Streamlit web interface for exploring recommendations

Each component has its own detailed README with comprehensive usage instructions, architecture details, and configuration options.

## Component Independence and Documentation

An important aspect of this system is that each component operates independently with its own configuration, dependencies, and documentation. This modular design allows you to:

1. **Use Components Independently**: Each component can be used on its own without requiring the others
2. **Mix and Match**: Combine components as needed for your specific use case
3. **Separate Development**: Each component can be developed and updated independently

### Detailed Documentation

Each component has its own comprehensive README file with detailed installation instructions, usage examples, and configuration options:

- **[Scraper Documentation](scraper/README_scraper.md)**: Complete guide to the EUR-Lex document scraper
- **[Summarization Documentation](summarization/README_summariser.md)**: Instructions for the document summarization system
- **[Recommender Documentation](recommender/README_recommender.md)**: Guide to the recommendation engine and Streamlit app

### Installation and Setup

Refer to each component's README for specific installation instructions. Generally, each component follows this pattern:

```bash
# Navigate to the component directory
cd [component_name]  # scraper, summarization, or recommender

# Install dependencies
pip install -r requirements.txt

# Follow component-specific setup instructions in the README
```

## Component Highlights

### Scraper Features

The scraper component is designed to efficiently extract EU legal documents from EUR-Lex with high reliability:

- **Automated Crawling**: Systematically extracts documents from the Official Journal L series
- **Comprehensive Metadata**: Captures over 20 metadata fields including CELEX numbers, ELI URIs, and directory codes
- **Error Resilience**: Implements exponential backoff, rate limiting, and automatic recovery from failures
- **Structured Output**: Organizes documents in a hierarchical database with consistent JSON format

See the [Scraper Documentation](scraper/README_scraper.md) for detailed usage instructions.

### Summarization Capabilities

The summarization component processes EU legal documents using a sophisticated multi-tier approach:

- **Adaptive Processing**: Tailors summarization strategy based on document length (from 0 to 20,000+ words)
- **Keyword Extraction**: Uses KeyBERT to identify important terms and concepts
- **Section-Aware**: Preserves document structure for more coherent summaries
- **Configurable Output**: Adjustable summary lengths and formats for different use cases

See the [Summarization Documentation](summarization/README_summariser.md) for detailed usage instructions.

### Recommender Interface

The recommender component includes a Streamlit application that provides a user-friendly interface:

- **Multiple Recommendation Modes**: Query-based, document similarity, and profile-based recommendations
- **Document Exploration**: View metadata, summaries, and related documents
- **Client Profiles**: Industry-specific profiles with expert descriptions and categorical preferences
- **Full-Width Profile Display**: Optimized UI for viewing detailed profile information

Each profile contains an expert description of business interests, historical documents, categorical preferences, and component weights for the recommendation algorithm.

See the [Recommender Documentation](recommender/README_recommender.md) for detailed usage instructions.

## Command-Line Interfaces

Each component provides its own command-line interface for common tasks:

### Scraper CLI

```bash
# Navigate to the scraper directory
cd scraper

# Run the scraper with a specific date range
python src/main.py --start-date 2023-10-01 --end-date 2023-10-31
```

### Summarization CLI

```bash
# Navigate to the summarization directory
cd summarization

# Run the summarization pipeline
python run_summarization.py --tier 2
```

### Recommender CLI

```bash
# Navigate to the recommender directory
cd recommender

# Get recommendations
python run_recommender.py --query "renewable energy regulations" --top-k 5
```

See the individual README files in each component directory for more specific CLI options.

## Project Structure

The project follows a modular architecture with three main components, each designed to work both independently and as part of the complete pipeline:

```
eu-legal-recommender/
├── scraper/                      # EUR-Lex document scraper
│   ├── src/                      # Source code for scraper
│   │   ├── scraper.py            # Core scraping logic
│   │   ├── parsers.py            # HTML and metadata parsers
│   │   ├── storage.py            # Data storage handlers
│   │   └── document_tracker.py   # Document tracking and deduplication
│   ├── config/                   # Configuration files
│   ├── data/                     # Scraped document database
│   └── README_scraper.md         # Scraper documentation
│
├── summarization/                # Document summarization system
│   ├── src/                      # Source code for summarization
│   │   ├── abstractive/          # Abstractive summarization components
│   │   ├── extractive/           # Extractive summarization components
│   │   ├── preprocessing/        # Document preprocessing utilities
│   │   ├── keyword_extractor.py  # Keyword extraction with KeyBERT
│   │   └── pipeline.py           # Multi-tier summarization pipeline
│   ├── config/                   # Configuration files
│   ├── models/                   # Pre-trained summarization models
│   └── README_summariser.md      # Summarization documentation
│
├── recommender/                  # Recommendation engine
│   ├── src/                      # Core implementation
│   │   ├── models/               # Recommender algorithms
│   │   │   ├── embeddings.py     # BERT-based embeddings generator
│   │   │   ├── features.py       # Categorical feature processing
│   │   │   ├── pinecone_recommender.py  # Core recommender
│   │   │   ├── personalized_recommender.py  # Personalized recommender
│   │   │   └── user_profile.py   # User profile management
│   │   └── utils/                # Utility functions
│   ├── streamlit_app/            # Interactive web application
│   │   ├── app.py                # Main Streamlit application
│   │   ├── document_cache.py     # Document caching system
│   │   └── components/           # UI components
│   ├── profiles/                 # User and client profiles
│   │   ├── renewable_energy_client.json  # Sample profile
│   │   └── fake_clients/         # Industry-specific profiles
│   └── README_recommender.md     # Recommender documentation
│
└── README.md                     # Main project documentation
```

## License

[MIT License](LICENSE)

