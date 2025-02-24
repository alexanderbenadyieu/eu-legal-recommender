import logging
import yaml
from pathlib import Path
from summarization.src.pipeline import SummarizationPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    config_path = Path(__file__).parent / 'config' / 'summarisation_config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load configuration
    config = load_config()
    
    # Initialize pipeline
    db_path = Path(__file__).parent.parent / 'scraper' / 'data' / 'eurlex.db'
    pipeline = SummarizationPipeline(str(db_path), config)
    
    # Process a single document
    docs = pipeline.process_documents(limit=1)
    
    # Print results
    for doc in docs:
        print(f"\nDocument: {doc['celex_number']} - {doc['title']}\n")
        
        for section in doc['sections']:
            print(f"\nSection: {section['title']} (Level {section['level']})\n")
            
            print("Chunks:")
            for i, chunk in enumerate(section['chunks'], 1):
                print(f"\nChunk {i}:\n{chunk}\n{'='*50}")
            
            if section['summary']:
                print(f"\nSummary:\n{section['summary']}\n{'='*50}")

if __name__ == '__main__':
    main()