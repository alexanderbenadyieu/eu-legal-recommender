"""
Post-process summaries using DeepSeek API to improve clarity and coherence while preserving legal context.
"""
import os
import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SummaryPostProcessor:
    def __init__(
        self,
        api_key: str,
        db_path: str,
        batch_size: int = 10,
        max_retries: int = 3,
        retry_delay: int = 5,
        max_workers: int = 5
    ):
        """
        Initialize the post-processor.
        
        Args:
            api_key: DeepSeek API key
            db_path: Path to SQLite database
            batch_size: Number of summaries to process in parallel
            max_retries: Maximum number of API call retries
            retry_delay: Delay between retries in seconds
            max_workers: Maximum number of concurrent threads
        """
        self.api_key = api_key
        self.db_path = Path(db_path)
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_workers = max_workers
        
        # DeepSeek API configuration
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def _get_documents_by_tier(self, tier: int) -> List[Tuple[str, str]]:
        """Get all documents and their summaries for a specific tier."""
        query = """
        SELECT doc_id, summary 
        FROM summaries 
        WHERE tier = ? 
        AND summary IS NOT NULL
        """
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (tier,))
            return cursor.fetchall()

    def _call_deepseek_api(self, prompt: str, summary: str) -> Optional[str]:
        """Make API call to DeepSeek with retries."""
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json={
                        "messages": [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": summary}
                        ],
                        "model": "deepseek-chat",
                        "temperature": 0.3,
                        "max_tokens": 1000
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                elif response.status_code == 429:  # Rate limit
                    wait_time = int(response.headers.get("Retry-After", self.retry_delay))
                    time.sleep(wait_time)
                else:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                logger.error(f"API call failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    
        return None

    def _process_single_summary(
        self,
        doc_id: str,
        summary: str,
        prompt: str
    ) -> Tuple[str, Optional[str]]:
        """Process a single summary and return the result."""
        try:
            improved_summary = self._call_deepseek_api(prompt, summary)
            return doc_id, improved_summary
        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {str(e)}")
            return doc_id, None

    def _update_summary_in_db(self, doc_id: str, new_summary: str) -> bool:
        """Update the summary in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE summaries SET summary = ?, updated_at = CURRENT_TIMESTAMP WHERE doc_id = ?",
                    (new_summary, doc_id)
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Database update failed for {doc_id}: {str(e)}")
            return False

    def process_tier(self, tier: int, prompt: str) -> Dict[str, str]:
        """
        Process all summaries for a specific tier.
        
        Args:
            tier: The tier number to process (1, 2, or 3)
            prompt: The system prompt for DeepSeek
            
        Returns:
            Dictionary mapping document IDs to processing status
        """
        documents = self._get_documents_by_tier(tier)
        logger.info(f"Found {len(documents)} documents for tier {tier}")
        
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for doc_id, summary in documents:
                future = executor.submit(
                    self._process_single_summary,
                    doc_id,
                    summary,
                    prompt
                )
                futures.append(future)
            
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Processing tier {tier}"
            ):
                doc_id, improved_summary = future.result()
                
                if improved_summary:
                    if self._update_summary_in_db(doc_id, improved_summary):
                        results[doc_id] = "success"
                    else:
                        results[doc_id] = "db_error"
                else:
                    results[doc_id] = "api_error"
        
        # Log statistics
        success_count = sum(1 for status in results.values() if status == "success")
        logger.info(f"Tier {tier} processing complete:")
        logger.info(f"Total processed: {len(documents)}")
        logger.info(f"Successful updates: {success_count}")
        logger.info(f"Failed updates: {len(documents) - success_count}")
        
        return results

def main():
    # Get API key from environment
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set")
    
    # Initialize post-processor
    processor = SummaryPostProcessor(
        api_key=api_key,
        db_path="path/to/your/database.db",
        batch_size=10,
        max_workers=5
    )
    
    # Your custom prompt
    prompt = """You are an expert legal document summarizer. Your task is to improve 
    the clarity and readability of legal document summaries while preserving all essential 
    legal terminology and key details. Focus on:
    1. Completing any unfinished sentences
    2. Removing redundant phrases
    3. Improving coherence and flow
    4. Maintaining precise legal language
    5. Preserving all important details
    
    Respond only with the improved summary, without any additional commentary."""
    
    # Process each tier
    for tier in [1, 2, 3]:
        logger.info(f"Starting processing for tier {tier}")
        results = processor.process_tier(tier, prompt)
        
        # Save results to file
        with open(f"tier_{tier}_results.json", "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
