from openai import OpenAI
import logging
import time
from typing import Optional, Tuple
from openai.types.chat import ChatCompletion
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class DeepSeekPostProcessor:
    def __init__(self, api_key: str):
        """Initialize DeepSeek client with API key."""
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        
    def _construct_prompt(self, title: str, summary: str) -> str:
        """Construct the prompt for DeepSeek API."""
        return f"""Title: {title}

Original Summary:
{summary}"""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _call_api(self, prompt: str) -> Tuple[Optional[ChatCompletion], Optional[str]]:
        """Make API call with retry logic."""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": """Please improve the following legal document summary. Your revised summary must:
1. Preserve all key legal terms and vocabulary exactly as in the original.
2. Improve overall readability, clarity, and redaction. Add sentences and explanations as needed.
3. Ensure that there are no unfinished or fragmented sentences.
4. Eliminate any redundancies or repetitive phrases.
5. Maintain a final length that is roughly similar to the original summary, though it may be slightly more detailed or concise as needed.
6. If the summary provides incoherent or too little information, try to construct something rational from the information you obtain from the title and summary. 
7. Eliminate unnecessary sections (such as the ones titled 'FROM WHEN DOES THIS REGULATIONS APPLY?', 'MAIN DOCUMENT', 'RELATED DOCUMENTS'…). For example, the date since which a regulation has been in effect is unnecessary - we need to focus on the scope of the legal document and what it entails.
8. Return only the revised summary with no additional commentary. It must be formatted as pure text, without bulletpoints, divided in paragraphs."""},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                timeout=30  # 30 second timeout
            )
            return response, None
        except Exception as e:
            return None, str(e)

    def refine_summary(self, title: str, summary: str, doc_id: str) -> Optional[str]:
        """
        Refine a legal document summary using DeepSeek API.
        
        Args:
            title: Title of the legal document
            summary: Original summary to refine
            doc_id: Document identifier for logging
            
        Returns:
            Refined summary or None if processing fails
        """
        try:
            prompt = self._construct_prompt(title, summary)
            logger.info(f"Processing document {doc_id} with title: {title[:100]}...")
            
            start_time = time.time()
            response, error = self._call_api(prompt)
            elapsed_time = time.time() - start_time
            
            if error:
                logger.error(f"API error for document {doc_id}: {error} (took {elapsed_time:.2f}s)")
                return None
                
            if not response or not response.choices:
                logger.error(f"No response content from DeepSeek API for document {doc_id} (took {elapsed_time:.2f}s)")
                return None
                
            logger.info(f"Successfully processed document {doc_id} in {elapsed_time:.2f}s")
            return response.choices[0].message.content.strip()
                
        except Exception as e:
            logger.error(f"Unexpected error processing document {doc_id}: {str(e)}")
            return None
