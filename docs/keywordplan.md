# Dynamic Keyword Generation for Legal Documents

This document explains how to generate a dynamic number of keywords based on the length of a document using KeyBERT with a lightweight transformer model (e.g., DistilBERT). We use a mathematical function to determine the number of keywords (`top_n`) extracted from a document, ensuring that shorter documents yield fewer keywords while longer documents are capped at a maximum number.

---

## Mathematical Function for Keyword Count

We define the number of keywords to extract as:

\[
\text{top\_n}(L) = \text{clamp}\Big( a \times \ln(L) + b, \, \text{min\_keywords}, \, \text{max\_keywords} \Big)
\]

Where:
- \(L\) is the document length (in words).
- \(\ln(L)\) is the natural logarithm of the document length.
- \(a\) and \(b\) are scaling constants.
- `clamp` restricts the output between a minimum and maximum number of keywords.

### Example Parameters
- \(a = 2.5\)
- \(b = -4.77\)
- \(\text{min\_keywords} = 2\)
- \(\text{max\_keywords} = 20\)

### Example Calculations

- **15-word document:**  
  \(\ln(15) \approx 2.71\)  
  \(2.5 \times 2.71 - 4.77 \approx 2.00\) → **2 keywords**

- **150-word document:**  
  \(\ln(150) \approx 5.01\)  
  \(2.5 \times 5.01 - 4.77 \approx 7.76\) → **8 keywords (rounded)**

- **600-word document:**  
  \(\ln(600) \approx 6.40\)  
  \(2.5 \times 6.40 - 4.77 \approx 11.23\) → **11 keywords**

- **1500-word document:**  
  \(\ln(1500) \approx 7.31\)  
  \(2.5 \times 7.31 - 4.77 \approx 13.51\) → **14 keywords**

- **20,000-word document:**  
  \(\ln(20000) \approx 9.90\)  
  \(2.5 \times 9.90 - 4.77 \approx 19.98\) → **20 keywords** (capped)

- **Documents >20,000 words:**  
  The function continues to increase, but the result is clamped to a maximum of **20 keywords**.

---

## Implementation Using KeyBERT with DistilBERT

KeyBERT leverages transformer-based embeddings to extract the most relevant keywords from a text. In this approach, we use a lightweight model such as `distilbert-base-nli-mean-tokens` (or similar) to generate fast, high-quality embeddings.

### Steps

1. **Determine Document Length:**  
   Count the number of words in the document.

2. **Compute `top_n`:**  
   Use the mathematical function defined above to calculate the number of keywords to extract:
   
3. **Keyword Extraction with KeyBERT:**  We use a DistilBERT-based model for efficiency. DistilBERT is a distilled version of BERT, which is faster and lighter while retaining much of the performance.

### Code Implementation

```python
from keybert import KeyBERT
import numpy as np
from typing import List, Tuple

class DynamicKeywordExtractor:
    def __init__(self, 
                 a: float = 2.5,
                 b: float = -4.77,
                 min_keywords: int = 2,
                 max_keywords: int = 20,
                 model_name: str = 'distilbert-base-nli-mean-tokens'):
        self.a = a
        self.b = b
        self.min_keywords = min_keywords
        self.max_keywords = max_keywords
        self.kw_model = KeyBERT(model=model_name)
    
    def calculate_num_keywords(self, doc_length: int) -> int:
        """Calculate number of keywords based on document length."""
        num = self.a * np.log(doc_length) + self.b
        return int(np.clip(num, self.min_keywords, self.max_keywords))
    
    def extract_keywords(self, 
                        text: str,
                        ngram_range: Tuple[int, int] = (1, 2),
                        stop_words: str = 'english',
                        use_maxsum: bool = True,
                        diversity: float = 0.7) -> List[Tuple[str, float]]:
        """Extract keywords with dynamic count based on text length."""
        # Count words (approximate)
        word_count = len(text.split())
        
        # Calculate number of keywords
        top_n = self.calculate_num_keywords(word_count)
        
        # Extract keywords
        keywords = self.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=ngram_range,
            stop_words=stop_words,
            top_n=top_n,
            use_maxsum=use_maxsum,
            diversity=diversity
        )
        
        return keywords

# Example usage
extractor = DynamicKeywordExtractor()

# Example documents
short_doc = "This is a very short legal document about trade."
medium_doc = "..." # 500 words
long_doc = "..."   # 5000 words

# Extract keywords
short_keywords = extractor.extract_keywords(short_doc)
medium_keywords = extractor.extract_keywords(medium_doc)
long_keywords = extractor.extract_keywords(long_doc)
```

### Additional Features

1. **Diversity Control:**
   - The `diversity` parameter (0.0-1.0) controls how different the keywords should be
   - Higher values produce more diverse but potentially less relevant keywords

2. **N-gram Range:**
   - Supports both single words and phrases
   - Default range (1,2) allows both unigrams and bigrams

3. **MaxSum Algorithm:**
   - Uses MMR (Maximal Marginal Relevance) for keyword diversity
   - Balances relevance with diversity

### Performance Considerations

1. **Model Loading:**
   - Load model once and reuse for multiple documents
   - Consider using GPU if available for larger batches

2. **Batch Processing:**
   - Process documents in batches for better efficiency
   - Use multiprocessing for large document collections

3. **Memory Management:**
   - Clear GPU memory between large batches if needed
   - Monitor RAM usage for very large documents

---
