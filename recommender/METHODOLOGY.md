# EU Legal Document Recommender System: Detailed Methodology

## Overview

This document provides a comprehensive explanation of the methodology used in our EU legal document recommender system. The system employs a hybrid approach combining semantic text analysis with categorical feature matching to provide personalized document recommendations.

## 1. Text Representation

### 1.1 BERT Embeddings

We use the all-MiniLM-L6-v2 model from sentence-transformers for generating document embeddings. This model was chosen for several reasons:
- Strong performance on semantic similarity tasks
- Efficient computation (distilled from larger models)
- Good balance between model size and accuracy
- Native support for multiple languages

The embedding process:
1. Text preprocessing
   ```python
   def preprocess_text(text: str) -> str:
       # Remove special characters
       text = re.sub(r'[^\w\s]', ' ', text)
       # Normalize whitespace
       text = ' '.join(text.split())
       return text
   ```

2. Embedding generation
   - Dimension: 384 (model's native dimension)
   - Normalized using L2 normalization
   - Batch processing for efficiency

### 1.2 Document Representation

Each document is represented by a combination of:

1. Summary embedding (70% weight)
   - Generated from our multi-tier summarization pipeline
   - Captures main document content and purpose
   - Weight: 0.7 in final embedding

2. Keyword embedding (30% weight)
   - Keywords extracted using KeyBERT
   - Concatenated with spaces between
   - Weight: 0.3 in final embedding

Mathematical representation:
```
doc_embedding = normalize(0.7 * summary_embedding + 0.3 * keyword_embedding)
```

## 2. Feature Engineering

### 2.1 Categorical Features

Features are processed using one-hot encoding with the following structure:

1. Document Type
   - Regulation
   - Directive
   - Decision
   - Other

2. Subject Matter (multi-hot encoded)
   - Environment
   - Transport
   - Energy
   - Finance
   - Agriculture
   - etc.

3. Geographic Scope
   - EU-wide
   - Member States
   - Third Countries
   - Regional

4. Legal Basis
   - TFEU Articles
   - Previous Regulations/Directives
   - International Agreements

### 2.2 Feature Vector Construction

```python
def construct_feature_vector(features: Dict[str, Any]) -> np.ndarray:
    # One-hot encode each feature
    encoded_features = []
    for feature_name, value in features.items():
        encoder = one_hot_encoders[feature_name]
        encoded = encoder.transform([[value]])
        encoded_features.append(encoded)
    
    # Concatenate all features
    return np.concatenate(encoded_features, axis=1)
```

## 3. Similarity Computation

### 3.1 Text Similarity

We use cosine similarity for comparing text embeddings:

```python
def compute_text_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
```

### 3.2 Categorical Similarity

For categorical features, we use a weighted Jaccard similarity:

```python
def compute_categorical_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
    intersection = np.sum(np.logical_and(feat1, feat2))
    union = np.sum(np.logical_or(feat1, feat2))
    return intersection / union if union > 0 else 0.0
```

### 3.3 Combined Similarity

The final similarity score is a weighted combination:

```python
final_similarity = (
    0.7 * text_similarity +
    0.3 * categorical_similarity
)
```

## 4. Efficient Similarity Search

### 4.1 FAISS Implementation

We use FAISS (Facebook AI Similarity Search) for efficient similarity computation:

1. Index Creation
   ```python
   dimension = text_embedding_dim + categorical_dim
   index = faiss.IndexFlatIP(dimension)  # Inner product similarity
   ```

2. Combined Vector Construction
   ```python
   combined_vector = np.concatenate([
       0.7 * normalized_text_embedding,
       0.3 * normalized_categorical_features
   ])
   ```

3. Search Implementation
   ```python
   def find_similar(query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
       distances, indices = index.search(query_vector.reshape(1, -1), k)
       return indices[0], distances[0]
   ```

### 4.2 Optimization Techniques

1. Batch Processing
   - Documents are processed in batches
   - Configurable batch size based on available memory
   - GPU utilization when available

2. Caching System
   ```python
   class EmbeddingCache:
       def __init__(self, cache_dir: Path):
           self.cache_dir = cache_dir
           self.cache = {}
           
       def get(self, doc_id: str) -> Optional[np.ndarray]:
           if doc_id in self.cache:
               return self.cache[doc_id]
           
           cache_file = self.cache_dir / f"{doc_id}.npy"
           if cache_file.exists():
               embedding = np.load(cache_file)
               self.cache[doc_id] = embedding
               return embedding
           return None
   ```

## 5. Ranking Algorithm

### 5.1 Base Ranking

Documents are initially ranked by combined similarity score:

```python
def rank_documents(query_vector: np.ndarray, 
                  documents: List[Dict],
                  k: int) -> List[Tuple[str, float]]:
    # Get similar documents using FAISS
    indices, distances = index.search(query_vector.reshape(1, -1), k)
    
    # Convert to similarity scores (distance is inner product)
    similarities = (distances + 1) / 2  # Convert to [0,1] range
    
    return list(zip([documents[i]['id'] for i in indices], similarities))
```

### 5.2 Ranking Refinements

1. Temporal Boost
   ```python
   temporal_boost = exp(-lambda * days_since_publication)
   final_score = similarity_score * temporal_boost
   ```

2. Relevance Threshold
   ```python
   MIN_SIMILARITY = 0.3
   recommendations = [
       (doc_id, score) for doc_id, score in ranked_docs
       if score >= MIN_SIMILARITY
   ]
   ```

## 6. Performance Optimization

### 6.1 Memory Management

1. Streaming Processing
   ```python
   def process_documents_stream(doc_iterator: Iterator[Dict],
                              batch_size: int = 32):
       for batch in batched(doc_iterator, batch_size):
           embeddings = generate_embeddings(batch)
           update_index(embeddings)
           clear_batch_memory()
   ```

2. GPU Memory Optimization
   ```python
   def optimize_batch_size(available_memory: int,
                          embedding_size: int) -> int:
       return min(
           32,  # Default batch size
           available_memory // (embedding_size * 4)  # 4 bytes per float
       )
   ```

### 6.2 Computation Optimization

1. Parallel Processing
   ```python
   def parallel_process_documents(documents: List[Dict],
                                n_workers: int = 4):
       with ProcessPoolExecutor(max_workers=n_workers) as executor:
           futures = [
               executor.submit(process_document, doc)
               for doc in documents
           ]
           return [f.result() for f in futures]
   ```

2. Incremental Updates
   ```python
   def update_index(new_documents: List[Dict]):
       new_vectors = process_documents(new_documents)
       index.add(new_vectors)
   ```

## 7. Evaluation Metrics

### 7.1 Relevance Metrics

1. Mean Reciprocal Rank (MRR)
   ```python
   def mrr(relevant_docs: Set[str], ranked_docs: List[str]) -> float:
       for i, doc_id in enumerate(ranked_docs, 1):
           if doc_id in relevant_docs:
               return 1.0 / i
       return 0.0
   ```

2. Normalized Discounted Cumulative Gain (NDCG)
   ```python
   def ndcg(relevant_docs: Set[str], ranked_docs: List[str], k: int) -> float:
       dcg = sum(
           1 / log2(i + 1) for i, doc in enumerate(ranked_docs[:k], 1)
           if doc in relevant_docs
       )
       idcg = sum(
           1 / log2(i + 1) for i in range(1, min(len(relevant_docs), k) + 1)
       )
       return dcg / idcg if idcg > 0 else 0.0
   ```

### 7.2 Performance Metrics

1. Response Time
   ```python
   def measure_response_time(func):
       @wraps(func)
       def wrapper(*args, **kwargs):
           start = time.perf_counter()
           result = func(*args, **kwargs)
           duration = time.perf_counter() - start
           return result, duration
       return wrapper
   ```

2. Memory Usage
   ```python
   def monitor_memory():
       process = psutil.Process()
       return process.memory_info().rss / 1024 / 1024  # MB
   ```

## 8. Future Improvements

1. Dynamic Weight Adjustment
   - Learn optimal weights from user feedback
   - Implement A/B testing framework
   - Personalized weight profiles

2. Enhanced Feature Engineering
   - Add document complexity metrics
   - Include user interaction history
   - Incorporate expert feedback

3. Model Improvements
   - Fine-tune BERT on legal documents
   - Implement multi-lingual support
   - Add domain-specific features

4. System Scalability
   - Implement distributed processing
   - Add real-time update capability
   - Optimize for larger document sets

## References

1. Sentence-Transformers: https://www.sbert.net/
2. FAISS: https://github.com/facebookresearch/faiss
3. KeyBERT: https://github.com/MaartenGr/KeyBERT
