# EU Legal Document Recommender System: Detailed Methodology

## Overview

This document provides a comprehensive explanation of the methodology used in our EU legal document recommender system. The system employs a hybrid approach combining semantic text analysis with categorical feature matching to provide personalized document recommendations. The system leverages Pinecone vector database for efficient similarity search and retrieval.

## 1. Text Representation

### 1.1 Legal-BERT Embeddings

We use the nlpaueb/legal-bert-small-uncased model from sentence-transformers for generating document embeddings. This model was chosen for several reasons:
- Domain-specific training on legal documents
- Strong performance on legal semantic similarity tasks
- Optimized for legal terminology and concepts
- Support for multiple European languages

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
   - Dimension: 512 (model's native dimension)
   - Normalized using L2 normalization
   - Batch processing for efficiency

### 1.2 Document Representation

Each document is represented using three distinct embeddings:

1. **Summary embedding**
   - Generated from our multi-tier summarization pipeline
   - Captures main document content and purpose
   - Stored separately with ID format: `{document_id}_summary`

2. **Keyword embedding**
   - Keywords extracted using KeyBERT
   - Concatenated with spaces between
   - Stored separately with ID format: `{document_id}_keyword`

3. **Combined embedding (primary)**
   - Weighted combination of summary and keyword embeddings
   - Weights dynamically adjusted based on number of keywords
   - Stored as the primary embedding with ID format: `{document_id}`

For the combined embedding, the weights are dynamically adjusted based on the number n of keywords extracted:

- For shorter documents with fewer keywords (n=2), the summary embedding gets more weight (0.9)
- For longer documents with more keywords (n=20), the keyword embedding gets more weight (0.4)

Mathematical representation:

```
w_keyword(n) = 0.1 + 0.01667 × (n - 2)
w_summary(n) = 1 - w_keyword(n)

combined_embedding = normalize(w_summary(n) * summary_embedding + w_keyword(n) * keyword_embedding)
```

This approach ensures that for shorter documents, where the summary likely captures most of the content, we rely more heavily on the summary embedding. For longer, more complex documents with many keywords, we give more weight to the keyword embedding to ensure important details are not overlooked.

### 1.3 Flexible Query Strategies

Storing separate embeddings enables flexible query strategies:

1. **Combined embedding queries** (default)
   - Uses the dynamically weighted combination of summary and keyword embeddings
   - Provides balanced results considering both document summaries and specific keywords

2. **Summary-only queries**
   - Targets only the summary embeddings
   - Useful when looking for documents with similar overall content or purpose
   - Effective for conceptual or thematic searches

3. **Keyword-only queries**
   - Targets only the keyword embeddings
   - Useful for finding documents with specific terminology or technical content
   - Effective for precise, terminology-focused searches

Clients can specify which embedding type to use for their query, allowing them to tailor the search strategy to their specific needs.

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

Our system processes four key categorical variables from each document:

1. **Form**: Type of legal initiative (e.g., Regulation, Directive, Decision)
2. **Subject Matters**: Multi-valued, reflecting various thematic areas a document covers
3. **EuroVoc Descriptors**: Multi-valued official classification terms from the EU thesaurus
4. **Author**: Typically multi-valued (often two authors), such as European Parliament and a Directorate General

We use two encoding approaches:
- **Multi-hot encoding** for multi-valued features (Subject Matters, EuroVoc Descriptors)
- **One-hot encoding** for single-valued features (Form)

The implementation handles both types appropriately:

```python
def encode_features(features: Dict[str, Union[str, List[str]]]) -> np.ndarray:
    encoded_features = []
    
    for feature_name, encoder in encoders.items():
        value = features.get(feature_name, None)
        
        if value is None:
            # Use zero vector for missing features
            encoded = np.zeros(len(encoder.categories_[0]))
        elif feature_name in multi_valued_features and isinstance(value, list):
            # Multi-hot encoding for multi-valued features
            encoded = np.zeros(len(encoder.categories_[0]))
            for val in value:
                val_idx = np.where(encoder.categories_[0] == val)[0]
                if len(val_idx) > 0:
                    encoded[val_idx[0]] = 1
        else:
            # One-hot encoding for single-valued features
            encoded = encoder.transform([[value]])[0]
        
        encoded_features.append(encoded)
    
    # Concatenate all features
    return np.concatenate(encoded_features)
```

### 2.3 Categorical Variable Encoding

In addition to semantic embeddings, our system extracts several categorical features from each document and processes them using appropriate encoding methods. Four key categorical variables are considered:

1. **Form**: Type of legal initiative (e.g., Regulation, Directive, Decision)
2. **Subject Matters**: Multi-valued, reflecting the various thematic areas a document covers as provided by the EU
3. **EuroVoc Descriptors**: Multi-valued official classification terms from the EU thesaurus
4. **Author**: Typically multi-valued (often two authors), for example, European Parliament and the Directorate General of Agriculture

The features that can have multiple values (Subject Matters and EuroVoc Descriptors) are encoded using multi-hot encoding, while the single-valued features (Form) are processed using one-hot encoding. These encoded vectors are concatenated to form a comprehensive categorical feature vector that directly complements the document embedding during similarity computations.

### 2.4 Client Preference Weighting

Notably, while Subject Matters and EuroVoc Descriptors directly influence the similarity score through the categorical feature vector, the Form and Author features are treated differently. These latter variables can be weighted by the client based on their specific interests. Such weights adjust the final ranking of recommendations without directly impacting the computed similarity, allowing for personalized adjustments that reflect the client's preferences.

The implementation applies client preferences as a bonus to the similarity score:

```python
def apply_client_preferences(score, document_metadata, client_preferences):
    preference_bonus = 0.0
    
    # Apply form preference if available
    if 'form' in client_preferences and 'form' in document_metadata:
        form_value = document_metadata['form']
        form_weight = client_preferences.get('form', 0.0)
        if form_weight > 0:
            preference_bonus += form_weight
    
    # Apply author preference if available
    if 'author' in client_preferences and 'author' in document_metadata:
        author_value = document_metadata['author']
        author_weight = client_preferences.get('author', 0.0)
        if author_weight > 0:
            preference_bonus += author_weight
    
    # Add preference bonus to score
    return score + preference_bonus
```

This approach allows clients to express preferences for specific document types or authors without modifying the underlying similarity calculation.

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

### 4.1 Pinecone Implementation

We use Pinecone vector database for efficient similarity computation and retrieval:

1. Index Creation
   ```python
   # Initialize Pinecone client
   pc = Pinecone(api_key=api_key)
   
   # Create index if it doesn't exist
   if index_name not in pc.list_indexes().names():
       pc.create_index(
           name=index_name,
           dimension=512,  # Legal-BERT embedding dimension
           metric="cosine",
           spec=ServerlessSpec(cloud="aws", region="us-west-2")
       )
   ```

2. Vector Upsert with Metadata
   ```python
   # Prepare vectors with metadata
   vectors_with_metadata = []
   for i, (doc_id, embedding) in enumerate(zip(doc_ids, embeddings)):
       # Store categorical features as metadata
       metadata = {
           "type": doc_types[i],
           "subject": doc_subjects[i],
           "year": doc_years[i],
           "categorical_features": json.dumps(categorical_features[i].tolist())
       }
       
       vectors_with_metadata.append({
           "id": doc_id,
           "values": embedding.tolist(),
           "metadata": metadata
       })
   
   # Upsert vectors to Pinecone
   index.upsert(vectors=vectors_with_metadata)
   ```

3. Search Implementation
   ```python
   def find_similar(query_vector: np.ndarray, k: int, filter: Dict = None) -> List[Dict]:
       results = index.query(
           vector=query_vector.tolist(),
           top_k=k,
           include_metadata=True,
           filter=filter
       )
       return results.matches
   ```

### 4.2 Optimization Techniques

1. Batch Processing
   - Documents are processed in batches
   - Configurable batch size based on available memory
   - GPU utilization when available
   - Pinecone bulk upsert for efficient indexing

2. Metadata Filtering
   ```python
   # Filter by document type and year
   filter_dict = {
       "metadata": {
           "type": "regulation",
           "year": {"$gte": 2015}  # Documents from 2015 or later
       }
   }
   
   # Apply filter during search
   results = index.query(
       vector=query_vector.tolist(),
       top_k=k,
       include_metadata=True,
       filter=filter_dict
   )
   ```
   
3. Hybrid Search with Text and Categorical Features
   ```python
   # First search by text embedding
   results = index.query(
       vector=query_vector.tolist(),
       top_k=k * 2,  # Get more results for reranking
       include_metadata=True
   )
   
   # Rerank results using categorical features
   reranked_results = []
   for match in results.matches:
       # Extract categorical features from metadata
       doc_categorical = np.array(json.loads(match.metadata['categorical_features']))
       
       # Calculate combined score
       text_score = match.score
       cat_score = compute_categorical_similarity(query_categorical, doc_categorical)
       combined_score = 0.8 * text_score + 0.2 * cat_score
       
       reranked_results.append({
           'id': match.id,
           'score': combined_score,
           'metadata': match.metadata
       })
   
   # Sort by combined score
   reranked_results.sort(key=lambda x: x['score'], reverse=True)
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

1. Hybrid Scoring with Categorical Features
   ```python
   # Extract categorical features from metadata
   doc_categorical = np.array(json.loads(match.metadata['categorical_features']))
   
   # Calculate text and categorical similarity scores
   text_score = match.score  # Cosine similarity from Pinecone
   cat_score = np.dot(query_categorical, doc_categorical) / (
       np.linalg.norm(query_categorical) * np.linalg.norm(doc_categorical)
   ) if np.linalg.norm(doc_categorical) > 0 else 0.0
   
   # Combine scores with weights
   combined_score = 0.75 * text_score + 0.25 * cat_score
   ```

### 5.3 Similarity Computation and Ranking Calculation

When a similarity query is initiated, the system retrieves the semantic embeddings and categorical feature vectors that were previously stored during preprocessing. The similarity between the semantic embeddings is calculated with cosine similarity, while the similarity between the categorical variables (transformed into feature vectors) is calculated with a Jaccard measure. Finally, the overall similarity score between two documents is obtained by combining the semantic and categorical similarity scores using a weighted sum:

```
text_final_similarity = 0.75 × text_similarity + 0.25 × categorical_similarity
```

This weighted approach ensures that while semantic similarity (based on document content) remains the primary factor in determining relevance, categorical features (such as document type, subject matter, and author) also contribute meaningfully to the final ranking.

The implementation of this formula can be found in the `PineconeRecommender` class, where:

```python
# Combine scores with weights
score = self.text_weight * match.score + self.categorical_weight * cat_sim
```

With `self.text_weight` set to 0.75 and `self.categorical_weight` set to 0.25.

After the combined similarity score is calculated, client preferences for specific features (such as Form and Author) are applied as additional bonuses to the score, further refining the ranking based on client-specific requirements without altering the underlying similarity calculation.
   ```

2. Metadata Filtering
   ```python
   # Filter by document type and year
   filter_dict = {
       "metadata": {
           "type": "regulation",
           "year": {"$gte": 2015}  # Documents from 2015 or later
       }
   }
   
   # Apply filter during search
   results = index.query(
       vector=query_vector.tolist(),
       top_k=k,
       include_metadata=True,
       filter=filter_dict
   )
   ```

3. Relevance Threshold
   ```python
   MIN_SIMILARITY = 0.3
   recommendations = [
       rec for rec in recommendations
       if rec['score'] >= MIN_SIMILARITY
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
