# AI Support Platform — RAG Pipeline

Retrieval-Augmented Generation (RAG) is used to ground the LLM's responses in factual, domain-specific knowledge, reducing hallucinations and providing accurate support steps.

## 1. Knowledge Base

The knowledge base (`data/knowledge_base/`) consists of static JSON files representing curated support articles. Each article (`KBDocument`) contains:
- `doc_id`: Unique identifier.
- `title`: Article title.
- `content`: The core support instructions.
- `category`: The domain category (Billing, Technical, Account, General).
- `tags`: Keywords associated with the document.

The knowledge base is loaded into memory during application startup.

## 2. Retriever (`app/rag/retriever.py`)

We implemented a custom, lightweight TF-IDF-based retriever. While vector databases (like Pinecone or Chroma) and dense embeddings are popular, for a focused support knowledge base, TF-IDF is highly effective, requires no external infrastructure, and has near-zero latency.

### Retrieval Mechanism
1. **Indexing**: On startup, all knowledge base documents are combined into full-text strings (Title + Content + Tags) and vectorized using a `TfidfVectorizer`.
2. **Querying**: When a ticket arrives, its text is vectorized.
3. **Similarity**: We compute the Cosine Similarity between the ticket vector and all document vectors in the index.

### Category Boosting
To improve retrieval accuracy, the retriever integrates tightly with the ML classifier. 
When retrieving chunks, the Orchestrator passes the ML-predicted `Category` to the retriever. The retriever applies a **Boost Factor** (e.g., +0.2) to the similarity score of any document whose category matches the predicted category. 
This ensures that a technical ticket about a "crash" pulls from the Technical articles rather than a coincidental keyword match in a Billing article.

### Thresholding
Only documents exceeding a minimum similarity threshold (`MIN_SIMILARITY_THRESHOLD = 0.1`) are returned. If no documents meet the threshold, the system gracefully falls back to the `SUPPORT_REPLY_FALLBACK_V1` prompt template.
