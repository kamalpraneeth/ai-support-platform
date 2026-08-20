"""
Knowledge Retriever: TF-IDF cosine similarity search over the knowledge base.

Architecture:
  - Uses sklearn's TfidfVectorizer (already a project dependency) to build
    a vector index over knowledge-base documents.
  - Retrieval uses cosine similarity between the query vector and each document
    vector, returning the top-K most similar documents.
  - Category boost: documents matching the predicted ticket category receive
    a configurable similarity boost, making retrieval category-aware.

Why TF-IDF instead of a vector database (FAISS/Chroma/Qdrant)?
  - The knowledge base has ~30 documents. A vector DB adds significant infra
    complexity for marginal benefit at this scale.
  - TF-IDF is already a project dependency (sklearn), so no new packages needed.
  - The approach is fully explainable in an interview: term overlap + category
    filtering is transparent and deterministic.
  - For a production system with thousands of documents, migrating to sentence
    embeddings + FAISS/Chroma would be the natural next step.

Limitations (documented honestly):
  - TF-IDF does not capture semantic similarity (synonyms, paraphrasing).
    A query about "I cannot get in" may not match "login issues" as well as
    a dense embedding model would.
  - Category boost is a heuristic. If the classifier is wrong, the boost
    may surface less relevant documents.
"""

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.rag.knowledge_base import KBDocument

logger = logging.getLogger(__name__)

# Default category boost applied to documents whose category matches
# the predicted ticket category. This shifts the ranking toward on-topic docs.
DEFAULT_CATEGORY_BOOST = 0.15

# Minimum similarity score for a document to be considered relevant.
# Documents below this threshold are not returned, even if they are top-K.
MIN_SIMILARITY_THRESHOLD = 0.05


@dataclass
class RetrievedChunk:
    """A retrieved knowledge base document with its similarity score."""
    doc_id: str
    title: str
    content: str
    category: str
    similarity_score: float     # Raw cosine similarity [0.0, 1.0]
    boosted_score: float        # Score after category boost (used for ranking)
    category_matched: bool      # Whether this doc's category matched the query

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content,
            "category": self.category,
            "similarity_score": round(self.similarity_score, 4),
            "boosted_score": round(self.boosted_score, 4),
            "category_matched": self.category_matched,
        }


class KnowledgeRetriever:
    """
    TF-IDF-based retriever over the knowledge base.

    Usage:
        retriever = KnowledgeRetriever()
        retriever.build_index(documents)
        chunks = retriever.retrieve("I cannot login", category="Account", top_k=3)
    """

    def __init__(self, category_boost: float = DEFAULT_CATEGORY_BOOST) -> None:
        self._documents: list[KBDocument] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._doc_matrix = None          # shape: (n_docs, n_features)
        self._is_built: bool = False
        self.category_boost = category_boost

    def build_index(self, documents: list[KBDocument]) -> None:
        """
        Build the TF-IDF index over the provided documents.

        Args:
            documents: List of KBDocument objects from the knowledge base loader.

        This method is called once at application startup (lifespan event).
        """
        if not documents:
            logger.warning("KnowledgeRetriever: no documents provided. Retrieval will return empty results.")
            self._is_built = True
            return

        self._documents = documents
        corpus = [doc.full_text for doc in documents]

        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )
        self._doc_matrix = self._vectorizer.fit_transform(corpus)
        self._is_built = True

        logger.info(
            "KnowledgeRetriever index built: %d documents, vocabulary size=%d",
            len(documents),
            len(self._vectorizer.vocabulary_),
        )

    def retrieve(
        self,
        query: str,
        category: str | None = None,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-K most relevant knowledge base documents for a query.

        Args:
            query: The ticket text or a reformulated search query.
            category: The predicted ticket category. Documents in this category
                      receive a similarity boost to rank them higher.
            top_k: Maximum number of documents to return.

        Returns:
            List of RetrievedChunk objects, ordered by boosted similarity score
            (highest first). May return fewer than top_k if similarity scores
            fall below MIN_SIMILARITY_THRESHOLD.

        Methodology (documented for transparency):
            1. Vectorize the query using the fitted TF-IDF vocabulary.
            2. Compute cosine similarity between query vector and all doc vectors.
            3. Apply a category boost to documents matching the predicted category.
            4. Rank by boosted score and return top-K above the threshold.

        Limitations:
            TF-IDF similarity measures term overlap, not semantic meaning.
            Paraphrased queries may not retrieve the best document if they use
            different vocabulary from the knowledge base article.
        """
        if not self._is_built:
            raise RuntimeError("KnowledgeRetriever.build_index() must be called before retrieve().")

        if not self._documents or self._vectorizer is None:
            return []

        # Vectorize the query
        query_vector = self._vectorizer.transform([query])

        # Compute cosine similarity
        similarities = cosine_similarity(query_vector, self._doc_matrix)[0]

        # Build scored results
        scored: list[tuple[float, float, bool, int]] = []  # (boosted, raw, matched, idx)
        for idx, (doc, raw_score) in enumerate(zip(self._documents, similarities)):
            raw_score = float(raw_score)
            category_matched = (category is not None and doc.category == category)
            boost = self.category_boost if category_matched else 0.0
            boosted_score = min(1.0, raw_score + boost)
            scored.append((boosted_score, raw_score, category_matched, idx))

        # Sort by boosted score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Select top-K above threshold
        results: list[RetrievedChunk] = []
        for boosted_score, raw_score, category_matched, idx in scored[:top_k]:
            if raw_score < MIN_SIMILARITY_THRESHOLD:
                break  # Below minimum relevance threshold
            doc = self._documents[idx]
            results.append(
                RetrievedChunk(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    content=doc.content,
                    category=doc.category,
                    similarity_score=raw_score,
                    boosted_score=boosted_score,
                    category_matched=category_matched,
                )
            )

        logger.debug(
            "RAG retrieve: query='%s...' category=%s -> %d chunks returned (top_k=%d)",
            query[:40], category, len(results), top_k,
        )
        return results

    @property
    def document_count(self) -> int:
        """Number of documents in the index."""
        return len(self._documents)

    @property
    def is_ready(self) -> bool:
        """True if the index has been built and is ready for queries."""
        return self._is_built
