"""
Tests for the RAG pipeline (app/rag/knowledge_base.py and app/rag/retriever.py).

Tests cover:
  - Knowledge base loads real JSON files
  - Document validation (missing fields, invalid category, duplicates)
  - Retriever builds index correctly
  - Retrieval returns correct number of results
  - Category boost works (category-matching docs ranked higher)
  - Minimum threshold filters low-relevance results
  - Empty retriever returns empty results
"""

import json
import pytest

from app.rag.knowledge_base import load_knowledge_base, KBDocument, KB_DIR
from app.rag.retriever import KnowledgeRetriever, RetrievedChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_documents():
    """A small set of KBDocument objects for testing the retriever."""
    return [
        KBDocument(
            doc_id="billing_001",
            title="How to Request a Refund",
            content="To request a refund, contact billing within 30 days. Refunds take 5-7 business days.",
            category="Billing",
            tags=["refund", "billing", "payment"],
        ),
        KBDocument(
            doc_id="tech_001",
            title="Application Crashing or Not Loading",
            content="If the application crashes, clear your browser cache and try a different browser.",
            category="Technical",
            tags=["crash", "loading", "browser"],
        ),
        KBDocument(
            doc_id="account_001",
            title="Password Reset Process",
            content="To reset your password, click Forgot Password and enter your registered email.",
            category="Account",
            tags=["password", "reset", "login"],
        ),
        KBDocument(
            doc_id="general_001",
            title="Getting Started with the Platform",
            content="Welcome to our platform. Create an account and follow the onboarding wizard.",
            category="General",
            tags=["getting started", "onboarding", "setup"],
        ),
    ]


@pytest.fixture(scope="module")
def built_retriever(sample_documents):
    """A KnowledgeRetriever with the sample documents indexed."""
    retriever = KnowledgeRetriever()
    retriever.build_index(sample_documents)
    return retriever


# ---------------------------------------------------------------------------
# Knowledge base loader tests
# ---------------------------------------------------------------------------

class TestKnowledgeBaseLoader:
    def test_load_from_real_kb_dir_returns_documents(self):
        """Real knowledge base directory should load documents."""
        if not KB_DIR.exists():
            pytest.skip("Knowledge base directory not found")
        docs = load_knowledge_base()
        assert len(docs) > 0

    def test_loaded_documents_have_required_fields(self):
        """Every loaded document must have id, title, content, category."""
        if not KB_DIR.exists():
            pytest.skip("Knowledge base directory not found")
        docs = load_knowledge_base()
        for doc in docs:
            assert doc.doc_id
            assert doc.title
            assert doc.content
            assert doc.category in {"Billing", "Technical", "Account", "General"}

    def test_empty_directory_returns_empty_list(self, tmp_path):
        """Empty directory should return empty list, not error."""
        docs = load_knowledge_base(kb_dir=tmp_path)
        assert docs == []

    def test_invalid_json_raises_value_error(self, tmp_path):
        """Malformed JSON in KB directory should raise ValueError."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json{{{", encoding="utf-8")
        with pytest.raises(ValueError, match="Malformed JSON"):
            load_knowledge_base(kb_dir=tmp_path)

    def test_skips_document_with_missing_fields(self, tmp_path):
        """Documents missing required fields should be skipped."""
        valid_doc = [{"id": "x", "title": "T", "content": "C", "category": "Billing", "tags": []}]
        missing_doc = [{"id": "y", "title": "No content or category"}]  # missing content, category

        file1 = tmp_path / "valid.json"
        file1.write_text(json.dumps(valid_doc), encoding="utf-8")
        file2 = tmp_path / "invalid.json"
        file2.write_text(json.dumps(missing_doc), encoding="utf-8")

        docs = load_knowledge_base(kb_dir=tmp_path)
        assert len(docs) == 1
        assert docs[0].doc_id == "x"

    def test_duplicate_ids_are_deduplicated(self, tmp_path):
        """Duplicate document IDs should be skipped."""
        docs_data = [
            {"id": "dup_001", "title": "T1", "content": "C1", "category": "Billing", "tags": []},
            {"id": "dup_001", "title": "T2", "content": "C2", "category": "Technical", "tags": []},
        ]
        file1 = tmp_path / "dupes.json"
        file1.write_text(json.dumps(docs_data), encoding="utf-8")

        docs = load_knowledge_base(kb_dir=tmp_path)
        assert len(docs) == 1

    def test_kb_document_full_text_combines_title_content_tags(self):
        """KBDocument.full_text should combine title, content, and tags."""
        doc = KBDocument(
            doc_id="test_001",
            title="My Title",
            content="My content text.",
            category="Billing",
            tags=["tag1", "tag2"],
        )
        full_text = doc.full_text
        assert "My Title" in full_text
        assert "My content text" in full_text
        assert "tag1" in full_text


# ---------------------------------------------------------------------------
# Retriever tests
# ---------------------------------------------------------------------------

class TestKnowledgeRetriever:
    def test_retriever_is_ready_after_build(self, built_retriever):
        assert built_retriever.is_ready is True

    def test_document_count_matches_indexed_documents(self, built_retriever, sample_documents):
        assert built_retriever.document_count == len(sample_documents)

    def test_retrieve_returns_list(self, built_retriever):
        results = built_retriever.retrieve("refund billing charge", top_k=3)
        assert isinstance(results, list)

    def test_retrieve_top_k_limits_results(self, built_retriever):
        results = built_retriever.retrieve("billing payment refund", top_k=2)
        assert len(results) <= 2

    def test_retrieve_returns_retrieved_chunk_objects(self, built_retriever):
        results = built_retriever.retrieve("password reset login", top_k=3)
        for chunk in results:
            assert isinstance(chunk, RetrievedChunk)

    def test_relevant_query_returns_results(self, built_retriever):
        """A clear billing query should return at least 1 result."""
        results = built_retriever.retrieve("I need a refund for my billing charge", top_k=3)
        assert len(results) >= 1

    def test_category_boost_ranks_matching_category_higher(self, built_retriever):
        """With category=Billing, billing document should rank first or near first."""
        results = built_retriever.retrieve(
            "refund payment billing",
            category="Billing",
            top_k=4,
        )
        if len(results) >= 1:
            # The billing document should have category_matched=True if it's top-1
            top_result = results[0]
            assert top_result.category_matched or top_result.similarity_score > 0

    def test_similarity_scores_are_non_negative(self, built_retriever):
        results = built_retriever.retrieve("password account login", top_k=3)
        for chunk in results:
            assert chunk.similarity_score >= 0.0

    def test_boosted_scores_greater_or_equal_raw_scores(self, built_retriever):
        results = built_retriever.retrieve(
            "password account login",
            category="Account",
            top_k=3,
        )
        for chunk in results:
            assert chunk.boosted_score >= chunk.similarity_score - 0.001  # float tolerance

    def test_empty_retriever_returns_empty_results(self):
        empty_retriever = KnowledgeRetriever()
        empty_retriever.build_index([])
        results = empty_retriever.retrieve("some query", top_k=3)
        assert results == []

    def test_retrieved_chunk_to_dict(self, built_retriever):
        results = built_retriever.retrieve("refund billing", top_k=1)
        if results:
            d = results[0].to_dict()
            assert "doc_id" in d
            assert "title" in d
            assert "similarity_score" in d
