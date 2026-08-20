"""
Tests for the prompt engineering modules
(app/prompts/templates.py and app/prompts/builder.py).

Tests cover:
  - Templates are correctly structured (frozen dataclasses)
  - Builder returns correct PromptPayload
  - RAG chunks are injected into the prompt
  - Fallback template used when no chunks
  - to_messages() returns correct format
"""

import pytest

from app.prompts.templates import (
    SUPPORT_REPLY_V1,
    SUPPORT_REPLY_FALLBACK_V1,
    PromptTemplate,
)
from app.prompts.builder import build_support_prompt, PromptPayload
from app.rag.retriever import RetrievedChunk


# ---------------------------------------------------------------------------
# Helper: create sample RetrievedChunks for testing
# ---------------------------------------------------------------------------

def _make_chunk(doc_id: str, title: str, content: str,
                category: str) -> RetrievedChunk:
    return RetrievedChunk(
        doc_id=doc_id,
        title=title,
        content=content,
        category=category,
        similarity_score=0.8,
        boosted_score=0.9,
        category_matched=True,
    )


# ---------------------------------------------------------------------------
# Template tests
# ---------------------------------------------------------------------------

class TestPromptTemplates:
    def test_support_reply_v1_is_prompt_template(self):
        assert isinstance(SUPPORT_REPLY_V1, PromptTemplate)

    def test_support_reply_v1_has_version(self):
        assert SUPPORT_REPLY_V1.version == "v1"

    def test_support_reply_v1_system_prompt_not_empty(self):
        assert len(SUPPORT_REPLY_V1.system_prompt) > 100

    def test_support_reply_v1_has_safety_rules(self):
        assert "SAFETY RULES" in SUPPORT_REPLY_V1.system_prompt

    def test_support_reply_v1_mentions_no_fabrication(self):
        assert "NOT invent" in SUPPORT_REPLY_V1.system_prompt or "not make up" in SUPPORT_REPLY_V1.system_prompt.lower()

    def test_fallback_template_has_version(self):
        assert SUPPORT_REPLY_FALLBACK_V1.version == "v1"

    def test_fallback_template_system_prompt_not_empty(self):
        assert len(SUPPORT_REPLY_FALLBACK_V1.system_prompt) > 50

    def test_templates_are_frozen_immutable(self):
        with pytest.raises((AttributeError, TypeError)):
            SUPPORT_REPLY_V1.version = "v99"  # type: ignore


# ---------------------------------------------------------------------------
# Prompt builder tests
# ---------------------------------------------------------------------------

class TestBuildSupportPrompt:
    @pytest.fixture
    def sample_chunks(self):
        return [
            _make_chunk(
                "billing_001",
                "How to Request a Refund",
                "Contact billing within 30 days. Refunds take 5-7 days.",
                "Billing",
            ),
        ]

    def test_returns_prompt_payload(self, sample_chunks):
        result = build_support_prompt(
            ticket_text="I was charged twice this month",
            category="Billing",
            urgency="High",
            sentiment="Negative",
            retrieved_chunks=sample_chunks,
        )
        assert isinstance(result, PromptPayload)

    def test_system_message_not_empty(self, sample_chunks):
        result = build_support_prompt(
            ticket_text="I was charged twice this month",
            category="Billing",
            urgency="Medium",
            sentiment="Neutral",
            retrieved_chunks=sample_chunks,
        )
        assert len(result.system_message) > 50

    def test_user_message_contains_ticket_text(self, sample_chunks):
        ticket = "I cannot connect to the API"
        result = build_support_prompt(
            ticket_text=ticket,
            category="Technical",
            urgency="High",
            sentiment="Negative",
            retrieved_chunks=sample_chunks,
        )
        assert ticket in result.user_message

    def test_user_message_contains_category(self, sample_chunks):
        result = build_support_prompt(
            ticket_text="Some ticket",
            category="Billing",
            urgency="Low",
            sentiment="Neutral",
            retrieved_chunks=sample_chunks,
        )
        assert "Billing" in result.user_message

    def test_rag_chunks_are_injected(self, sample_chunks):
        result = build_support_prompt(
            ticket_text="I need a refund",
            category="Billing",
            urgency="Medium",
            sentiment="Neutral",
            retrieved_chunks=sample_chunks,
        )
        assert result.rag_chunks_used == 1
        assert "How to Request a Refund" in result.user_message

    def test_fallback_template_used_when_no_chunks(self):
        result = build_support_prompt(
            ticket_text="Some ticket about something",
            category="General",
            urgency="Low",
            sentiment="Neutral",
            retrieved_chunks=[],  # No chunks
        )
        assert result.rag_chunks_used == 0
        assert result.template_name == SUPPORT_REPLY_FALLBACK_V1.name

    def test_rag_template_used_when_chunks_available(self, sample_chunks):
        result = build_support_prompt(
            ticket_text="Billing issue",
            category="Billing",
            urgency="High",
            sentiment="Negative",
            retrieved_chunks=sample_chunks,
        )
        assert result.template_name == SUPPORT_REPLY_V1.name

    def test_high_urgency_adds_note_to_prompt(self, sample_chunks):
        result = build_support_prompt(
            ticket_text="Urgent billing issue",
            category="Billing",
            urgency="High",
            sentiment="Positive",
            retrieved_chunks=sample_chunks,
        )
        assert "HIGH URGENCY" in result.user_message

    def test_to_messages_returns_correct_format(self, sample_chunks):
        result = build_support_prompt(
            ticket_text="Billing question",
            category="Billing",
            urgency="Low",
            sentiment="Neutral",
            retrieved_chunks=sample_chunks,
        )
        messages = result.to_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert len(messages[0]["content"]) > 0
        assert len(messages[1]["content"]) > 0
