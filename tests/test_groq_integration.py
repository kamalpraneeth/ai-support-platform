"""
Optional integration tests for real Groq API calls.
These tests only run if GROQ_API_KEY is available.
"""

import os
import pytest
from app.ai_orchestrator import orchestrate


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY is unavailable")
def test_real_groq_api_call_success():
    """Verify the real Groq API responds properly without a mock."""
    result = orchestrate(
        ticket_text="How do I reset my password?",
        category="Account",
        confidence=0.9,
        urgency="Low",
        sentiment="Neutral",
        retriever=None
    )

    assert result.is_ai_generated is True
    assert len(result.reply) > 20
    assert result.attempts >= 1
    assert result.llm_latency_ms > 0
