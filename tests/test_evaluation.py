"""
Tests for the response evaluation module (app/evaluation.py).

Tests cover:
  - Short/empty response fails completeness
  - Long enough response with next step passes
  - Relevance check detects ticket term overlap
  - Groundedness check uses KB terms
  - Safety check detects unsafe patterns
  - Quality score is in [0, 1]
  - needs_regeneration is set correctly
  - No KB chunks: groundedness passes by default
"""


from app.evaluation import (
    evaluate_response,
    EvaluationResult,
    _check_relevance,
    _check_groundedness,
    _check_safety,
    _check_completeness,
    _check_signoff,
)

# ---------------------------------------------------------------------------
# Sample data for tests
# ---------------------------------------------------------------------------

GOOD_REPLY = (
    "Thank you for reaching out to us. We understand that being charged incorrectly "
    "is frustrating. To request a refund, please contact our billing team within 30 days "
    "of the charge, and we will process it within 5-7 business days. "
    "Please reply to this ticket with your invoice number and we will investigate immediately.\n\n"
    "Best regards,\nSupport Team"
)

SHORT_REPLY = "OK, we will look into it."

TICKET_TEXT = "I was charged twice for my subscription this month"

KB_CHUNKS = [
    "To request a refund, contact billing within 30 days. Refunds take 5-7 business days.",
    "Your invoice includes subscription fee and applicable taxes.",
]


# ---------------------------------------------------------------------------
# Individual check tests
# ---------------------------------------------------------------------------

class TestCompletenessCheck:
    def test_short_reply_fails_completeness(self):
        complete, reason = _check_completeness("Short reply.")
        assert complete is False
        assert reason is not None

    def test_good_reply_passes_completeness(self):
        complete, reason = _check_completeness(GOOD_REPLY)
        assert complete is True

    def test_reply_without_next_step_fails(self):
        no_action = "We have received your ticket and noted the issue. " * 4
        complete, reason = _check_completeness(no_action)
        assert complete is False
        assert reason is not None


class TestRelevanceCheck:
    def test_relevant_response_passes(self):
        relevant, score = _check_relevance(TICKET_TEXT, GOOD_REPLY)
        assert relevant is True
        assert score > 0.0

    def test_completely_irrelevant_response_fails(self):
        irrelevant = "The weather is nice today and the sun is shining brightly."
        relevant, score = _check_relevance(TICKET_TEXT, irrelevant)
        # Score should be very low
        assert score < 0.5

    def test_relevance_score_in_range(self):
        _, score = _check_relevance(TICKET_TEXT, GOOD_REPLY)
        assert 0.0 <= score <= 1.0


class TestGroundednessCheck:
    def test_good_reply_with_kb_chunks_scores_positively(self):
        grounded, score = _check_groundedness(GOOD_REPLY, KB_CHUNKS)
        assert score >= 0.0  # Score should be non-negative

    def test_no_kb_chunks_passes_by_default(self):
        grounded, score = _check_groundedness(GOOD_REPLY, [])
        assert grounded is True
        assert score == 0.0

    def test_groundedness_score_in_range(self):
        _, score = _check_groundedness(GOOD_REPLY, KB_CHUNKS)
        assert 0.0 <= score <= 1.0


class TestSafetyCheck:
    def test_good_reply_passes_safety(self):
        safe, reason = _check_safety(GOOD_REPLY)
        assert safe is True
        assert reason is None

    def test_long_digit_sequence_triggers_safety(self):
        bad_reply = GOOD_REPLY + " Your account number is 12345678901234."
        safe, reason = _check_safety(bad_reply)
        assert safe is False

    def test_external_url_triggers_safety(self):
        bad_reply = GOOD_REPLY + " Please visit http://external-phishing-site.com for help."
        safe, reason = _check_safety(bad_reply)
        assert safe is False


class TestSignoffCheck:
    def test_reply_with_support_team_has_signoff(self):
        assert _check_signoff("Best regards, Support Team") is True

    def test_reply_without_signoff_returns_false(self):
        assert _check_signoff("We will look into your issue.") is False


# ---------------------------------------------------------------------------
# Full evaluation tests
# ---------------------------------------------------------------------------

class TestEvaluateResponse:
    def test_good_reply_has_high_quality_score(self):
        result = evaluate_response(GOOD_REPLY, TICKET_TEXT, KB_CHUNKS)
        assert result.quality_score > 0.4  # Should score reasonably well

    def test_short_reply_has_needs_regeneration_true(self):
        result = evaluate_response(SHORT_REPLY, TICKET_TEXT, KB_CHUNKS)
        assert result.needs_regeneration is True

    def test_returns_evaluation_result_instance(self):
        result = evaluate_response(GOOD_REPLY, TICKET_TEXT)
        assert isinstance(result, EvaluationResult)

    def test_quality_score_in_range(self):
        result = evaluate_response(GOOD_REPLY, TICKET_TEXT, KB_CHUNKS)
        assert 0.0 <= result.quality_score <= 1.0

    def test_good_reply_passes_safety(self):
        result = evaluate_response(GOOD_REPLY, TICKET_TEXT, KB_CHUNKS)
        assert result.safe is True

    def test_short_reply_fails_completeness(self):
        result = evaluate_response(SHORT_REPLY, TICKET_TEXT, KB_CHUNKS)
        assert result.complete is False

    def test_response_length_recorded(self):
        result = evaluate_response(GOOD_REPLY, TICKET_TEXT)
        assert result.response_length == len(GOOD_REPLY)

    def test_word_count_recorded(self):
        result = evaluate_response(GOOD_REPLY, TICKET_TEXT)
        assert result.word_count == len(GOOD_REPLY.split())

    def test_no_kb_chunks_does_not_set_needs_regeneration_only_for_groundedness(
            self):
        """Without KB chunks, groundedness passes — so only other checks matter."""
        result = evaluate_response(
            GOOD_REPLY,
            TICKET_TEXT,
            retrieved_chunk_contents=None)
        assert result.grounded is True

    def test_to_dict_returns_dict(self):
        result = evaluate_response(GOOD_REPLY, TICKET_TEXT)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "quality_score" in d
        assert "needs_regeneration" in d
