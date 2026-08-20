"""
AI Orchestrator: coordinates the full ticket-to-response pipeline.

This module replaces the direct generate_reply() call in main.py with a
structured pipeline that:

  1. Validates input (Responsible AI)
  2. Runs ML classification with confidence score
  3. Checks escalation conditions
  4. Retrieves relevant KB context (RAG)
  5. Constructs a structured prompt
  6. Calls the LLM (Groq llama-3.1-8b-instant)
  7. Evaluates the response (heuristic checks)
  8. Accepts, regenerates (once), or returns fallback
  9. Tracks latency and all events via monitoring module

Architecture:
  Input Ticket Text
        |
  [Responsible AI: validate_input()]
        |
  [ML: predict_with_confidence()]
        |
  [Responsible AI: should_escalate()]
        |
  [RAG: retriever.retrieve()]
        |
  [Prompt: build_support_prompt()]
        |
  [LLM: Groq API call]
        |
  [Evaluation: evaluate_response()]
        |
  [Accept / Regenerate once / Fallback]
        |
  OrchestratorResult

Design decisions:
  - One regeneration attempt maximum. Infinite retry loops add latency.
  - Fallback to static template on LLM failure — the endpoint never breaks.
  - Latency is tracked per LLM call and reported in the result.
  - All events are logged to monitoring.py counters for /analytics.
"""

import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

from app.evaluation import evaluate_response, EvaluationResult
from app.monitoring import (
    log_llm_event,
    log_rag_event,
    log_evaluation_event,
    log_escalation_event,
    counters,
)
from app.prompts.builder import build_support_prompt
from app.rag.retriever import KnowledgeRetriever, RetrievedChunk
from app.responsible_ai import validate_input, validate_output, should_escalate

logger = logging.getLogger(__name__)

# LLM configuration
LLM_MODEL = "llama-3.1-8b-instant"
LLM_MAX_TOKENS = 350
LLM_TEMPERATURE = 0.7
MAX_REGENERATION_ATTEMPTS = 1   # Maximum one retry

# Fallback reply used when LLM is unavailable or all attempts fail evaluation
FALLBACK_REPLY = (
    "Thank you for reaching out to us. We have received your ticket and our "
    "support team is reviewing it as a priority. You can expect a detailed "
    "response within 24 hours. If this is urgent, please reply to this message "
    "and we will escalate accordingly.\n\nBest regards,\nSupport Team"
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorResult:
    """
    The complete result of the AI orchestration pipeline.

    Fields:
        reply:              The final response text to send to the customer.
        is_ai_generated:    True if the reply came from the LLM.
        category:           ML-predicted ticket category.
        confidence:         ML classifier confidence score [0.0, 1.0].
        urgency:            Rule-based urgency.
        sentiment:          VADER sentiment.
        escalated:          True if ticket should be escalated to human.
        escalation_reason:  Why the ticket was escalated (if applicable).
        rag_chunks_used:    Number of KB chunks retrieved and injected.
        llm_latency_ms:     Total LLM time in milliseconds (sum of all attempts).
        evaluation:         EvaluationResult from the final accepted response.
        attempts:           How many LLM calls were made (1 or 2).
        prompt_template:    Name of the prompt template used.
        input_rejected:     True if Responsible AI rejected the input.
        rejection_reason:   Reason for input rejection if applicable.
    """
    reply: str
    is_ai_generated: bool
    category: str
    confidence: float
    urgency: str
    sentiment: str
    escalated: bool = False
    escalation_reason: str = ""
    rag_chunks_used: int = 0
    llm_latency_ms: float = 0.0
    evaluation: Optional[EvaluationResult] = None
    attempts: int = 0
    prompt_template: str = ""
    input_rejected: bool = False
    rejection_reason: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.evaluation:
            d["evaluation"] = self.evaluation.to_dict()
        return d


# ---------------------------------------------------------------------------
# LLM calling (thin wrapper — preserves original Groq integration)
# ---------------------------------------------------------------------------

def _call_llm(messages: list[dict]) -> tuple[str, float, bool]:
    """
    Call the Groq LLM API.

    Returns:
        (response_text, latency_ms, success)

    This is a thin wrapper around the Groq SDK. The original ai_reply.py
    integration is preserved — we call the same API with the same model.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not set — LLM call skipped.")
        return "", 0.0, False

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        start = time.perf_counter()

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        text = response.choices[0].message.content.strip()
        logger.info("LLM call success: %.1f ms, %d chars", latency_ms, len(text))
        return text, latency_ms, True

    except Exception as exc:
        logger.error("LLM call failed [%s]: %s", type(exc).__name__, str(exc)[:200])
        return "", 0.0, False


# ---------------------------------------------------------------------------
# Main orchestration function
# ---------------------------------------------------------------------------

def orchestrate(
    ticket_text: str,
    category: str,
    confidence: float,
    urgency: str,
    sentiment: str,
    retriever: Optional[KnowledgeRetriever] = None,
    cv_objects: Optional[list[dict]] = None,
) -> OrchestratorResult:
    """
    Execute the full AI orchestration pipeline for a ticket.

    Args:
        ticket_text: The raw customer ticket text.
        category: ML-predicted category (from predict_with_confidence).
        confidence: ML confidence score [0.0, 1.0].
        urgency: Rule-based urgency.
        sentiment: VADER sentiment.
        retriever: Initialized KnowledgeRetriever instance (optional).
                   If None, RAG step is skipped.
        cv_objects: Optional list of objects detected by Computer Vision, e.g., 
                    [{"label": "laptop", "confidence": 0.91}].

    Returns:
        OrchestratorResult with the final reply and all pipeline metadata.
    """

    # ---- Step 1: Input Validation ----
    input_validation = validate_input(ticket_text)
    if not input_validation.is_valid:
        counters.injection_attempts_total += 1
        logger.warning("Input rejected by Responsible AI: %s", input_validation.rejection_reason)
        return OrchestratorResult(
            reply=(
                "We could not process your request as submitted. "
                "Please resubmit a standard support ticket. "
                "If you believe this is an error, contact support directly."
            ),
            is_ai_generated=False,
            category=category,
            confidence=confidence,
            urgency=urgency,
            sentiment=sentiment,
            input_rejected=True,
            rejection_reason=input_validation.rejection_reason or "",
        )

    # ---- Step 2: Escalation Check ----
    escalated, escalation_reason = should_escalate(category, confidence, urgency, sentiment)
    if escalated:
        log_escalation_event(escalation_reason, category, urgency)

    # ---- Step 3: RAG Retrieval ----
    retrieved_chunks: list[RetrievedChunk] = []
    if retriever is not None and retriever.is_ready:
        retrieved_chunks = retriever.retrieve(ticket_text, category=category, top_k=3)
        top_score = retrieved_chunks[0].similarity_score if retrieved_chunks else 0.0
        log_rag_event(
            query_length=len(ticket_text),
            category=category,
            chunks_returned=len(retrieved_chunks),
            top_score=top_score,
        )

    # ---- Step 4: Prompt Construction ----
    chunk_contents = [c.content for c in retrieved_chunks]
    prompt_payload = build_support_prompt(
        ticket_text=ticket_text,
        category=category,
        urgency=urgency,
        sentiment=sentiment,
        retrieved_chunks=retrieved_chunks,
        cv_objects=cv_objects,
    )

    # ---- Step 5 + 6: LLM Call + Evaluation (with one retry) ----
    total_latency_ms = 0.0
    final_reply = ""
    final_eval: Optional[EvaluationResult] = None
    is_ai_generated = False
    attempts = 0

    for attempt in range(1 + MAX_REGENERATION_ATTEMPTS):
        attempts = attempt + 1
        response_text, latency_ms, llm_success = _call_llm(prompt_payload.to_messages())
        total_latency_ms += latency_ms

        log_llm_event(
            success=llm_success,
            latency_ms=latency_ms,
            model=LLM_MODEL,
            is_fallback=not llm_success,
            response_length=len(response_text),
        )

        if not llm_success or not response_text:
            logger.warning("LLM call failed on attempt %d — using fallback", attempts)
            break

        # ---- Output Validation (Responsible AI) ----
        output_validation = validate_output(response_text)
        if not output_validation.is_safe:
            counters.output_validation_failures_total += 1
            logger.warning(
                "Output validation failed on attempt %d: %s",
                attempts, output_validation.rejection_reason,
            )
            if attempt < MAX_REGENERATION_ATTEMPTS:
                logger.info("Regenerating response (attempt %d)...", attempts + 1)
                continue
            else:
                break  # All attempts failed output validation

        # ---- Response Evaluation ----
        eval_result = evaluate_response(
            response=response_text,
            ticket_text=ticket_text,
            retrieved_chunk_contents=chunk_contents,
        )
        log_evaluation_event(
            quality_score=eval_result.quality_score,
            passed=not eval_result.needs_regeneration,
            needs_regen=eval_result.needs_regeneration,
        )

        if eval_result.needs_regeneration and attempt < MAX_REGENERATION_ATTEMPTS:
            logger.info(
                "Response evaluation triggered regeneration (attempt %d): %s",
                attempts, eval_result.failure_reason,
            )
            continue

        # ---- Accept this response ----
        final_reply = response_text
        final_eval = eval_result
        is_ai_generated = True
        break

    # ---- Step 7: Fallback if no valid AI reply ----
    if not final_reply:
        final_reply = FALLBACK_REPLY
        is_ai_generated = False
        logger.info("Using static fallback reply after %d attempt(s)", attempts)

    return OrchestratorResult(
        reply=final_reply,
        is_ai_generated=is_ai_generated,
        category=category,
        confidence=confidence,
        urgency=urgency,
        sentiment=sentiment,
        escalated=escalated,
        escalation_reason=escalation_reason,
        rag_chunks_used=len(retrieved_chunks),
        llm_latency_ms=round(total_latency_ms, 1),
        evaluation=final_eval,
        attempts=attempts,
        prompt_template=prompt_payload.template_name,
    )
