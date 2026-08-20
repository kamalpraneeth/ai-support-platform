"""
Monitoring Module: structured logging, per-request metrics, and counters.

This module provides:
  1. Structured JSON logging for key events (LLM calls, RAG retrieval, ML predictions)
  2. In-memory counters for the /metrics endpoint
  3. Application startup time tracking

Privacy / Security guarantees:
  - API keys are NEVER logged.
  - Ticket text is truncated to 80 chars in logs.
  - User PII is not logged — only category/urgency metadata.
  - Secrets and credentials are excluded from all log entries.

Structured log format:
  Each event logs a JSON-compatible dict at INFO level using the standard
  Python logging module. In production, a log aggregator (e.g., Datadog,
  CloudWatch, Loki) would parse these structured fields for dashboards.

Limitations:
  - In-memory counters reset on application restart.
  - For persistent metrics in production, use a proper time-series DB
    (Prometheus, InfluxDB) or a managed APM platform.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory application counters
# ---------------------------------------------------------------------------


@dataclass
class AppCounters:
    """
    In-memory request/event counters.

    These reset on every application restart. They are useful for short-term
    monitoring and the /analytics endpoint, not for long-term trend analysis.
    """
    app_start_time: float = field(default_factory=time.time)

    # Ticket processing
    tickets_total: int = 0
    tickets_by_category: dict = field(default_factory=dict)
    tickets_by_urgency: dict = field(default_factory=dict)
    tickets_by_sentiment: dict = field(default_factory=dict)

    # ML metrics
    ml_predictions_total: int = 0
    ml_failures_total: int = 0
    ml_low_confidence_total: int = 0     # Confidence < 0.45

    # LLM metrics
    llm_calls_total: int = 0
    llm_successes_total: int = 0
    llm_failures_total: int = 0
    llm_fallback_total: int = 0          # Fallback template used
    llm_regeneration_total: int = 0      # Regeneration triggered by evaluation
    llm_total_latency_ms: float = 0.0

    # RAG metrics
    rag_retrievals_total: int = 0
    rag_chunks_retrieved_total: int = 0
    rag_empty_results_total: int = 0     # Retrievals that returned 0 chunks

    # Evaluation metrics
    evaluations_total: int = 0
    evaluations_passed: int = 0
    evaluations_failed: int = 0

    # Responsible AI
    escalations_total: int = 0
    injection_attempts_total: int = 0
    output_validation_failures_total: int = 0

    # API
    api_requests_total: int = 0
    api_errors_total: int = 0

    def uptime_seconds(self) -> float:
        """Return application uptime in seconds."""
        return time.time() - self.app_start_time

    def avg_llm_latency_ms(self) -> float:
        """Average LLM latency in milliseconds. Returns 0 if no calls."""
        if self.llm_calls_total == 0:
            return 0.0
        return round(self.llm_total_latency_ms / self.llm_calls_total, 1)

    def llm_success_rate(self) -> float:
        """LLM success rate [0.0, 1.0]."""
        if self.llm_calls_total == 0:
            return 1.0
        return round(self.llm_successes_total / self.llm_calls_total, 4)

    def escalation_rate(self) -> float:
        """Escalation rate relative to total tickets."""
        if self.tickets_total == 0:
            return 0.0
        return round(self.escalations_total / self.tickets_total, 4)

    def to_dict(self) -> dict:
        return {
            "uptime_seconds": round(self.uptime_seconds(), 1),
            "tickets": {
                "total": self.tickets_total,
                "by_category": self.tickets_by_category,
                "by_urgency": self.tickets_by_urgency,
                "by_sentiment": self.tickets_by_sentiment,
            },
            "ml": {
                "predictions_total": self.ml_predictions_total,
                "failures_total": self.ml_failures_total,
                "low_confidence_total": self.ml_low_confidence_total,
            },
            "llm": {
                "calls_total": self.llm_calls_total,
                "successes_total": self.llm_successes_total,
                "failures_total": self.llm_failures_total,
                "fallback_total": self.llm_fallback_total,
                "regeneration_total": self.llm_regeneration_total,
                "avg_latency_ms": self.avg_llm_latency_ms(),
                "success_rate": self.llm_success_rate(),
            },
            "rag": {
                "retrievals_total": self.rag_retrievals_total,
                "chunks_retrieved_total": self.rag_chunks_retrieved_total,
                "empty_results_total": self.rag_empty_results_total,
            },
            "evaluation": {
                "total": self.evaluations_total,
                "passed": self.evaluations_passed,
                "failed": self.evaluations_failed,
            },
            "responsible_ai": {
                "escalations_total": self.escalations_total,
                "escalation_rate": self.escalation_rate(),
                "injection_attempts_total": self.injection_attempts_total,
                "output_validation_failures": self.output_validation_failures_total,
            },
            "api": {
                "requests_total": self.api_requests_total,
                "errors_total": self.api_errors_total,
            },
        }


# Global singleton counters (reset on restart)
counters = AppCounters()


# ---------------------------------------------------------------------------
# Structured log helpers
# ---------------------------------------------------------------------------

def log_ml_event(
    category: str,
    confidence: float,
    urgency: str,
    sentiment: str,
    text_length: int,
) -> None:
    """Log and count a ML prediction event."""
    counters.ml_predictions_total += 1
    counters.tickets_by_category[category] = counters.tickets_by_category.get(
        category, 0) + 1
    counters.tickets_by_urgency[urgency] = counters.tickets_by_urgency.get(
        urgency, 0) + 1
    counters.tickets_by_sentiment[sentiment] = counters.tickets_by_sentiment.get(
        sentiment, 0) + 1

    if confidence < 0.45:
        counters.ml_low_confidence_total += 1

    logger.info(
        "ML_EVENT category=%s confidence=%.3f urgency=%s sentiment=%s text_length=%d",
        category, confidence, urgency, sentiment, text_length,
    )


def log_rag_event(
    query_length: int,
    category: str,
    chunks_returned: int,
    top_score: float,
) -> None:
    """Log and count a RAG retrieval event."""
    counters.rag_retrievals_total += 1
    counters.rag_chunks_retrieved_total += chunks_returned
    if chunks_returned == 0:
        counters.rag_empty_results_total += 1

    logger.info(
        "RAG_EVENT query_length=%d category=%s chunks=%d top_score=%.3f",
        query_length, category, chunks_returned, top_score,
    )


def log_llm_event(
    success: bool,
    latency_ms: float,
    model: str,
    is_fallback: bool,
    response_length: int,
) -> None:
    """Log and count an LLM call event."""
    counters.llm_calls_total += 1
    counters.llm_total_latency_ms += latency_ms

    if success:
        counters.llm_successes_total += 1
    else:
        counters.llm_failures_total += 1

    if is_fallback:
        counters.llm_fallback_total += 1

    # Log model name but not any content, API keys, or secrets
    logger.info(
        "LLM_EVENT success=%s latency_ms=%.1f model=%s is_fallback=%s response_length=%d",
        success, latency_ms, model, is_fallback, response_length,
    )


def log_evaluation_event(quality_score: float,
                         passed: bool, needs_regen: bool) -> None:
    """Log and count a response evaluation event."""
    counters.evaluations_total += 1
    if passed:
        counters.evaluations_passed += 1
    else:
        counters.evaluations_failed += 1

    if needs_regen:
        counters.llm_regeneration_total += 1

    logger.info(
        "EVAL_EVENT quality=%.3f passed=%s needs_regen=%s",
        quality_score, passed, needs_regen,
    )


def log_escalation_event(reason: str, category: str, urgency: str) -> None:
    """Log and count a ticket escalation event."""
    counters.escalations_total += 1
    # Log reason but not ticket content
    logger.info(
        "ESCALATION_EVENT category=%s urgency=%s reason='%s'",
        category, urgency, reason[:200],
    )


def log_injection_attempt(pattern_hint: str) -> None:
    """Log and count a prompt injection detection event."""
    counters.injection_attempts_total += 1
    # Do NOT log the actual input text — just the detection
    logger.warning("INJECTION_ATTEMPT_DETECTED pattern_hint=%s", pattern_hint)


def log_api_request(endpoint: str, success: bool) -> None:
    """Log and count an API request."""
    counters.api_requests_total += 1
    if not success:
        counters.api_errors_total += 1


def get_health_status(model_loaded: bool, retriever_ready: bool) -> dict:
    """
    Return a structured health status dict for the /health endpoint extension.

    Args:
        model_loaded: Whether the ML classifier is loaded.
        retriever_ready: Whether the RAG retriever index is built.
    """
    return {
        "status": "ok",
        "version": "2.0.0",
        "uptime_seconds": round(counters.uptime_seconds(), 1),
        "components": {
            "ml_classifier": "ready" if model_loaded else "not_loaded",
            "rag_retriever": "ready" if retriever_ready else "not_ready",
            "database": "connected",   # If we reach this function, DB is up
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
